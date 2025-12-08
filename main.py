import sys
import time
import os
import csv
import taichi as ti
import taichi.math as tm
from pathlib import Path

os.environ['TI_AOT_ONLY'] = '1'

def resource_path(relative_path: str) -> Path:
    try:
        base_path = Path(sys._MEIPASS)
        print("Yup")
    except Exception:
        print("Nah")
        base_path = Path(__file__).parent
    return base_path / relative_path


TABLE_1_PATH = resource_path("table1.csv")
TABLE_2_PATH = resource_path("table2.csv")

bounds = (900, 900)

N_BELTS = 18
TOLERANCE = 0.1
D_TEMP = 0.001
MAX_TRIES = 100
LEARNING_RATE = 0.24

# Unit Conversions
LY_PER_KLY = 1000.0
DYNES_PER_MB = 1000.0
SECONDS_PER_YEAR = 31557600.0


# Physics Constants (CGS)
STEFAN_BOLTZMANN = 1.356e-12    # ly / (sec * K^4)
GRAVITY_CM_S2 = 1e3             # cm / sec^2
GAS_CONSTANT_AIR = 6.8579e-2    # cal / (gm * K)
SPECIFIC_HEAT_AIR = 0.24        # cal / (gm * K)
LATENT_HEAT_WATER = 590.0       # cal / gm
MOL_WEIGHT_RATIO = 0.622        # epsilon (dimensionless)
EARTH_RADIUS_CM = 6.371e8       # cm

# Model Parameters
BELT_WIDTH_CM = 1.11e8          # Delta y (cm)
ATM_ATTENUATION = 0.5           # m (dimensionless)
REF_PRESSURE_MB = 1000.0        # p (mb)
REF_PRESSURE_DYNES = REF_PRESSURE_MB * DYNES_PER_MB # p (dynes/cm^2)


LY_PER_KLY = 1000.0
SECONDS_PER_YEAR = 31557600.0 # 365.25 * 24 * 3600
DYNES_PER_MB = 1000.0

@ti.dataclass
class LatBelt:
    # State Variables
    temp_kelvin: ti.f32             # T_0 [kelvin]
    albedo: ti.f32                   # a_s []

    # Geometry and location
    lat_bottom_deg: ti.f32          # latitude of southern boundary [deg]
    lat_top_deg: ti.f32             # latitude of northern boundary [deg]
    length_bottom_cm: ti.f32        # circumference of southern boundary (l_0) [cm]
    length_top_cm: ti.f32           # circumference of northern boundary (l_1) [cm]
    area_cm2: ti.f32                # surface area of belt (A_0) [cm^2]
    ocean_fraction: ti.f32          # fraction of width covered by ocean (f_ocean) [0.0-1.0]

    # Radiation parameters
    solar_flux_ly_sec: ti.f32       # Q_s [langleys/sec] 1 ly = 1 cal/cm^2
    albedo_coeff_b: ti.f32          # b [Dimensionless] (Empirical constant)
    elevation_meters: ti.f32        # Z [Meters] (Used with 0.0065 K/m lapse rate)

    # Transport physics
    exchange_coeff_a: ti.f32        # a [cm / (sec * K)] (scales wind speed)
    pressure_depth_dynes: ti.f32    # del_p [dynes/cm^2] (mass of atmospheric column)
    ocean_depth_cm: ti.f32          # del_z [cm] (depth of active ocean layer)

    # Eddy diffusivity coefficients
    diff_vapor_cm2_s: ti.f32        # K_w [cm^2/sec] (water vapor mixing)
    diff_air_cm2_s: ti.f32          # K_h [cm^2/sec] (thermal air mixing)
    diff_ocean_cm2_s: ti.f32        # K_0 [cm^2/sec] (thermal ocean mixing)


global belts, belts_n, display, avg_grad_t, avg_t, avg_albedo

def load_tables():
    def f(v: str):
        try:
            return float(v)
        except ValueError:
            return 0.0
    
    with TABLE_1_PATH.open("r") as handle:
        reader = csv.reader(handle)
        for i, row in enumerate(reader):
            if i == 0: continue
            _, b, z, _, _, del_p, del_z = row
            belt = belts[i - 1]

            
            belt.albedo_coeff_b = f(b)
            belt.elevation_meters = f(z) 
            
            belt.pressure_depth_dynes = f(del_p) * DYNES_PER_MB 
            
            # 1 km = 100,000 cm
            belt.ocean_depth_cm = f(del_z) * 1.0e5
            belts[i - 1] = belt

    with TABLE_2_PATH.open("r") as handle:
        reader = csv.reader(handle)
        for i, row in enumerate(reader):
            if i == 0: continue
            _, Q_s, _, K_h, K_w, K_0, a = row
            belt = belts[i - 1]
            belt.solar_flux_ly_sec = f(Q_s) * LY_PER_KLY / SECONDS_PER_YEAR
            
            belt.diff_air_cm2_s = f(K_h) * 1.0e10   # Header says 10^10
            belt.diff_vapor_cm2_s = f(K_w) * 1.0e9  # Header says 10^9
            belt.diff_ocean_cm2_s = f(K_0) * 1.0e6  # Header says 10^6
            
            belt.exchange_coeff_a = f(a)
            belts[i - 1] = belt
    
@ti.kernel
def setup():
    for i in range(N_BELTS):
        belt = belts[i]
        
        belt.lat_top_deg = 90.0 - (i * (180.0 / N_BELTS))
        belt.lat_bottom_deg = 90.0 - ((i + 1) * (180.0 / N_BELTS))

        lat_rb = tm.radians(belt.lat_bottom_deg)
        lat_rt = tm.radians(belt.lat_top_deg)

        belt.length_bottom_cm = 2 * tm.pi * EARTH_RADIUS_CM * tm.cos(lat_rb)
        belt.length_top_cm = 2 * tm.pi * EARTH_RADIUS_CM * tm.cos(lat_rt)
        
        belt.area_cm2 = 2 * tm.pi * (EARTH_RADIUS_CM ** 2) * \
            ((1 - tm.sin(lat_rb)) - (1 - tm.sin(lat_rt)))
        
        belt.temp_kelvin = belt.temp_kelvin = 260.0 + 40.0 * tm.cos((lat_rb + lat_rt) / 2)
        belt.ocean_fraction = 0.5
        
        belts[i] = belt


@ti.kernel
def compute_length_weighted_averages(active_belts: ti.template()):
    grad_sum_weighted: ti.f32 = 0
    temp_sum_weighted: ti.f32 = 0
    albedo_sum_weighted: ti.f32 = 0

    sum_length: ti.f32 = 0

    for i in range(N_BELTS - 1):
        belt_0 = active_belts[i]
        belt_1 = active_belts[i + 1]

        t0 = belt_0.temp_kelvin
        t1 = belt_1.temp_kelvin
        abs_gradient = abs(t0 - t1)

        grad_sum_weighted += abs_gradient * belt_0.length_top_cm
        temp_sum_weighted += t0 * belt_0.length_top_cm
        albedo_sum_weighted += belt_0.albedo * belt_0.length_top_cm

        sum_length += belt_0.length_top_cm
    
    avg_grad_t[None] = grad_sum_weighted / sum_length
    avg_t[None] = temp_sum_weighted / sum_length
    avg_albedo[None] = albedo_sum_weighted / sum_length


@ti.kernel
def update_temps(b_in: ti.template(), b_out: ti.template()):
    for belt_i in range(N_BELTS):
        belt = b_in[belt_i]

        belt_north = belt
        belt_south = belt

        if belt_i + 1 < N_BELTS:
            belt_north = b_in[belt_i + 1]
        if belt_i > 0:
            belt_south = b_in[belt_i - 1] 
        
        val: ti.f32 = 1e10
        temp = belt.temp_kelvin
        tries = 0

        while abs(val) > TOLERANCE and tries < MAX_TRIES:
            tries += 1
            # we look northwards for gradient
            val = solver_target(
                temp, belt, belt_i, belt_north, belt_south, avg_grad_t[None]
            )

            val_step = solver_target(
                temp + D_TEMP, belt, belt_i, belt_north, belt_south, avg_grad_t[None]
            )

            derivative = (val_step - val) / D_TEMP
            change = val / derivative
            change = tm.clamp(change, -10.0, 10.0)
            temp -= change

        # solver_target(
        #     temp + D_TEMP, belt, belt_i, belt_north, belt_south, avg_grad_t[None], debug_print=True
        # )
        old_temp = b_in[belt_i].temp_kelvin
        new_temp = old_temp * (1.0 - LEARNING_RATE) + temp * LEARNING_RATE
        

        b_out[belt_i] = b_in[belt_i]
        b_out[belt_i].temp_kelvin = new_temp
        b_out[belt_i].albedo = calculate_albedo(b_out[belt_i], new_temp)
        

@ti.func
def calculate_transport(
    belt: LatBelt,
    T_boundary: ti.f32, 
    del_T: ti.f32,
    avg_grad_global: ti.f32
) -> ti.f32:
    
    K_w = belt.diff_vapor_cm2_s
    K_h = belt.diff_air_cm2_s
    K_0 = belt.diff_ocean_cm2_s
    a_coeff = belt.exchange_coeff_a

    p_depth = belt.pressure_depth_dynes
    o_depth = belt.ocean_depth_cm
    o_frac = belt.ocean_fraction

    
    v: ti.f32 = 0.0
    if belt.lat_bottom_deg >= 5.0: # northern hemisphere logic
        v = a_coeff * (del_T + avg_grad_global)
    else:
        v = a_coeff * (del_T - avg_grad_global)

    # using the Clausius-Clapeyron approximation for T_boundary
    e_sat = 6.11 * tm.exp(5300.0 * (1.0/273.0 - 1.0/T_boundary)) * DYNES_PER_MB
    
    q = MOL_WEIGHT_RATIO * e_sat / REF_PRESSURE_DYNES
    
    numerator = MOL_WEIGHT_RATIO * LATENT_HEAT_WATER * e_sat * del_T
    denominator = REF_PRESSURE_DYNES * GAS_CONSTANT_AIR * T_boundary * T_boundary
    del_q = numerator / denominator

    
    c_term = (v * q - K_w * del_q / BELT_WIDTH_CM) * (p_depth / GRAVITY_CM_S2)
    flux_Latent = LATENT_HEAT_WATER * c_term
    flux_SensibleAir = (v * T_boundary - K_h * del_T / BELT_WIDTH_CM) * \
                       (SPECIFIC_HEAT_AIR * p_depth / GRAVITY_CM_S2)

    flux_SensibleOcean = -K_0 * o_depth * o_frac * (del_T / BELT_WIDTH_CM)
    return flux_Latent + flux_SensibleAir + flux_SensibleOcean


@ti.func
def calculate_albedo(belt: LatBelt, temp_kelvin: ti.f32) -> float:
    T_g = temp_kelvin - 0.0065 * belt.elevation_meters
    a_s: ti.f32 = 0.2
    if T_g <=  283.16:
        a_s = belt.albedo_coeff_b - 0.009 * T_g
    else:
        a_s = belt.albedo_coeff_b - 2.584

    a_s = max(0.0, min(1.0, a_s))
    return a_s

@ti.func
def calculate_radiation_balance(belt: LatBelt, temp_kelvin: ti.f32) -> ti.f32:
    Q_s = belt.solar_flux_ly_sec
    a_s: ti.f32 = calculate_albedo(belt, temp_kelvin)
    attenuation_term = 1.9e-16 * tm.pow(temp_kelvin, 6)
    I_s = STEFAN_BOLTZMANN * temp_kelvin ** 4 * (
        1 - ATM_ATTENUATION * tm.tanh(attenuation_term)
    )
    R_s = Q_s * (1 - a_s) - I_s

    return R_s

@ti.func
def solver_target(
    T_0: ti.f32, # only varied parameter
    belt: LatBelt,
    belt_i: ti.f32,
    belt_north: LatBelt,
    belt_south: LatBelt,
    avg_grad_T: ti.f32,
    debug_print=False
) -> ti.f32:
    
    
    P_in: ti.f32 = 0.0
    P_out: ti.f32 = 0.0
    del_T: ti.f32 = 0.0

    if belt_i + 1 < N_BELTS:
        del_T = belt_north.temp_kelvin - T_0
        t_boundary = (T_0 + belt_north.temp_kelvin) * 0.5
        P_in = calculate_transport(belt, t_boundary, del_T, avg_grad_T)
    if belt_i > 0:
        del_T = T_0 - belt_south.temp_kelvin
        t_boundary = (T_0 + belt_south.temp_kelvin) * 0.5
        P_out = calculate_transport(belt, t_boundary, del_T, avg_grad_T)

    R_s = calculate_radiation_balance(belt, T_0)

    l_north = belt.length_top_cm
    l_south = belt.length_bottom_cm
    area = belt.area_cm2

    if debug_print:
        print(
            "Estimate Zeros -- Lat: ", belt.lat_bottom_deg,
            " T_0: ", T_0,
            " R_s: ", R_s,
            " P_in: ", P_in,
            " P_out: ", P_out,
            " Result: ", (R_s * area) - (P_out * l_north) + (P_in * l_south)
        )
    return (R_s * area) - (P_out * l_north) + (P_in * l_south)


@ti.kernel
def render():
    for x, y in display:
        prog: ti.f32 = y / bounds[1] # never hits one

        index: ti.u32 = ti.u32(prog * N_BELTS)
        temperature = belts[index].temp_kelvin
        albedo = belts[index].albedo
        a = 0.9

        display[x, y] = (
            (temperature - 240) / 100 + albedo * a, 
            albedo * a, 
            albedo * a
        )

def update() -> None:
    global belts, belts_n

    compute_length_weighted_averages(belts)
    update_temps(belts, belts_n)
    belts, belts_n = belts_n, belts
    render()

def main() -> None:
    global belts, belts_n, display, avg_grad_t, avg_t, avg_albedo
    ti.init(arch=ti.gpu)

    belts = LatBelt.field(shape=(N_BELTS,))
    belts_n = LatBelt.field(shape=(N_BELTS,))
    display = ti.Vector.field(n=3, dtype=ti.f32, shape=bounds)
    avg_grad_t = ti.field(dtype=ti.f32, shape=())
    avg_t = ti.field(dtype=ti.f32, shape=())
    avg_albedo = ti.field(dtype=ti.f32, shape=())

    load_tables()
    gui = ti.GUI(name="Hello World", res=bounds)
    setup()

    space_pressed = False
    paused = False

    update()
    while gui.running:
        gui.get_event()  # must be called before is_pressed
        if not gui.is_pressed(ti.GUI.SHIFT):
            time.sleep(0.05)

        if gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False
        
        if gui.is_pressed(ti.GUI.BACKSPACE):
            setup()
            update()

        if gui.is_pressed(ti.GUI.SPACE) and not space_pressed:
            space_pressed = True
            paused = not paused

        if not gui.is_pressed(ti.GUI.SPACE) and space_pressed:
            space_pressed = False

        if not paused:
            update()

        gui.set_image(display)

        gui.text(content=f"Avg Temp:      {avg_t[None]:.2f}", pos=(0,1), font_size=20, color=0x0)
        gui.text(content=f"Avg Albedo:   {avg_albedo[None]:.4f}", pos=(0,0.97), font_size=20, color=0x0)
        if paused:
            gui.text(content="PAUSED", pos=(0.85,1), font_size=30, color=0xff0000)

        info_text = '''
            press BACKSPACE to reset
            press SPACE to pause/resume
            press ESCAPE to quit
            press SHIFT to speed up
        '''

        for i, line in enumerate(info_text.strip().split('\n')):
            gui.text(content=line.strip(), pos=(0,(i+1) * 0.03), font_size=20, color=0x0)
        
        gui.show()



if __name__ == '__main__':
    main()
