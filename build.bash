pyinstaller --name="ASTR65_Final" \
    --onefile \
    --windowed \
    --add-data "table1.csv:." \
    --add-data "table2.csv:." \
    --add-data "/Users/natemurphy/Desktop/_projects/ASTR65_Final/.venv/lib/python3.13/site-packages/taichi:taichi" \
    --add-data "main.py:." \
    main.py
