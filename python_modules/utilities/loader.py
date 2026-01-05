import os
import importlib.util
import pandas as pd
from pathlib import Path

def load_strategies_from_folder(folder_path="python_modules/strategies"):
    """Utility to scan a folder and dynamically load Python modules."""
    strategies_info = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            file_path = os.path.join(folder_path, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'get_metadata'):
                strategies_info.append(module.get_metadata())
    return strategies_info

def extract_daily_prices_to_csv(file_path: str, config_path: str, output_dir: str = "data/prices"):
    """
    Reads raw_prices_daily.xlsx, maps Bloomberg Tickers (Sheet Names) to
    Instrument Names using contract_specs.csv, and saves cleaned CSVs.
    """
    xlsx_path = Path(file_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    # 1. Load mapping: Bloomberg Ticker -> Instrument Name
    specs_df = pd.read_csv(config_path)
    ticker_to_name = {
        str(k).lower().strip(): str(v).lower().strip()
        for k, v in zip(specs_df['bbg_ticker'], specs_df['instrument'])
    }

    # 2. Process Excel File
    print(f"Opening {xlsx_path}...")
    if not xlsx_path.exists():
        print(f"Error: File not found at {xlsx_path}")
        return

    xls = pd.ExcelFile(xlsx_path)
    for sheet_name in xls.sheet_names:
        if sheet_name.lower() in ['summary', 'metadata', 'config']:
            continue

        lookup_key = sheet_name.lower().strip()
        commodity_name = ticker_to_name.get(lookup_key)

        if not commodity_name:
            print(f"  Warning: No mapping found for tab '{sheet_name}', skipping.")
            continue

        print(f"  Processing {sheet_name} (Ticker) -> {commodity_name}_daily.csv (Instrument)...")
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="A:G")
        if df.empty: continue

        if 'date' not in df.columns:
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'close'])
        df = df.sort_values('date')

        # Save using the clean Instrument Name
        output_file = out_path / f"{commodity_name}_daily.csv"
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    raw_file = project_root / "data" / "raw" / "raw_prices_daily.xlsx"
    config_file = project_root / "data" / "config" / "contract_specs.csv"
    output_dir = project_root / "data" / "prices"
    extract_daily_prices_to_csv(str(raw_file), str(config_file), output_dir=str(output_dir))