import pandas as pd
from pathlib import Path
import numpy as np


def reprocess_existing_report(file_name):
    # Get current directory and ensure we don't double-nest 'reports'
    current_dir = Path(__file__).resolve().parent

    # If the file is already in the current dir, use it;
    # otherwise look for a 'reports' subfolder
    report_path = current_dir / file_name
    if not report_path.exists() and (current_dir / 'reports').exists():
        report_path = current_dir / 'reports' / file_name

    output_path = report_path.parent / f"fixed_{file_name}"

    if not report_path.exists():
        print(f"Error: Could not find '{file_name}' at {report_path}")
        return

    print(f"Reading {file_name}...")
    df = pd.read_excel(report_path, sheet_name='All Patterns')

    # --- 1. RE-CALCULATE COMPOSITE SCORE ---
    s_min, s_max = df['sharpe_ratio'].min(), df['sharpe_ratio'].max()
    df['sharpe_norm'] = (df['sharpe_ratio'] - s_min) / (s_max - s_min + 1e-4)
    wr = df['win_rate'] / 100 if df['win_rate'].max() > 1 else df['win_rate']
    df['winrate_norm'] = (wr - 0.5) / 0.5
    df['composite_score'] = (df['sharpe_norm'] * 0.5 + df['winrate_norm'] * 0.5) * 100

    # --- 2. OVERLAP FILTER ---
    def filter_overlapping_strategies(df, window_days=5):
        df = df.sort_values('composite_score', ascending=False)
        unique_patterns = []
        seen_windows = set()
        for _, row in df.iterrows():
            month, day = map(int, row['entry_date'].split('-'))
            direction = row['direction']
            is_overlap = False
            for wd in range(-window_days, window_days + 1):
                if (month, day + wd, direction) in seen_windows:
                    is_overlap = True;
                    break
            if not is_overlap:
                unique_patterns.append(row)
                seen_windows.add((month, day, direction))
        return pd.DataFrame(unique_patterns)

    distinct_sig = filter_overlapping_strategies(df)

    # --- 3. FORMATTED EXCEL EXPORT ---
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        def format_sheet(writer, sheet_name, data):
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                column_letter = col[0].column_letter
                header_val = str(worksheet.cell(1, col[0].column).value).lower()
                for cell in col:
                    if isinstance(cell.value, (int, float)):
                        if "pnl" in header_val:
                            cell.number_format = '#,##0.00'
                        else:
                            cell.number_format = '0.00'
                worksheet.column_dimensions[column_letter].width = 20

        format_sheet(writer, 'All Patterns', df)
        for metric in ['sharpe_ratio', 'win_rate', 'total_pnl', 'composite_score']:
            top_data = distinct_sig.nlargest(30, metric)
            format_sheet(writer, f'Top {metric[:10]}', top_data)

    print(f"✓ Success! Fixed report saved as: {output_path.name}")


if __name__ == "__main__":
    reprocess_existing_report('../../../reports/seasonal_soybeans_daily.xlsx')
