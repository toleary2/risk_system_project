import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in path for imports
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from research.strats_construction.td_combo_seq import TDSequentialCombo

def run_demark_test():
    # 1. Setup paths
    data_path = root / 'data' / 'testing_data' / 'tradingview_soy_test_demark.csv'
    output_path = root / 'data' / 'testing_data' / 'demark_comparison_output.csv'
    
    if not data_path.exists():
        print(f"Error: Could not find input file at {data_path}")
        return

    # 2. Load Data
    print(f"Reading Soy data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure column names match expected internal format (lowercase)
    df.columns = [c.lower() for c in df.columns]
    
    # 3. Initialize and Calculate TD Indicators
    td = TDSequentialCombo()
    results = td.calculate(df)
    
    # 4. Filter for Comparison Columns
    # We combine buy/sell counts into single columns for easier reading
    comparison_df = pd.DataFrame()
    comparison_df['date'] = results['date']
    comparison_df['open'] = results['open']
    comparison_df['high'] = results['high']
    comparison_df['low'] = results['low']
    comparison_df['close'] = results['close']
    
    # Setups (1-9)
    comparison_df['setup_count'] = results['td_setup_buy'].where(results['td_setup_buy'] > 0, -results['td_setup_sell'])
    
    # Sequential Countdown (1-13)
    comparison_df['seq_countdown'] = results['td_seq_buy'].where(results['td_seq_buy'] > 0, -results['td_seq_sell'])
    
    # Combo Countdown (1-13)
    comparison_df['combo_countdown'] = results['td_combo_buy'].where(results['td_combo_buy'] > 0, -results['td_combo_sell'])
    
    # Risk Levels (for checking perfection and invalidation)
    comparison_df['risk_9'] = results['td_risk9_buy'].fillna(results['td_risk9_sell'])
    comparison_df['risk_13'] = results['td_risk13_buy'].fillna(results['td_risk13_sell'])

    # 5. Export
    comparison_df.to_csv(output_path, index=False)
    print(f"Test complete. Comparison file created at: {output_path}")

if __name__ == "__main__":
    run_demark_test()
