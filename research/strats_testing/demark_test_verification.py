import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from research.strats_construction.td_combo_seq import TDComboSequential


def verify_calculations():
    # --- FIXED PATH LOGIC ---
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "testing_data" / "tradingview_soy_test_demark.csv"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    # 1. Load and Normalize the data
    df = pd.read_csv(data_path)

    # Standardize column names (lowercase, no spaces)
    df.columns = [c.lower().strip() for c in df.columns]

    # Verify required columns exist regardless of their order in the CSV
    required = ['date', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return

    # Ensure chronological order (Oldest -> Newest)
    # This is critical for DeMark calculations!
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Run calculations
    td = TDComboSequential()
    results = td.calculate(df)

    print(f"--- Dataset Statistics ---")
    print(f"Total Bars: {len(results)}")

    # 3. Check for Rule Coverage
    print("\n--- Coverage Check ---")
    metrics = {
        "Buy Setups (Max)": results['td_setup_buy'].max(),
        "Sell Setups (Max)": results['td_setup_sell'].max(),
        "Perfected Buys Found": results['td_perfected_buy'].sum(),
        "Max Sequential CD (Buy)": results['td_seq_buy'].max(),
        "Max Combo CD (Buy)": results['td_combo_buy'].max()
    }

    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 4. Spot Check Logic: TDST Support Persistence
    if results['tdst_sup'].notna().any():
        print("\n--- Logic Spot-Check: TDST Support ---")
        idx = results['tdst_sup'].first_valid_index()
        val = results.loc[idx, 'tdst_sup']

        # Checking persistence on the sorted data
        is_constant = (results['tdst_sup'].iloc[idx:idx + 5] == val).all()
        print(f"Support Level Established: {val:.2f}")
        print(f"Support Persistence (5 bars): {'PASS' if is_constant else 'FAIL'}")

    # 5. Logic Spot-Check: Combo vs Sequential Timing ---
    print("\n--- Logic Spot-Check: Combo vs Sequential Timing ---")
    early_combo = results[(results['td_setup_buy'] < 9) & (results['td_combo_buy'] > 0)]
    early_seq = results[(results['td_setup_buy'] < 9) & (results['td_seq_buy'] > 0)]

    print(f"Combo active during early setup: {'YES' if not early_combo.empty else 'NO'}")
    print(f"Sequential active during early setup: {'YES' if not early_seq.empty else 'NO'} (Expected: NO)")

    # Save results with a fallback for PermissionErrors
    output_path = project_root / "data" / "testing_data" / "tradingview_stress_results.csv"
    try:
        results.to_csv(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(f"tradingview_stress_results_{timestamp}.csv")
        results.to_csv(fallback_path, index=False)
        print(f"\nWarning: {output_path} is locked. Saved to: {fallback_path}")


if __name__ == "__main__":
    verify_calculations()