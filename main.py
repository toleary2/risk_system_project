import pandas as pd
# Update our import to the new Utility location
from python_modules.utilities.loader import load_strategies_from_folder
from python_modules.sizing_engine import calculate_base_size


def main():
    print("--- Personal PM Toolkit: Strategy & Sizing ---")

    # 1. Use the Utility Loader to find our strategies
    active_strategies = load_strategies_from_folder()

    # 2. Load Sizing Rules
    try:
        rules_df = pd.read_csv("data/sizing_rules.csv")
    except FileNotFoundError:
        print("Error: data/sizing_rules.csv not found.")
        return

    results = []
    base_capital = 1_000_000

    # 3. Process the strategies found by the loader
    for strat in active_strategies:
        multiplier = calculate_base_size(strat['Conviction'], rules_df)

        # Simple sizing logic: 5% base allocation * multiplier
        recommended_pct = 0.05 * multiplier
        allocation_eur = base_capital * recommended_pct

        results.append({
            "Strategy": strat['StrategyName'],
            "Bucket": strat['Bucket'],
            "Conviction": strat['Conviction'],
            "Allocation %": recommended_pct,
            "Allocation EUR": allocation_eur
        })

    # 4. Display Results
    results_df = pd.DataFrame(results)
    print("\n--- Current Strategy Allocations ---")
    if not results_df.empty:
        print(results_df[['Strategy', 'Conviction', 'Allocation %', 'Allocation EUR']])
    else:
        print("No active strategies found in python_modules/strategies/")


if __name__ == "__main__":
    main()