import pandas as pd
import numpy as np
import os
from pathlib import Path


def generate_large_stress_test(filename="soybeans_stress_large.csv", periods=1000):
    """
    Generates a large synthetic dataset that cycles through different
    market 'states' to trigger DeMark conditions repeatedly.
    """
    np.random.seed(42)
    data = []
    price = 1000.0

    # State definitions: (trend_bias, volatility, duration)
    states = [
        (0, 1, 50),  # Initial Calm
        (-4, 2, 100),  # Steady Bear (Setup & Countdown Buy)
        (6, 3, 80),  # Aggressive Bull (Setup & TDST Break)
        (0, 8, 60),  # High Vol Sideways (Setup Aborts)
        (-2, 1, 150),  # Slow Bear (Combo Buy focus)
        (5, 2, 120),  # Steady Bull (Setup & Countdown Sell)
        (-15, 10, 40),  # Market Crash (Extreme testing)
        (2, 2, 300)  # Recovery Phase
    ]

    for trend, vol, duration in states:
        for _ in range(duration):
            move = trend + np.random.normal(0, vol)
            price += move
            high = price + abs(np.random.normal(0, vol)) + 1
            low = price - abs(np.random.normal(0, vol)) - 1
            open_p = price - (move * 0.5)

            data.append({
                'open': open_p,
                'high': max(high, open_p, price),
                'low': min(low, open_p, price),
                'close': price
            })

    df = pd.DataFrame(data)
    df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='B')

    # --- FIXED PATH LOGIC ---
    # Find the project root (3 levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    target_dir = project_root / "data" / "testing_data"

    # Ensure the directory actually exists
    target_dir.mkdir(parents=True, exist_ok=True)

    final_path = target_dir / filename
    df.to_csv(final_path, index=False)

    print(f"Successfully generated {len(df)} bars.")
    print(f"File saved to: {final_path}")
    return df


if __name__ == "__main__":
    generate_large_stress_test()