import pandas as pd
import numpy as np


class TDComboSequential:
    def __init__(self, setup_lookback=4, countdown_lookback=2):
        self.lookback = setup_lookback
        self.cd_lookback = countdown_lookback

    def calculate(self, df):
        df = df.copy()
        n = len(df)

        # Initialize signal arrays
        buy_setup, sell_setup = np.zeros(n), np.zeros(n)
        buy_perf, sell_perf = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        tdst_res, tdst_sup = np.full(n, np.nan), np.full(n, np.nan)
        buy_risk, sell_risk = np.full(n, np.nan), np.full(n, np.nan)
        buy_seq, sell_seq = np.zeros(n), np.zeros(n)
        buy_combo, sell_combo = np.zeros(n), np.zeros(n)

        # Pattern & Reaction (Rules 44-67)
        buy_9139, sell_9139 = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        close, high, low = df['close'].values, df['high'].values, df['low'].values

        b_cnt, s_cnt = 0, 0
        b_seq_cnt, s_seq_cnt = 0, 0
        b_combo_cnt, s_combo_cnt = 0, 0

        # State flags
        b_seq_active, s_seq_active = False, False
        b_combo_active, s_combo_active = False, False

        curr_res, curr_sup = np.nan, np.nan
        curr_b_risk, curr_s_risk = np.nan, np.nan

        for i in range(self.lookback + 1, n):
            # --- 1. SETUP LOGIC (Price Flip) ---
            if close[i] < close[i - self.lookback]:
                if b_cnt > 0:
                    b_cnt += 1
                elif close[i - 1] >= close[i - self.lookback - 1]:
                    b_cnt = 1
                else:
                    b_cnt = 0
                s_cnt = 0
            elif close[i] > close[i - self.lookback]:
                if s_cnt > 0:
                    s_cnt += 1
                elif close[i - 1] <= close[i - self.lookback - 1]:
                    s_cnt = 1
                else:
                    s_cnt = 0
                b_cnt = 0
            else:
                b_cnt, s_cnt = 0, 0

            # Cap setup at 9
            if b_cnt > 9: b_cnt = 0
            if s_cnt > 9: s_cnt = 0
            buy_setup[i], sell_setup[i] = b_cnt, s_cnt

            # --- 2. COMBO COUNTDOWN (Starts at Bar 1) ---
            # Rule 17/18: Combo starts at Bar 1 of Setup
            if b_cnt == 1:
                b_combo_active = True
                b_combo_cnt = 0
            if s_cnt == 1:
                s_combo_active = True
                s_combo_cnt = 0

            if b_combo_active:
                if close[i] <= low[i - self.cd_lookback]:
                    b_combo_cnt += 1
                if b_combo_cnt >= 13:
                    buy_combo[i] = 13
                    b_combo_active = False  # Reset on completion
                else:
                    buy_combo[i] = b_combo_cnt

            if s_combo_active:
                if close[i] >= high[i - self.cd_lookback]:
                    s_combo_cnt += 1
                if s_combo_cnt >= 13:
                    sell_combo[i] = 13
                    s_combo_active = False  # Reset on completion
                else:
                    sell_combo[i] = s_combo_cnt

            # --- 3. SEQUENTIAL COUNTDOWN (Starts after Bar 9) ---
            if b_cnt == 9:
                b_seq_active = True
                b_seq_cnt = 0
                # TDST & Risk Logic
                buy_perf[i] = low[i] <= min(low[i - 2], low[i - 3]) or low[i - 1] <= min(low[i - 3], low[i - 4])
                curr_sup, curr_b_risk = np.min(low[i - 8:i + 1]), np.min(low[i - 1:i + 1])
                # Cancel opposite
                s_seq_active, s_combo_active = False, False

            if s_cnt == 9:
                s_seq_active = True
                s_seq_cnt = 0
                # TDST & Risk Logic
                sell_perf[i] = high[i] >= max(high[i - 2], high[i - 3]) or high[i - 1] >= max(high[i - 3], high[i - 4])
                curr_res, curr_s_risk = np.max(high[i - 8:i + 1]), np.max(high[i - 1:i + 1])
                # Cancel opposite
                b_seq_active, b_combo_active = False, False

            if b_seq_active:
                if close[i] <= low[i - self.cd_lookback]:
                    b_seq_cnt += 1
                if b_seq_cnt >= 13:
                    buy_seq[i] = 13
                    b_seq_active = False
                else:
                    buy_seq[i] = b_seq_cnt

            if s_seq_active:
                if close[i] >= high[i - self.cd_lookback]:
                    s_seq_cnt += 1
                if s_seq_cnt >= 13:
                    sell_seq[i] = 13
                    s_seq_active = False
                else:
                    sell_seq[i] = s_seq_cnt

            # Invalidation Logic
            if close[i] > curr_res: curr_res = np.nan
            if close[i] < curr_sup: curr_sup = np.nan
            tdst_res[i], tdst_sup[i], buy_risk[i], sell_risk[i] = curr_res, curr_sup, curr_b_risk, curr_s_risk

        # Final assignments
        df['td_setup_buy'], df['td_setup_sell'] = buy_setup, sell_setup
        df['td_perfected_buy'], df['td_perfected_sell'] = buy_perf, sell_perf
        df['tdst_res'], df['tdst_sup'] = tdst_res, tdst_sup
        df['td_buy_risk'], df['td_sell_risk'] = buy_risk, sell_risk
        df['td_seq_buy'], df['td_seq_sell'] = buy_seq, sell_seq
        df['td_combo_buy'], df['td_combo_sell'] = buy_combo, sell_combo

        return df