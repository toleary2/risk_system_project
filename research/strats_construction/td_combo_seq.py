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

        # Expiry Trackers (Rules 44, 53)
        b_seq_bars_since_13, s_seq_bars_since_13 = 0, 0
        b_combo_bars_since_13, s_combo_bars_since_13 = 0, 0

        b_seq_active, s_seq_active = False, False
        b_combo_active, s_combo_active = False, False
        curr_res, curr_sup = np.nan, np.nan
        curr_b_risk, curr_s_risk = np.nan, np.nan

        # Tracking for 9-13-9
        b_13_found, s_13_found = False, False

        for i in range(self.lookback, n):
            # --- SETUP (Rules 1-8, 23-24) ---
            if close[i] < close[i - self.lookback]:
                b_cnt = 1 if b_cnt >= 9 else b_cnt + 1
                s_cnt = 0
            elif close[i] > close[i - self.lookback]:
                s_cnt = 1 if s_cnt >= 9 else s_cnt + 1
                b_cnt = 0
            else:
                b_cnt, s_cnt = 0, 0

            buy_setup[i], sell_setup[i] = b_cnt, s_cnt

            # --- PERFECTED & TDST & RISK (Rules 9-10, 30-37) ---
            if b_cnt == 9:
                buy_perf[i] = low[i] <= min(low[i - 2], low[i - 3]) or low[i - 1] <= min(low[i - 3], low[i - 4])
                curr_sup, curr_b_risk = np.min(low[i - 8:i + 1]), np.min(low[i - 1:i + 1])

                # Rule 11/25: Activate Buy Countdown if not already in progress
                if not b_seq_active or b_seq_cnt == 0:
                    b_seq_active, b_seq_cnt, b_seq_bars_since_13 = True, 0, 0

                if not b_combo_active or b_combo_cnt == 0:
                    b_combo_active, b_combo_cnt, b_combo_bars_since_13 = True, 0, 0

                # Rule 28: Opposite setup cancels countdowns
                s_seq_active, s_seq_cnt, s_seq_bars_since_13 = False, 0, 0
                s_combo_active, s_combo_cnt, s_combo_bars_since_13 = False, 0, 0
                if b_13_found: buy_9139[i] = True; b_13_found = False

            if s_cnt == 9:
                sell_perf[i] = high[i] >= max(high[i - 2], high[i - 3]) or high[i - 1] >= max(high[i - 3], high[i - 4])
                curr_res, curr_s_risk = np.max(high[i - 8:i + 1]), np.max(high[i - 1:i + 1])

                # Activate Sell Countdown if not already in progress
                if not s_seq_active or s_seq_cnt == 0:
                    s_seq_active, s_seq_cnt, s_seq_bars_since_13 = True, 0, 0

                if not s_combo_active or s_combo_cnt == 0:
                    s_combo_active, s_combo_cnt, s_combo_bars_since_13 = True, 0, 0

                # Rule 27: Opposite setup cancels countdowns
                b_seq_active, b_seq_cnt, b_seq_bars_since_13 = False, 0, 0
                b_combo_active, b_combo_cnt, b_combo_bars_since_13 = False, 0, 0
                if s_13_found: sell_9139[i] = True; s_13_found = False

            # Invalidation (Rules 32-33, 36-37)
            if close[i] > curr_res: curr_res = np.nan
            if close[i] < curr_sup: curr_sup = np.nan
            if close[i] > curr_s_risk: curr_s_risk = np.nan
            if close[i] < curr_b_risk: curr_b_risk = np.nan

            tdst_res[i], tdst_sup[i], buy_risk[i], sell_risk[i] = curr_res, curr_sup, curr_b_risk, curr_s_risk

            # --- SEQUENTIAL COUNTDOWNS (Rules 11-16, 44-45) ---
            b_seq_signal, s_seq_signal = 0, 0
            if b_seq_active:
                if b_seq_cnt < 13:
                    if close[i] <= low[i - self.cd_lookback]:
                        b_seq_cnt += 1
                        b_seq_signal = b_seq_cnt
                        if b_seq_cnt == 13: b_13_found, b_seq_bars_since_13 = True, 0
                else:
                    b_seq_bars_since_13 += 1
                    b_seq_signal = 13
                    if b_seq_bars_since_13 > 12:
                        b_seq_cnt, b_seq_active, b_seq_signal = 0, False, 0

            if s_seq_active:
                if s_seq_cnt < 13:
                    if close[i] >= high[i - self.cd_lookback]:
                        s_seq_cnt += 1
                        s_seq_signal = s_seq_cnt
                        if s_seq_cnt == 13: s_13_found, s_seq_bars_since_13 = True, 0
                else:
                    s_seq_bars_since_13 += 1
                    s_seq_signal = 13
                    if s_seq_bars_since_13 > 12:
                        s_seq_cnt, s_seq_active, s_seq_signal = 0, False, 0

            # --- COMBO COUNTDOWNS (Rules 17-22, 53-54) ---
            b_combo_signal, s_combo_signal = 0, 0

            # Reset Combo if setup is broken
            if b_cnt == 0:
                b_combo_active, b_combo_cnt = False, 0
            if s_cnt == 0:
                s_combo_active, s_combo_cnt = False, 0

            if b_combo_active:
                if b_combo_cnt < 13:
                    if close[i] <= low[i - self.cd_lookback]:
                        b_combo_cnt += 1
                        b_combo_signal = b_combo_cnt
                        if b_combo_cnt == 13: b_combo_bars_since_13 = 0
                else:
                    b_combo_bars_since_13 += 1
                    b_combo_signal = 13
                    if b_combo_bars_since_13 > 12:
                        b_combo_cnt, b_combo_active, b_combo_signal = 0, False, 0

            if s_combo_active:
                if s_combo_cnt < 13:
                    if close[i] >= high[i - self.cd_lookback]:
                        s_combo_cnt += 1
                        s_combo_signal = s_combo_cnt
                        if s_combo_cnt == 13: s_combo_bars_since_13 = 0
                else:
                    s_combo_bars_since_13 += 1
                    s_combo_signal = 13
                    if s_combo_bars_since_13 > 12:
                        s_combo_cnt, s_combo_active, s_combo_signal = 0, False, 0

            buy_seq[i], sell_seq[i] = b_seq_signal, s_seq_signal
            buy_combo[i], sell_combo[i] = b_combo_signal, s_combo_signal

        df['td_setup_buy'], df['td_setup_sell'] = buy_setup, sell_setup
        df['td_perfected_buy'], df['td_perfected_sell'] = buy_perf, sell_perf
        df['tdst_res'], df['tdst_sup'] = tdst_res, tdst_sup
        df['td_buy_risk'], df['td_sell_risk'] = buy_risk, sell_risk
        df['td_seq_buy'], df['td_seq_sell'] = buy_seq, sell_seq
        df['td_combo_buy'], df['td_combo_sell'] = buy_combo, sell_combo
        df['td_9139_buy'], df['td_9139_sell'] = buy_9139, sell_9139

        return df