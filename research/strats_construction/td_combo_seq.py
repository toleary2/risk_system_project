import numpy as np
import pandas as pd


class TDSequentialCombo:
    """
    TradingView-accurate TD Sequential + TD Combo implementation.
    Matches TradingView behaviour bar-by-bar:
    - Sequential countdown uses close vs close[i-2]
    - Combo countdown uses close vs high/low[i-2]
    - Combo starts at setup bar 1 (not bar 9)
    - Both countdowns evaluate setup bar 9 itself
    """

    def __init__(self, setup_lookback=4, cd_lookback=2):
        self.setup_lookback = setup_lookback
        self.cd_lookback = cd_lookback

    def calculate(self, df):
        df = df.copy()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        # OUTPUT ARRAYS
        buy_setup = np.zeros(n, dtype=int)
        sell_setup = np.zeros(n, dtype=int)

        buy_seq = np.zeros(n, dtype=int)
        sell_seq = np.zeros(n, dtype=int)

        buy_combo = np.zeros(n, dtype=int)
        sell_combo = np.zeros(n, dtype=int)

        # INTERNAL STATE
        b_setup_cnt = 0
        s_setup_cnt = 0

        b_seq_cnt = 0
        s_seq_cnt = 0
        b_seq_active = False
        s_seq_active = False

        b_combo_cnt = 0
        s_combo_cnt = 0
        b_combo_active = False
        s_combo_active = False

        buy_setup1_idx = None
        sell_setup1_idx = None

        for i in range(self.setup_lookback + 1, n):

            # -------------------------------------------------------------
            # 1) PRICE FLIP + SETUP COUNTING
            # -------------------------------------------------------------
            buy_flip = (
                close[i] < close[i - self.setup_lookback]
                and close[i - 1] >= close[i - 1 - self.setup_lookback]
            )
            sell_flip = (
                close[i] > close[i - self.setup_lookback]
                and close[i - 1] <= close[i - 1 - self.setup_lookback]
            )

            if close[i] < close[i - self.setup_lookback]:
                if b_setup_cnt > 0:
                    b_setup_cnt += 1
                elif buy_flip:
                    b_setup_cnt = 1
                else:
                    b_setup_cnt = 0
                s_setup_cnt = 0

            elif close[i] > close[i - self.setup_lookback]:
                if s_setup_cnt > 0:
                    s_setup_cnt += 1
                elif sell_flip:
                    s_setup_cnt = 1
                else:
                    s_setup_cnt = 0
                b_setup_cnt = 0

            else:
                b_setup_cnt = 0
                s_setup_cnt = 0

            if b_setup_cnt > 9:
                b_setup_cnt = 1
            if s_setup_cnt > 9:
                s_setup_cnt = 1

            buy_setup[i] = b_setup_cnt
            sell_setup[i] = s_setup_cnt

            # -------------------------------------------------------------
            # 2) COMBO COUNTDOWN STARTS AT SETUP BAR 1
            # -------------------------------------------------------------
            if b_setup_cnt == 1:
                b_combo_active = True
                b_combo_cnt = 0

            if s_setup_cnt == 1:
                s_combo_active = True
                s_combo_cnt = 0

            # -------------------------------------------------------------
            # 3) SEQUENTIAL COUNTDOWN STARTS AT SETUP BAR 9
            #    AND MUST EVALUATE BAR 9 ITSELF
            # -------------------------------------------------------------
            if b_setup_cnt == 9:
                b_seq_active = True
                # evaluate bar 9 immediately
                if close[i] <= close[i - self.cd_lookback]:
                    b_seq_cnt += 1
                    buy_seq[i] = b_seq_cnt

            if s_setup_cnt == 9:
                s_seq_active = True
                if close[i] >= close[i - self.cd_lookback]:
                    s_seq_cnt += 1
                    sell_seq[i] = s_seq_cnt

            # -------------------------------------------------------------
            # 4) SEQUENTIAL COUNTDOWN (close vs close)
            # -------------------------------------------------------------
            if b_seq_active and b_setup_cnt != 9 and i >= self.cd_lookback:
                if close[i] <= close[i - self.cd_lookback]:
                    b_seq_cnt += 1
                    buy_seq[i] = b_seq_cnt
                if b_seq_cnt >= 13:
                    b_seq_active = False

            if s_seq_active and s_setup_cnt != 9 and i >= self.cd_lookback:
                if close[i] >= close[i - self.cd_lookback]:
                    s_seq_cnt += 1
                    sell_seq[i] = s_seq_cnt
                if s_seq_cnt >= 13:
                    s_seq_active = False

            # -------------------------------------------------------------
            # 5) COMBO COUNTDOWN (close vs high/low)
            #    MUST ALSO EVALUATE BAR 9
            # -------------------------------------------------------------
            if b_combo_active and i >= self.cd_lookback:
                if close[i] <= low[i - self.cd_lookback]:
                    b_combo_cnt += 1
                    buy_combo[i] = b_combo_cnt
                if b_combo_cnt >= 13:
                    b_combo_active = False

            if s_combo_active and i >= self.cd_lookback:
                if close[i] >= high[i - self.cd_lookback]:
                    s_combo_cnt += 1
                    sell_combo[i] = s_combo_cnt
                if s_combo_cnt >= 13:
                    s_combo_active = False

        # -------------------------------------------------------------
        # ASSIGN OUTPUT
        # -------------------------------------------------------------
        df["td_setup_buy"] = buy_setup
        df["td_setup_sell"] = sell_setup

        df["td_seq_buy"] = buy_seq
        df["td_seq_sell"] = sell_seq

        df["td_combo_buy"] = buy_combo
        df["td_combo_sell"] = sell_combo

        return df