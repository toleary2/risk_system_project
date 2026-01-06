import numpy as np
import pandas as pd


class TDSequentialCombo:
    """
    TD Sequential + TD Combo + TDST + Risk Levels + 9-13-9 pattern.
    Input: DataFrame with columns: 'open', 'high', 'low', 'close'
    Output: DataFrame with additional TD columns.
    """

    def __init__(self, setup_lookback: int = 4, cd_lookback: int = 2):
        self.setup_lookback = setup_lookback  # usually 4
        self.cd_lookback = cd_lookback        # usually 2

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        # --- OUTPUT ARRAYS ---
        buy_setup = np.zeros(n, dtype=int)
        sell_setup = np.zeros(n, dtype=int)
        buy_perfected = np.zeros(n, dtype=bool)
        sell_perfected = np.zeros(n, dtype=bool)

        tdst_res = np.full(n, np.nan)
        tdst_sup = np.full(n, np.nan)

        buy_seq = np.zeros(n, dtype=int)
        sell_seq = np.zeros(n, dtype=int)

        buy_combo = np.zeros(n, dtype=int)
        sell_combo = np.zeros(n, dtype=int)

        buy_risk_9 = np.full(n, np.nan)
        sell_risk_9 = np.full(n, np.nan)

        buy_risk_13 = np.full(n, np.nan)
        sell_risk_13 = np.full(n, np.nan)

        nine_thirteen_nine_buy = np.zeros(n, dtype=bool)
        nine_thirteen_nine_sell = np.zeros(n, dtype=bool)

        # --- INTERNAL STATE ---
        b_setup_cnt = 0
        s_setup_cnt = 0

        b_seq_active = False
        s_seq_active = False
        b_seq_cnt = 0
        s_seq_cnt = 0
        b_seq_bars = []  # indices of countdown bars
        s_seq_bars = []

        b_combo_active = False
        s_combo_active = False
        b_combo_cnt = 0
        s_combo_cnt = 0
        b_combo_bars = []
        s_combo_bars = []

        # TDST levels (current)
        curr_tdst_res = np.nan
        curr_tdst_sup = np.nan

        # Risk levels (current)
        curr_buy_risk_9 = np.nan
        curr_sell_risk_9 = np.nan

        curr_buy_risk_13 = np.nan
        curr_sell_risk_13 = np.nan

        # For 9-13-9 pattern detection
        last_buy_setup9_idx = None
        last_sell_setup9_idx = None
        last_buy_seq13_idx = None
        last_sell_seq13_idx = None

        # perfected 9 timing windows
        buy_perf_window_end = None
        sell_perf_window_end = None

        # seq 13 reaction windows
        buy_13_window_end = None
        sell_13_window_end = None

        # ---------------------------------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------------------------------
        for i in range(self.setup_lookback + 1, n):

            # -------------------------------------------------------------
            # 1) PRICE FLIP & SETUP COUNTING
            # -------------------------------------------------------------
            # Price flip conditions (DeMark)
            buy_price_flip = (
                close[i] < close[i - self.setup_lookback]
                and close[i - 1] >= close[i - 1 - self.setup_lookback]
            )
            sell_price_flip = (
                close[i] > close[i - self.setup_lookback]
                and close[i - 1] <= close[i - 1 - self.setup_lookback]
            )

            # Setup counting logic
            if close[i] < close[i - self.setup_lookback]:
                # potential buy setup continuation or start
                if b_setup_cnt > 0:
                    b_setup_cnt += 1
                elif buy_price_flip:
                    b_setup_cnt = 1
                else:
                    b_setup_cnt = 0
                s_setup_cnt = 0
            elif close[i] > close[i - self.setup_lookback]:
                # potential sell setup continuation or start
                if s_setup_cnt > 0:
                    s_setup_cnt += 1
                elif sell_price_flip:
                    s_setup_cnt = 1
                else:
                    s_setup_cnt = 0
                b_setup_cnt = 0
            else:
                # equal close => reset both
                b_setup_cnt = 0
                s_setup_cnt = 0

            # cap (9 then recycle)
            if b_setup_cnt > 9:
                b_setup_cnt = 1
            if s_setup_cnt > 9:
                s_setup_cnt = 1

            buy_setup[i] = b_setup_cnt
            sell_setup[i] = s_setup_cnt

            # -------------------------------------------------------------
            # 2) TDST / PERFECTED 9 / RISK LEVELS (9)
            # -------------------------------------------------------------
            # When setup 1 occurs, store bar1 index for potential TDST
            if b_setup_cnt == 1:
                buy_setup1_idx = i
            if s_setup_cnt == 1:
                sell_setup1_idx = i

            # TDST levels are set at Setup bar 9, using bar1 and bar2
            if b_setup_cnt == 9:
                # Buy setup 9 completed
                # Perfected 9: low of bar8 or bar9 <= lows of bars 6 & 7
                # indices relative:
                # bar9 = i, bar8 = i-1, bar7 = i-2, bar6 = i-3
                bar9 = i
                bar8 = i - 1
                bar7 = i - 2
                bar6 = i - 3

                buy_perfected[bar9] = (
                    low[bar8] <= min(low[bar6], low[bar7])
                    or low[bar9] <= min(low[bar6], low[bar7])
                )

                # TDST Support = min(low of setup bar1, low of setup bar2)
                if buy_setup1_idx is not None:
                    bar1 = buy_setup1_idx
                    bar2 = bar1 + 1
                    if bar2 < n:
                        curr_tdst_sup = min(low[bar1], low[bar2])

                # Risk level 9 (buy) = min(low of bar8, low of bar9)
                curr_buy_risk_9 = min(low[bar8], low[bar9])
                buy_risk_9[bar9] = curr_buy_risk_9

                # 4-bar reaction window from bar9
                buy_perf_window_end = bar9 + 4

                # Start Sequential countdown AFTER setup 9 completes
                b_seq_active = True
                b_seq_cnt = 0
                b_seq_bars = []

                # Start Combo countdown at setup bar1 (already in progress conceptually)
                b_combo_active = True
                # but do not reset combo count here – it starts from bar1
                # If you want to strictly start counting at bar9, reset here.

                # Cancel opposite seq/combo
                s_seq_active = False
                s_combo_active = False

            if s_setup_cnt == 9:
                # Sell setup 9 completed
                bar9 = i
                bar8 = i - 1
                bar7 = i - 2
                bar6 = i - 3

                sell_perfected[bar9] = (
                    high[bar8] >= max(high[bar6], high[bar7])
                    or high[bar9] >= max(high[bar6], high[bar7])
                )

                # TDST Resistance = max(high of setup bar1, high of setup bar2)
                if sell_setup1_idx is not None:
                    bar1 = sell_setup1_idx
                    bar2 = bar1 + 1
                    if bar2 < n:
                        curr_tdst_res = max(high[bar1], high[bar2])

                # Risk level 9 (sell) = max(high of bar8, high of bar9)
                curr_sell_risk_9 = max(high[bar8], high[bar9])
                sell_risk_9[bar9] = curr_sell_risk_9

                sell_perf_window_end = bar9 + 4

                # Start Sequential countdown AFTER setup 9 completes
                s_seq_active = True
                s_seq_cnt = 0
                s_seq_bars = []

                # Start Combo countdown in sell direction
                s_combo_active = True

                # Cancel opposite seq/combo
                b_seq_active = False
                b_combo_active = False

            # TDST invalidation by close
            if not np.isnan(curr_tdst_res) and close[i] > curr_tdst_res:
                curr_tdst_res = np.nan
            if not np.isnan(curr_tdst_sup) and close[i] < curr_tdst_sup:
                curr_tdst_sup = np.nan

            tdst_res[i] = curr_tdst_res
            tdst_sup[i] = curr_tdst_sup

            # -------------------------------------------------------------
            # 3) SEQUENTIAL COUNTDOWN (13)
            # -------------------------------------------------------------
            # Buy Sequential: close[i] <= low[i-2], non-consecutive, only after buy setup 9
            if b_seq_active and i >= self.cd_lookback:
                if close[i] <= low[i - self.cd_lookback]:
                    b_seq_cnt += 1
                    buy_seq[i] = b_seq_cnt
                    b_seq_bars.append(i)
                    if b_seq_cnt == 13:
                        b_seq_active = False
                        last_buy_seq13_idx = i
                        # risk level 13 = lowest low of countdown bars
                        lows = low[np.array(b_seq_bars, dtype=int)]
                        curr_buy_risk_13 = np.min(lows)
                        buy_risk_13[i] = curr_buy_risk_13
                        buy_13_window_end = i + 12

            # Sell Sequential: close[i] >= high[i-2]
            if s_seq_active and i >= self.cd_lookback:
                if close[i] >= high[i - self.cd_lookback]:
                    s_seq_cnt += 1
                    sell_seq[i] = s_seq_cnt
                    s_seq_bars.append(i)
                    if s_seq_cnt == 13:
                        s_seq_active = False
                        last_sell_seq13_idx = i
                        highs = high[np.array(s_seq_bars, dtype=int)]
                        curr_sell_risk_13 = np.max(highs)
                        sell_risk_13[i] = curr_sell_risk_13
                        sell_13_window_end = i + 12

            # -------------------------------------------------------------
            # 4) COMBO COUNTDOWN (13)
            #    starts at setup bar1, same condition as sequential here
            # -------------------------------------------------------------
            if b_combo_active and i >= self.cd_lookback:
                if close[i] <= low[i - self.cd_lookback]:
                    b_combo_cnt += 1
                    buy_combo[i] = b_combo_cnt
                    b_combo_bars.append(i)
                    if b_combo_cnt >= 13:
                        b_combo_active = False

            if s_combo_active and i >= self.cd_lookback:
                if close[i] >= high[i - self.cd_lookback]:
                    s_combo_cnt += 1
                    sell_combo[i] = s_combo_cnt
                    s_combo_bars.append(i)
                    if s_combo_cnt >= 13:
                        s_combo_active = False

            # -------------------------------------------------------------
            # 5) RISK VIOLATION CHECKS (9 & 13)
            # -------------------------------------------------------------
            # 9 risk violation
            if not np.isnan(curr_buy_risk_9) and close[i] < curr_buy_risk_9:
                curr_buy_risk_9 = np.nan
            if not np.isnan(curr_sell_risk_9) and close[i] > curr_sell_risk_9:
                curr_sell_risk_9 = np.nan

            buy_risk_9[i] = curr_buy_risk_9
            sell_risk_9[i] = curr_sell_risk_9

            # 13 risk violation
            if not np.isnan(curr_buy_risk_13) and close[i] < curr_buy_risk_13:
                curr_buy_risk_13 = np.nan
            if not np.isnan(curr_sell_risk_13) and close[i] > curr_sell_risk_13:
                curr_sell_risk_13 = np.nan

            buy_risk_13[i] = curr_buy_risk_13
            sell_risk_13[i] = curr_sell_risk_13

            # -------------------------------------------------------------
            # 6) REACTION WINDOWS (9 & 13)
            #    Here we only mark when a reaction occurs.
            #    You can add explicit columns if you want booleans.
            # -------------------------------------------------------------
            # 9 reaction: opposite close within 4 bars
            if buy_perf_window_end is not None and i <= buy_perf_window_end:
                # For a buy setup, reaction = at least one higher close
                pass  # marker logic can be added
            if sell_perf_window_end is not None and i <= sell_perf_window_end:
                # For a sell setup, reaction = at least one lower close
                pass

            # 13 reaction: opposite close within 12 bars
            if buy_13_window_end is not None and i <= buy_13_window_end:
                # For a buy 13, reaction = at least one higher close
                pass
            if sell_13_window_end is not None and i <= sell_13_window_end:
                # For a sell 13, reaction = at least one lower close
                pass

            # -------------------------------------------------------------
            # 7) 9-13-9 PATTERN DETECTION
            # -------------------------------------------------------------
            # track most recent setup9s
            if b_setup_cnt == 9:
                last_buy_setup9_idx = i
            if s_setup_cnt == 9:
                last_sell_setup9_idx = i

            # once a 13 exists AND a later 9 in same direction appears:
            # (this is a simple structural 9-13-9 detector; you can add
            #  full validity checks with risk & reaction if you want.)
            if last_buy_seq13_idx is not None and last_buy_setup9_idx is not None:
                if last_buy_setup9_idx > last_buy_seq13_idx:
                    nine_thirteen_nine_buy[last_buy_setup9_idx] = True

            if last_sell_seq13_idx is not None and last_sell_setup9_idx is not None:
                if last_sell_setup9_idx > last_sell_seq13_idx:
                    nine_thirteen_nine_sell[last_sell_setup9_idx] = True

        # ---------------------------------------------------------------------
        # ASSIGN TO DATAFRAME
        # ---------------------------------------------------------------------
        df["td_setup_buy"] = buy_setup
        df["td_setup_sell"] = sell_setup
        df["td_perfected_buy"] = buy_perfected
        df["td_perfected_sell"] = sell_perfected

        df["tdst_res"] = tdst_res
        df["tdst_sup"] = tdst_sup

        df["td_seq_buy"] = buy_seq
        df["td_seq_sell"] = sell_seq

        df["td_combo_buy"] = buy_combo
        df["td_combo_sell"] = sell_combo

        df["td_risk9_buy"] = buy_risk_9
        df["td_risk9_sell"] = sell_risk_9

        df["td_risk13_buy"] = buy_risk_13
        df["td_risk13_sell"] = sell_risk_13

        df["td_9_13_9_buy"] = nine_thirteen_nine_buy
        df["td_9_13_9_sell"] = nine_thirteen_nine_sell

        return df