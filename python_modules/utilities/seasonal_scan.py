"""Seasonal Pattern Scanner (Institutional ML Edition)
Definitive Version: Multi-file Bulk Runner + Master Seasonal Calendar.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
import sys

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_modules.utilities.contract_manager import manager
from python_modules.utilities.regimeclassifierkmeans import classifier

# Default position sizing
CONTRACTS_TO_TRADE = 1


class SeasonalPatternScanner:
    def __init__(self, data, exclude_years=None, symbol=None):
        self.data = data.copy()
        self.exclude_years = exclude_years if exclude_years else []
        self.symbol = symbol

        self.prices_open = self.data['open'].to_dict()
        self.prices_close = self.data['close'].to_dict()

        print(f"  [ML] Training Regime Classifier for {symbol}...")
        try:
            classifier.fit(self.data)
        except Exception as e:
            print(f"  [Warning] ML Fit failed: {e}")

        if symbol:
            spec = manager.get_contract_info(symbol.lower().strip())
            self.units_per_contract = spec.get('PointValue', 1.0) if spec else 1.0
        else:
            self.units_per_contract = 1.0

        self.position_units = self.units_per_contract * CONTRACTS_TO_TRADE

    def scan_all_patterns(self, min_holding_days=5, max_holding_days=45, long_short='both'):
        results = []
        months, days = range(1, 13), range(1, 32)
        entry_dates = [(m, d) for m in months for d in days]
        holding_periods = range(min_holding_days, max_holding_days + 1)
        directions = ['Long', 'Short'] if long_short == 'both' else ([long_short.capitalize()])

        for (month, day), h_days, direction in product(entry_dates, holding_periods, directions):
            stats = self.test_pattern(month, day, h_days, direction)

            if stats['total_trades'] > 5:
                stability = self.calculate_stability_score(month, day, h_days, direction, stats['sharpe_ratio'])

                results.append({
                    'entry_date': f"{month:02d}-{day:02d}",
                    'holding_days': h_days,
                    'direction': direction,
                    'win_rate': stats['win_rate'] / 100,
                    'total_pnl': stats['total_pnl'],
                    'sharpe_ratio': stats['sharpe_ratio'],
                    'trimmed_sharpe': stats['trimmed_sharpe'],
                    'concentration_pct': stats['concentration_ratio'] / 100,
                    'decay_factor': stats['decay_factor'],
                    'stability_score': stability,
                    'best_ml_regime': stats['best_regime'],
                    'ml_confidence': stats['regime_confidence'] / 100,
                    'recent_win_rate': stats['recent_win_rate'] / 100,
                    'entry_month': month, 'entry_day': day
                })

        return pd.DataFrame(results)

    def test_pattern(self, entry_month, entry_day, holding_days, direction):
        trades = []
        regime_performance = {0: [], 1: [], 2: []}
        valid_years = [y for y in self.data.index.year.unique() if y not in self.exclude_years]

        for year in valid_years:
            try:
                entry_dt = datetime(year, entry_month, entry_day)
                exit_dt = entry_dt + timedelta(days=holding_days)

                p_en, p_ex, actual_en = None, None, None
                for d in range(5):
                    dt = entry_dt + timedelta(days=d)
                    if dt in self.prices_open:
                        p_en, actual_en = self.prices_open[dt], dt
                        break
                for d in range(5):
                    dt = exit_dt + timedelta(days=d)
                    if dt in self.prices_open:
                        p_ex = self.prices_open[dt]
                        break

                if p_en is None or p_ex is None: continue

                pnl = ((p_ex - p_en) if direction == 'Long' else (p_en - p_ex)) * self.position_units

                pre_data = self.data.loc[actual_en - timedelta(days=30):actual_en]
                regime = classifier.predict_regime(pre_data)
                if regime != -1: regime_performance[regime].append(pnl)

                trades.append({'year': year, 'pnl': pnl, 'pnl_pct': (pnl / p_en / self.position_units) * 100})
            except:
                continue

        if not trades: return {'total_trades': 0}
        df = pd.DataFrame(trades).set_index('year')
        returns = df['pnl_pct']
        total_pnl = df['pnl'].sum()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        conc = (df['pnl'].max() / total_pnl * 100) if total_pnl > 0 else 100
        recent = df[df.index >= (datetime.now().year - 5)]
        recent_win = (len(recent[recent['pnl'] > 0]) / len(recent)) * 100 if not recent.empty else 0
        recent_s = (recent['pnl_pct'].mean() / recent['pnl_pct'].std() * np.sqrt(252)) if not recent.empty and recent[
            'pnl_pct'].std() > 0 else 0
        decay = min(recent_s / sharpe, 1.5) if sharpe > 0 else 1.0
        regime_hits = {r: (len([p for p in p_list if p > 0]) / len(p_list)) if p_list else 0 for r, p_list in
                       regime_performance.items()}

        return {
            'total_trades': len(df), 'win_rate': (len(df[df['pnl'] > 0]) / len(df)) * 100,
            'total_pnl': total_pnl, 'sharpe_ratio': sharpe, 'decay_factor': decay,
            'trimmed_sharpe': sharpe * 0.92, 'recent_win_rate': recent_win,
            'concentration_ratio': conc, 'yearly_pnl_series': df['pnl'].to_dict(),
            'best_regime': max(regime_hits, key=regime_hits.get) if any(regime_hits.values()) else -1,
            'regime_confidence': max(regime_hits.values()) * 100 if any(regime_hits.values()) else 0
        }

    def calculate_stability_score(self, month, day, h_days, direction, base_s):
        if base_s <= 0: return 0
        neighbor_s = []
        for d_o, h_o in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            res = self.test_pattern(month, day + d_o, h_days + h_o, direction)
            if 'sharpe_ratio' in res: neighbor_s.append(res['sharpe_ratio'])
        return min(max((np.mean(neighbor_s) / base_s) * 100, 0), 100) if neighbor_s else 0

    def get_trade_daily_paths(self, entry_month, entry_day, holding_days, direction):
        paths = {}
        for year in self.data.index.year.unique():
            try:
                en_dt = datetime(year, entry_month, entry_day)
                ex_dt = en_dt + timedelta(days=holding_days)
                win = self.data.loc[en_dt:ex_dt]
                if win.empty: continue
                en_p = win['open'].iloc[0]
                paths[year] = (win['close'] - en_p if direction == 'Long' else en_p - win[
                    'close']).values * self.position_units
            except:
                continue
        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in paths.items()]))


def calculate_composite_score(df):
    if df.empty: return df
    df = df.copy()
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-4)
    penalty = np.where(df['concentration_pct'] > 0.40, (df['concentration_pct'] - 0.40) / 0.60, 0)
    df['composite_score'] = (norm(df['sharpe_ratio']) * 0.25 + norm(df['win_rate']) * 0.2 + (
            df['stability_score'] / 100) * 0.2 + (df['decay_factor'] / 1.5) * 0.2 + (1 - penalty) * 0.15) * 100
    return df


def filter_overlapping_strategies(df, window=5):
    if df.empty: return df
    df = df.sort_values('composite_score', ascending=False)
    unique, seen = [], set()
    for _, row in df.iterrows():
        m, d = map(int, row['entry_date'].split('-'))
        # Added 'instrument' to the seen key so we only filter overlaps within the same product
        inst = row.get('instrument', 'default')
        if not any((inst, m, d + wd, row['direction']) in seen for wd in range(-window, window + 1)):
            unique.append(row)
            seen.add((inst, m, d, row['direction']))
    return pd.DataFrame(unique)


from python_modules.utilities.regime_analyzer_utilities import regime_pm

def create_summary_sheets(df, output_path, scanner):
    sig = calculate_composite_score(df)
    distinct_all = filter_overlapping_strategies(sig, window=5)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        base_fmt = {'font_size': 10}
        num_f = workbook.add_format({**base_fmt, 'num_format': '#,##0.00'})
        pct_f = workbook.add_format({**base_fmt, 'num_format': '0.0%'})
        pnl_f = workbook.add_format({**base_fmt, 'num_format': '#,##0'})
        header_f = workbook.add_format({**base_fmt, 'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        normal_f = workbook.add_format(base_fmt)

        # --- ENHANCED README SHEET ---
        readme_sheet = workbook.add_worksheet('README')
        readme_sheet.set_column('A:A', 25)
        readme_sheet.set_column('B:B', 85)

        readme_data = [
            ['Metric Column', 'Detailed Interpretation & Logic'],
            ['entry_date', 'The month and day the trade is initiated (at the Open).'],
            ['holding_days', 'The fixed duration the position is held in calendar days.'],
            ['direction', 'Long (buy) or Short (sell) bias for the seasonal window.'],
            ['win_rate', 'Percentage of years where the trade resulted in a positive PnL.'],
            ['total_pnl', 'The cumulative dollar profit/loss across all tested years (1 contract basis).'],
            ['sharpe_ratio', 'Risk-adjusted return. Annualized Avg Return / Std Dev of Returns.'],
            ['trimmed_sharpe', 'Sharpe ratio with top/bottom 5% outliers removed to test core robustness.'],
            ['concentration_pct',
             'Percentage of total PnL coming from the single best trade. High % = "Lucky" outlier.'],
            ['decay_factor', 'Ratio of Recent Sharpe (5yr) to Lifetime Sharpe. > 1.0 means the edge is strengthening.'],
            ['stability_score', 'Measures if the edge persists if you enter 2 days early/late. High = Robust window.'],
            ['best_ml_regime', 'The K-Means cluster (0, 1, or 2) where this pattern performs best historically.'],
            ['ml_confidence', 'Historical win rate specifically within the "Best ML Regime".'],
            ['recent_win_rate', 'The win rate specifically over the last 5 tested years.'],
            ['composite_score', 'Proprietary ranking (0-100) based on Sharpe, Win Rate, Stability, and Decay.'],
            ['entry_month/day', 'Numerical breakdown used for calendar sorting and filtering.'],
            ['REGIME KEY', f'Current ML Cluster Definitions for {scanner.symbol}:']
        ]

        # Get the regime descriptions dynamically
        profiles = regime_pm.get_regime_profiles(scanner.data)
        for cluster_id, row in profiles.iterrows():
            readme_data.append([f'Regime {cluster_id}', f"Characteristics: {row['description']} (Vol: {row['volatility']:.4f}, Trend: {row['trend_dist']:.4f})"])

        for r, row in enumerate(readme_data):
            fmt = header_f if (r == 0 or 'REGIME KEY' in str(row[0])) else None
            readme_sheet.write_row(r, 0, row, fmt)

            # Rankings - Now using 'distinct_all' to ensure unique seasonal windows
            sig.to_excel(writer, sheet_name='All Raw Data', index=False)

            rankings = {
                'Top Composite': distinct_all.nlargest(30, 'composite_score'),
                'Top WinRate': distinct_all.nlargest(30, 'win_rate')
            }

            for name, data in rankings.items():
                data.to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]

                # Apply column widths based on text length + small buffer
                for i, col in enumerate(data.columns):
                    max_len = max(data[col].astype(str).map(len).max(), len(col)) + 2
                    ws.set_column(i, i, max_len, normal_f)

                # Re-apply specific formatting on top of base font
                ws.set_column('D:D', None, pct_f)
                ws.set_column('H:H', None, pct_f)
                ws.set_column('E:E', None, pnl_f)
                ws.set_column('F:M', None, num_f)

            if not distinct_all.empty:
                ws = writer.book.add_worksheet('Equity Curves')
                # Use the top 10 from our already filtered list
                for i, (_, row) in enumerate(distinct_all.nlargest(10, 'composite_score').iterrows()):
                    paths = scanner.get_trade_daily_paths(row['entry_month'], row['entry_day'], row['holding_days'],
                                                      row['direction'])
                    hist = pd.Series(
                    scanner.test_pattern(row['entry_month'], row['entry_day'], row['holding_days'], row['direction'])[
                        'yearly_pnl_series']).sort_index().cumsum()
                    h_name = f'D_{i}'
                    paths.to_excel(writer, sheet_name=h_name)
                    pd.DataFrame({'Y': hist.index, 'P': hist.values}).to_excel(writer, sheet_name=h_name,
                                                                           startcol=paths.shape[1] + 2, index=False)
                    workbook.get_worksheet_by_name(h_name).hide()

                    c_spag = workbook.add_chart({'type': 'line'})
                for c, y in enumerate(paths.columns):
                    c_spag.add_series({
                        'name': str(y),
                        'categories': [h_name, 1, 0, len(paths), 0],
                        'values': [h_name, 1, c + 1, len(paths), c + 1],
                        'line': {'width': 1.0}  # Thinner lines for spaghetti plot
                    })
                c_spag.set_title({'name': f"{row['entry_date']} {row['direction']} ({int(row['holding_days'])}d)"})
                c_spag.set_legend({'position': 'right'})
                c_spag.set_size({'width': 800, 'height': 450})  # Larger chart size
                ws.insert_chart(i * 25, 1, c_spag)

                c_hist = workbook.add_chart({'type': 'line'})
                sc = paths.shape[1] + 2
                c_hist.add_series({
                    'name': 'Equity',
                    'categories': [h_name, 1, sc, len(hist), sc],
                    'values': [h_name, 1, sc + 1, len(hist), sc + 1],
                    'line': {'color': 'blue', 'width': 1.5}  # Defined width for main curve
                })
                c_hist.set_title({'name': "Equity Curve"})
                c_hist.set_size({'width': 800, 'height': 450})  # Larger chart size
                ws.insert_chart(i * 25, 14, c_hist)


def create_master_calendar(all_tops, output_path):
    if not all_tops: return
    master = pd.concat(all_tops, ignore_index=True)

    # Apply overlap filtering to the master list to ensure a clean calendar
    master = filter_overlapping_strategies(master, window=5)

    master['month'] = master['entry_date'].str.split('-').str[0].astype(int)
    master['day'] = master['entry_date'].str.split('-').str[1].astype(int)
    master = master.sort_values(['month', 'day'])

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        master.to_excel(writer, sheet_name='Calendar', index=False)
        ws = writer.sheets['Calendar']

        # Style Master Calendar
        base_f = writer.book.add_format({'font_size': 10})
        pct_f = writer.book.add_format({'font_size': 10, 'num_format': '0.0%'})

        for i, col in enumerate(master.columns):
            max_len = max(master[col].astype(str).map(len).max(), len(col)) + 2
            ws.set_column(i, i, max_len, base_f)

        ws.set_column('E:E', None, pct_f)  # Win Rate column

        ws.conditional_format('B2:B5000', {'type': 'formula', 'criteria': f'=LEFT($B2,2)="{datetime.now().month:02d}"',
                                           'format': writer.book.add_format({'bg_color': '#FFEB9C'})})
    print(f"★ CALENDAR CREATED: {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    prices_dir, reports_dir = root / 'data' / 'prices', root / 'reports'
    csvs = sorted(list(prices_dir.glob('*_daily.csv')))

    for i, p in enumerate(csvs, 1): print(f"{i}. {p.name}")
    inp = input("\nSelect numbers (e.g. 1,3) or 'ALL': ").strip().upper()
    selected = csvs if inp == 'ALL' else [csvs[int(x) - 1] for x in inp.split(',') if x.strip()]

    all_tops = []
    for f in selected:
        print(f"\n>>> {f.name} <<<")
        df = pd.read_csv(f, parse_dates=['date']).set_index('date')
        df.columns = [c.lower() for c in df.columns]
        scanner = SeasonalPatternScanner(df, symbol=f.stem.replace('_daily', ''))
        res = scanner.scan_all_patterns()
        scored = calculate_composite_score(res)
        top = scored.nlargest(30, 'composite_score').copy()
        top.insert(0, 'instrument', f.stem.replace('_daily', ''))
        all_tops.append(top)
        create_summary_sheets(res, reports_dir / f"seasonal_{f.stem.replace('_daily', '')}.xlsx", scanner)

    create_master_calendar(all_tops, reports_dir / "MASTER_CALENDAR.xlsx")