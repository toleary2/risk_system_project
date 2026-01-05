"""Seasonal Pattern Scanner
Scans historical data to find optimal seasonal entry and exit patterns.
Integrates with ContractManager for accurate PnL calculations and generates
cumulative PnL (Equity Curve) charts in the final Excel report.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
import sys

# Ensure project root is in path for imports
# This file is in: project_root/python_modules/utilities/seasonal_scan.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_modules.utilities.contract_manager import manager

# Default position sizing (1 contract for standard benchmarking)
CONTRACTS_TO_TRADE = 1


class SeasonalPatternScanner:
    def __init__(self, data, exclude_years=None, pre_signal_days=5, symbol=None):
        self.data = data.copy()
        self.exclude_years = exclude_years if exclude_years else []
        self.pre_signal_days = pre_signal_days
        self.results = None
        self.symbol = symbol

        # 1. Lookup contract specs for the instrument (e.g., 'soybeans')
        if symbol:
            spec = manager.get_contract_info(symbol.lower().strip())
            if spec:
                self.units_per_contract = spec.get('PointValue', 1.0)
                print(f"  [Config] Loaded {symbol}: PointValue={self.units_per_contract}")
            else:
                print(f"  [Warning] No specs for {symbol}, using 1.0 multiplier.")
                self.units_per_contract = 1.0
        else:
            self.units_per_contract = 1.0

        self.position_units = self.units_per_contract * CONTRACTS_TO_TRADE

    def scan_all_patterns(self, min_holding_days=0, max_holding_days=60, long_short='both'):
        results = []
        months = range(1, 13)
        days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }

        entry_dates = [(m, d) for m in months for d in range(1, days_in_month[m] + 1)]
        holding_periods = range(min_holding_days, max_holding_days + 1)
        directions = []
        if long_short in ['long', 'both']: directions.append('Long')
        if long_short in ['short', 'both']: directions.append('Short')

        total_tests = len(entry_dates) * len(holding_periods) * len(directions)
        print(f"\nScanning {total_tests:,} patterns for {self.symbol}...")

        test_count = 0
        for (month, day), holding_days, direction in product(entry_dates, holding_periods, directions):
            test_count += 1
            # PROGRESS LOGGING: Print update every 1,000 patterns
            if test_count % 1000 == 0:
                print(f"  Progress: {test_count:,} / {total_tests:,} ({(test_count / total_tests) * 100:.1f}%)")

            stats = self.test_pattern(month, day, holding_days, direction)

            if stats['total_trades'] > 0:
                results.append({
                    'entry_month': month, 'entry_day': day, 'entry_date': f"{month:02d}-{day:02d}",
                    'holding_days': holding_days, 'direction': direction,
                    'total_trades': stats['total_trades'], 'win_rate': stats['win_rate'],
                    'total_pnl': stats['total_pnl'], 'avg_pnl': stats['avg_pnl'],
                    'avg_pnl_pct': stats['avg_pnl_pct'], 'profit_factor': stats['profit_factor'],
                    'avg_days_held': stats['avg_days_held'], 'worst_drawdown': stats['worst_drawdown'],
                    'sharpe_ratio': stats['sharpe_ratio'], 'sortino_ratio': stats['sortino_ratio'],
                    'pre_signal_confidence': stats['pre_signal_confidence']
                })

        self.results = pd.DataFrame(results)
        print(f"\n[OK] Scan complete! Processing reports...")
        return self.results

    def test_pattern(self, entry_month, entry_day, holding_days, direction):
        trades = []
        yearly_pnls = {}
        for year in self.data.index.year.unique():
            if year in self.exclude_years: continue
            try:
                entry_date = datetime(year, entry_month, entry_day)
                potential_entries = self.data.loc[self.data.index >= entry_date]
                if potential_entries.empty: continue

                actual_entry_date = potential_entries.index[0]
                entry_price = potential_entries.iloc[0]['close']

                exit_date = actual_entry_date + timedelta(days=holding_days)
                potential_exits = self.data.loc[self.data.index >= exit_date]
                if potential_exits.empty: continue
                actual_exit_date = potential_exits.index[0]
                exit_price = potential_exits.iloc[0]['close']

                pnl_per_unit = (exit_price - entry_price) if direction == 'Long' else (entry_price - exit_price)
                pnl = pnl_per_unit * self.position_units
                yearly_pnls[year] = pnl

                max_dd, max_dd_pct = self.calculate_trade_drawdown(self.data.loc[actual_entry_date:actual_exit_date],
                                                                   entry_price, direction)
                pre_signal_returns = self.get_pre_signal_pattern(actual_entry_date)

                trades.append({
                    'pnl': pnl, 'pnl_pct': (pnl_per_unit / entry_price) * 100,
                    'days_held': (actual_exit_date - actual_entry_date).days,
                    'max_drawdown_pct': max_dd_pct, 'pre_signal_returns': pre_signal_returns
                })
            except:
                continue

        if not trades: return {'total_trades': 0}
        df = pd.DataFrame(trades)
        winning = df[df['pnl'] > 0];
        losing = df[df['pnl'] <= 0]
        returns = df['pnl_pct']

        return {
            'total_trades': len(df), 'win_rate': (len(winning) / len(df)) * 100,
            'total_pnl': df['pnl'].sum(), 'avg_pnl': df['pnl'].mean(),
            'avg_pnl_pct': returns.mean(),
            'profit_factor': abs(winning['pnl'].sum() / losing['pnl'].sum()) if not losing.empty and losing[
                'pnl'].sum() != 0 else 10.0,
            'avg_days_held': df['days_held'].mean(), 'worst_drawdown': df['max_drawdown_pct'].min(),
            'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'sortino_ratio': (returns.mean() / returns[returns < 0].std() * np.sqrt(252)) if not returns[
                returns < 0].empty and returns[returns < 0].std() > 0 else 0,
            'pre_signal_confidence': self.calculate_pre_signal_confidence(trades),
            'yearly_pnl_series': yearly_pnls
        }

    def calculate_trade_drawdown(self, trade_data, entry_price, direction):
        if trade_data.empty: return 0, 0
        if direction == 'Long':
            low = trade_data['low'].min()
            return low - entry_price, (low - entry_price) / entry_price * 100
        else:
            high = trade_data['high'].max()
            return entry_price - high, (entry_price - high) / entry_price * 100

    def get_pre_signal_pattern(self, entry_date):
        pre_data = self.data[self.data.index < entry_date].tail(self.pre_signal_days)
        if len(pre_data) < self.pre_signal_days: return []
        close = pre_data['close'].values
        return ((close[1:] - close[:-1]) / close[:-1]).tolist()

    def calculate_pre_signal_confidence(self, trades):
        valid = [t['pre_signal_returns'] for t in trades if
                 t.get('pre_signal_returns') and len(t['pre_signal_returns']) == self.pre_signal_days - 1]

        if len(valid) < 2:
            return 0.0

        # Vectorized correlation matrix calculation
        arr = np.array(valid)
        corr_matrix = np.corrcoef(arr)

        # Get indices for the upper triangle (excluding diagonal) to avoid self-correlation and duplicates
        rows, cols = np.triu_indices(corr_matrix.shape[0], k=1)
        corrs = corr_matrix[rows, cols]

        # Remove NaNs if any (e.g., from flat price periods)
        corrs = corrs[~np.isnan(corrs)]

        return round((np.mean(corrs) + 1) * 50, 2) if corrs.size > 0 else 0.0


def load_data_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.columns = [col.lower() for col in data.columns]
    return data


def calculate_composite_score(df):
    if df.empty: return df
    df = df.copy()
    s_min, s_max = df['sharpe_ratio'].min(), df['sharpe_ratio'].max()
    df['sharpe_norm'] = (df['sharpe_ratio'] - s_min) / (s_max - s_min + 1e-4)
    wr = df['win_rate'] / 100 if df['win_rate'].max() > 1 else df['win_rate']
    df['winrate_norm'] = (wr - 0.5) / 0.5
    pf = df['profit_factor']
    df['pf_norm'] = (pf - 1.0) / (pf.max() - 1.0 + 1e-4)
    df['composite_score'] = (df['sharpe_norm'] * 0.4 + df['winrate_norm'] * 0.3 + df['pf_norm'] * 0.3) * 100
    return df


def filter_overlapping_strategies(df, window_days=5):
    if df.empty: return df
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


def create_summary_sheets(df, output_path, scanner_instance):
    sig = calculate_composite_score(df)
    distinct_sig = filter_overlapping_strategies(sig)

    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # 1. Main Sheet
            df.to_excel(writer, sheet_name='All Patterns', index=False)
            worksheet = writer.sheets['All Patterns']
            for col_num, value in enumerate(df.columns.values):
                worksheet.set_column(col_num, col_num, 15)

            # 2. Top 10 Equity Curves
            if not distinct_sig.empty:
                top_10 = distinct_sig.nlargest(10, 'composite_score')
                charts_sheet = workbook.add_worksheet('Equity Curves')

                for i, (idx, row) in enumerate(top_10.iterrows()):
                    # Re-calculate yearly series for graph
                    p_data = scanner_instance.test_pattern(row['entry_month'], row['entry_day'], row['holding_days'],
                                                           row['direction'])
                    series = pd.Series(p_data['yearly_pnl_series']).sort_index().cumsum()

                    # Write to helper sheet
                    helper_name = f'Data_{i}'
                    pd.DataFrame({'Year': series.index, 'PnL': series.values}).to_excel(writer, sheet_name=helper_name,
                                                                                        index=False)
                    workbook.get_worksheet_by_name(helper_name).hide()

                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name': f"{row['entry_date']} {row['direction']}",
                        'categories': [helper_name, 1, 0, len(series), 0],
                        'values': [helper_name, 1, 1, len(series), 1],
                    })
                    chart.set_title({'name': f"Top {i + 1}: {row['entry_date']} Score {row['composite_score']:.0f}"})
                    charts_sheet.insert_chart((i // 2) * 15, (i % 2) * 8, chart)

        print(f"✓ Formatted Report Created: {output_path}")
    except PermissionError:
        print(f"!! ERROR: Could not save {output_path.name}. Is it open in Excel?")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    prices_dir = root / 'data' / 'prices'
    reports_dir = root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    csv_candidates = sorted(list(prices_dir.glob('*_daily.csv')))
    if not csv_candidates:
        print("No CSVs found. Run loader.py first!")
        sys.exit()

    for i, p in enumerate(csv_candidates, 1): print(f"{i}. {p.name}")
    choice = int(input("\nSelect file number: ")) - 1
    file_path = csv_candidates[choice]
    instr_name = file_path.stem.lower().replace('_daily', '').strip()

    scanner = SeasonalPatternScanner(load_data_from_csv(file_path), symbol=instr_name)
    results = scanner.scan_all_patterns(min_holding_days=5, max_holding_days=45)

    output_xlsx = reports_dir / f"seasonal_{instr_name}.xlsx"
    create_summary_sheets(results, output_xlsx, scanner)