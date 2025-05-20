#!/usr/bin/env python
"""
HISTORY: This is the history mode of the code that was moved in at May 14 2025.

Stage 1 – Compute all factors and write to feature_store (partitioned by factor_date + permno).
Example:
    python build_factors.py --start 2018-01-01 --end 2025-04-30 \
                            --reb W-MON --lookback 1 --n_jobs 8
"""
from __future__ import annotations

import argparse
import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

import data_storage.io.data_cache as dc  # Reuse your data layer

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    reb_freq: str
    lookback_years: int
    n_jobs: int
    debug: bool
    feature_store: Path = Path('feature_store')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build factor store")
    p.add_argument("--start", type=str, required=True, help="开始日期 YYYY-MM-DD")
    p.add_argument("--end",   type=str, required=True, help="结束日期 YYYY-MM-DD")
    p.add_argument("--reb",   default="W-MON", help="重平衡频率 (pandas offset)")
    p.add_argument("--lookback", type=int, default=1, help="rolling lookback (yrs)")
    p.add_argument("--n_jobs",   type=int, default=os.cpu_count() or 4)
    p.add_argument("--debug", action="store_true", help="启用调试模式")
    return p.parse_args()

# ----------------------------------------------------------------------------
# --- 因子计算函数
# ----------------------------------------------------------------------------

FACTOR_COLS = [
    'price_deviation_20',
    'price_position_20',
    'RSI_low_dist_14',
    'bb_position',
    'macd_hist',
    'vol_20d',
    'momentum_3',
    'momentum_5',
    'momentum_10',
    'reversal_1',
    'reversal_3',
    'reversal_weighted',
    'reversal_rsi',
    'vol_ratio',
    'RSI_14',
    'RSI_7',
    'RSI_3',
    'BB_upper',
    'BB_lower',
    'OBV',
    'MFI',
    'VWAP_diff_10',
    'vol_std_5',
    'vol_std_10',
    'MACD_default_line',
    'MACD_default_signal',
    'MACD_default_hist',
    'MACD_default_cross',
    'MACD_short_line',
    'MACD_short_signal',
    'MACD_short_hist',
    'MACD_short_cross',
]

_FACTOR_SCHEMA = pa.schema(
    [
        ("permno",       pa.int64()),
        ("factor_date",  pa.date32()),
    ] +
    [(c, pa.float32()) for c in FACTOR_COLS] + 
    [("weekly_return", pa.float32()), 
     ("weekly_rank", pa.float32())]
)

# --- helper math (vectorised pandas, but could switch to numba later) ---------------

def sma(s: pd.Series, w: int):
    return s.rolling(w, min_periods=max(2, w//2)).mean()

def rsi(s: pd.Series, w: int = 14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = sma(gain, w)  
    avg_loss = sma(loss, w)
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - 100 / (1 + rs)

def bbands(s: pd.Series, w: int = 20, k: int = 2):
    ma = sma(s, w)
    std = s.rolling(w, min_periods=max(2, w//2)).std()
    return ma + k*std, ma, ma - k*std

def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    emaf = s.ewm(span=fast, adjust=False, min_periods=max(2, fast//2)).mean()
    emas = s.ewm(span=slow, adjust=False, min_periods=max(2, slow//2)).mean()
    macd_line = emaf - emas
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=max(2, signal//2)).mean()
    return macd_line - signal_line

def compute_extra_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional technical factors"""
    # Make sure the data is sorted by date
    df = df.sort_values('date').copy()
    
    # --- Momentum 因子：close 滞后 pct_change ---
    mom = pd.concat(
        {f'momentum_{p}': df['close'].pct_change(p, fill_method=None) for p in (3, 5, 10)},
        axis=1
    )
    
    # --- Reversal 因子 ---
    rev_short = pd.concat(
        {f'reversal_{p}': -df['close'].pct_change(p, fill_method=None) for p in (1, 3)},
        axis=1
    )
    # 带权重的 Reversal，权重 1,2,3,4,5
    ret_lags = pd.concat(
        {i: df['close'].pct_change(i, fill_method=None) for i in range(1, 6)},
        axis=1
    )
    weights = np.arange(1, 6)
    weighted = -(ret_lags.mul(weights, axis=1).sum(axis=1) / weights.sum())
    rev_rsi = 50 - rsi(df['close'], 14)
    
    # --- Volume Ratio（今日成交量 / 20 日平均成交量）---
    vol_ratio = df['volume'] / sma(df['volume'], 20)
    
    # --- 不同窗口的 RSI ---
    rsis = pd.concat(
        {f'RSI_{w}': rsi(df['close'], w) for w in (14, 7, 3)},
        axis=1
    )
    
    # --- Bollinger 上下轨 ---
    bb_up, bb_mid, bb_low = bbands(df['close'], 20, 2)
    
    # --- OBV ---
    obv = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
    
    # --- MFI (14) ---
    if 'high' in df.columns and 'low' in df.columns:
        tp = (df['high'] + df['low'] + df['close']) / 3
    else:
        tp = df['close']  # Fallback if high/low not available
        
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)
    mfr = sma(pos_mf, 14) / (sma(neg_mf, 14) + 1e-10)
    mfi = 100 - 100 / (1 + mfr)
    
    # --- VWAP diff_10 (滚动 10 日 VWAP 与收盘价的相对差) ---
    vwap10 = (df['close'] * df['volume']).rolling(10, min_periods=5).sum() \
             / df['volume'].rolling(10, min_periods=5).sum()
    vwap_diff_10 = (df['close'] - vwap10) / vwap10
    
    # --- 波动率标准差 ---
    daily_ret = df['close'].pct_change(fill_method=None).fillna(0)
    vol_std_5  = daily_ret.rolling(5,  min_periods=5).std()
    vol_std_10 = daily_ret.rolling(10, min_periods=5).std()
    
    # --- MACD 系列（默认 12,26,9）---
    ema_fast   = df['close'].ewm(span=12, adjust=False, min_periods=6).mean()
    ema_slow   = df['close'].ewm(span=26, adjust=False, min_periods=13).mean()
    macd_line  = ema_fast - ema_slow
    macd_signal= macd_line.ewm(span=9, adjust=False, min_periods=5).mean()
    macd_hist  = macd_line - macd_signal
    macd_cross = (macd_hist > 0).astype(int)
    
    # --- MACD short（6,13,5）---
    efa_s = df['close'].ewm(span=6,  adjust=False, min_periods=3).mean()
    esa_s = df['close'].ewm(span=13, adjust=False, min_periods=7).mean()
    macd_s_line   = efa_s - esa_s
    macd_s_signal = macd_s_line.ewm(span=5, adjust=False, min_periods=3).mean()
    macd_s_hist   = macd_s_line - macd_s_signal
    macd_s_cross  = (macd_s_hist > 0).astype(int)
    
    # 合并所有新因子
    extras = pd.concat([
        mom,
        rev_short,
        pd.Series(weighted,      name='reversal_weighted'),
        pd.Series(rev_rsi,       name='reversal_rsi'),
        pd.Series(vol_ratio,     name='vol_ratio'),
        rsis,
        pd.Series(bb_up,         name='BB_upper'),
        pd.Series(bb_low,        name='BB_lower'),
        pd.Series(obv,           name='OBV'),
        pd.Series(mfi,           name='MFI'),
        pd.Series(vwap_diff_10,  name='VWAP_diff_10'),
        pd.Series(vol_std_5,     name='vol_std_5'),
        pd.Series(vol_std_10,    name='vol_std_10'),
        pd.Series(macd_line,     name='MACD_default_line'),
        pd.Series(macd_signal,   name='MACD_default_signal'),
        pd.Series(macd_hist,     name='MACD_default_hist'),
        pd.Series(macd_cross,    name='MACD_default_cross'),
        pd.Series(macd_s_line,   name='MACD_short_line'),
        pd.Series(macd_s_signal, name='MACD_short_signal'),
        pd.Series(macd_s_hist,   name='MACD_short_hist'),
        pd.Series(macd_s_cross,  name='MACD_short_cross'),
    ], axis=1)
    
    return df.join(extras)

# ----------------------------------------------------------------------------
# 无前视偏差的因子计算函数
# ----------------------------------------------------------------------------
def compute_factors_pti(df: pd.DataFrame, week: pd.Timestamp, config: Config = None) -> pd.Series | None:
    """
    Compute all factors *as of* the rebalance date (inclusive) and
    attach the 5‑day **future** return as the training target.

    Parameters
    ----------
    df    : price dataframe with columns ['date','close','volume', ...]
    week  : rebalance date (will be aligned to the most recent trading day)
    config: optional Config object for debug settings

    Returns
    -------
    pd.Series with FACTOR_COLS + ['future_return_5d', 'weekly_return', 'weekly_rank']   or   None
    """
    try:
        if df.empty:
            return None  # missing data for this stock
            
        # --- 1. 准备 trading_days ---
        trading_days = pd.to_datetime(df['date'].unique())
        trading_days = np.sort(trading_days)  # 保证按时间升序

        # --- 2. 对齐到 ≤ week 的最近交易日 ---
        prior = trading_days[trading_days <= week]
        if len(prior) == 0:
            return None  # no data for this stock before rebalance date
        ref_date = prior[-1]  # 最近交易日，如 2023-10-02
        
        # --- 3. 计算周收益率 ---
        # 找到本周的最后一个交易日（周五）
        week_end = week + pd.Timedelta(days=4)  # 假设是周五
        future_dates = trading_days[(trading_days > ref_date) & (trading_days <= week_end)]
        if len(future_dates) == 0:
            return None  # 没有本周的交易数据
            
        last_trading_day = future_dates[-1]
        start_price = df.loc[df['date'] == ref_date, 'close'].iloc[0]
        end_price = df.loc[df['date'] == last_trading_day, 'close'].iloc[0]
        weekly_return = end_price / start_price - 1
        
        # --- 4. 前向 5 天返回率 ---
        future_dates = trading_days[trading_days > ref_date]
        if len(future_dates) < 5:  # need at least 5 future days
            return None
        future_date_5d = future_dates[min(4, len(future_dates)-1)]
        
        # get current & future prices
        i = np.where(trading_days == ref_date)[0][0]
        j = np.where(trading_days == future_date_5d)[0][0]
        
        if i >= len(df) or j >= len(df):
            return None  # should never happen
        
        curr_price = df.loc[df['date'] == ref_date, 'close'].iloc[0]
        future_price = df.loc[df['date'] == future_date_5d, 'close'].iloc[0]
        future_return = future_price / curr_price - 1
        
        # --- 5. 基础价格动能因子 ---
        # 只使用 ref_date 及之前的数据计算因子
        sub_df = df[df['date'] <= ref_date].copy()
        
        # 至少需要 20 天数据
        if len(sub_df) < 20:
            return None
            
        # 确保按日期排序
        sub_df = sub_df.sort_values('date')
        
        # 计算 SMA 和 Bollinger Bands
        close = sub_df['close']
        sma20 = sma(close, 20)
        upper, mid, lower = bbands(close, 20, 2)

        # 获取最新收盘价（即 ref_date 的收盘价）
        last_close = close.iloc[-1]
        
        # 计算离 SMA20 的偏差百分比
        price_dev_20 = (last_close / sma20.iloc[-1] - 1)
        
        # 计算价格在 BB 带中的位置 (0-100%)
        # 0% = 下轨，50% = 中轨，100% = 上轨
        width = upper.iloc[-1] - lower.iloc[-1]
        if width <= 0:  # 防止出现零宽度 BB 带
            bb_pos = 50  # 默认居中
        else:
            bb_pos = 100 * (last_close - lower.iloc[-1]) / width
        
        # 计算当前价格与最低 RSI 的位置
        rsi14 = rsi(close, 14)
        lowest_rsi = rsi14.rolling(14).min().iloc[-1]
        rsi_low_dist = rsi14.iloc[-1] - lowest_rsi
        
        # MACD 柱状图
        macd_h = macd(close)
        
        # 20 日波动率
        vol_20d = close.pct_change(fill_method=None).rolling(20).std().iloc[-1]
        
        # --- 6. 计算所有高级因子 ---
        # 通过另一个函数计算额外因子，更整洁
        full_df = compute_extra_factors(sub_df)
        
        # --- 7. 构建完整因子 Series ---
        # 获取最新行（ref_date的数据）用于所有因子
        last_row = full_df.iloc[-1]
        
        # 构建一个因子 Series
        factors = pd.Series({
            'permno': df['permno'].iloc[0] if 'permno' in df.columns else None,
            'factor_date': week,
            'future_return_5d': future_return,
            'weekly_return': weekly_return,
            'weekly_rank': np.nan,  # 这个值会在后续处理中更新
            'price_deviation_20': price_dev_20,
            'price_position_20': bb_pos,
            'RSI_low_dist_14': rsi_low_dist,
            'bb_position': bb_pos,
            'macd_hist': macd_h.iloc[-1],
            'vol_20d': vol_20d,
        })
        
        # 添加额外因子（从 last_row 获取）
        for col in FACTOR_COLS:
            if col in last_row and col not in factors:
                factors[col] = last_row[col]
                
        if config and config.debug:
            print(f"Rebalance: {week}, Ref date: {ref_date}, "
                  f"Future date: {future_date_5d}")
            print(f"Computed {len(factors)} factors")
        
        return factors
        
    except Exception as e:
        print(f"Error computing factors: {e}")
        if config and config.debug:
            import traceback
            traceback.print_exc()
        return None

def persist_factors(factors: pd.Series, permno: int, factor_date: pd.Timestamp, config: Config) -> bool:
    """Save factor data to feature store."""
    try:
        # Create output directory path for the *date + permno*  (matches already_computed)
        output_dir = (
            config.feature_store
            / factor_date.strftime('%Y-%m-%d')
            / f"permno={permno}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / "factors.parquet"

        # Convert factors to DataFrame and ensure schema compliance
        df = pd.DataFrame([factors])
        table = pa.Table.from_pandas(df, schema=_FACTOR_SCHEMA)
        
        # Write to parquet with error handling
        try:
            pq.write_table(
                table, 
                file_path,
                compression='zstd',
                use_dictionary=False
            )
            return True
        except Exception as e:
            logger.error(f"Error writing parquet file for date {factor_date}: {e}")
            return False

    except Exception as e:
        logger.error(f"Error saving factors for permno {permno}: {e}")
        return False

def worker(row, config: Config, computed_factors=None) -> Optional[Tuple[int, pd.Series, pd.Timestamp]]:
    """Process one stock-week combination (for parallel processing)."""
    try:
        permno = row['permno']
        reb_date = row['reb_date']
        lookback_start = reb_date - pd.DateOffset(years=config.lookback_years)
        future_date = reb_date + pd.DateOffset(days=10)
        
        # 检查是否已经计算过
        if already_computed(permno, reb_date, computed_factors):
            if config.debug:
                print(f"Already computed: {permno} @ {reb_date}")
            return None
    
        # 从 price cache 加载价格数据
        price_df = dc.load_price_fast(permno, lookback_start, future_date)
        if price_df.empty:
            return None
            
        # 计算因子
        price_df['permno'] = permno
        factors = compute_factors_pti(price_df, reb_date, config)
        
        if factors is None:
            return None
            
        return permno, factors, reb_date
        
    except Exception as e:
        logger.exception(f"Error processing permno {row['permno']}: {e}")
        return None

def calculate_weekly_ranks(results: list) -> list:
    """Calculate weekly ranks for all stocks in the same rebalance period."""
    # Group results by rebalance date
    grouped_results = {}
    for permno, factors, reb_date in results:
        if reb_date not in grouped_results:
            grouped_results[reb_date] = []
        grouped_results[reb_date].append((permno, factors))
    
    # Calculate ranks for each rebalance date
    ranked_results = []
    for reb_date, stocks in grouped_results.items():
        # Create DataFrame with weekly returns
        returns_df = pd.DataFrame([
            {'permno': permno, 'weekly_return': factors['weekly_return']}
            for permno, factors in stocks
        ])
        
        # Calculate ranks (1 is highest return)
        returns_df['weekly_rank'] = returns_df['weekly_return'].rank(ascending=False, method='min')
        
        # Update factors with ranks
        for permno, factors in stocks:
            rank = returns_df.loc[returns_df['permno'] == permno, 'weekly_rank'].iloc[0]
            factors['weekly_rank'] = rank
            ranked_results.append((permno, factors, reb_date))
    
    return ranked_results

def already_computed(permno, reb_date, computed_factors=None) -> bool:
    """Check if factors have already been computed for this stock-date."""
    if computed_factors is not None:
        # Use cache if provided
        date_str = reb_date.strftime('%Y-%m-%d')
        return (date_str, permno) in computed_factors
    
    # Fallback to file system check if no cache
    date_str = reb_date.strftime('%Y-%m-%d')
    expected_path = Path('feature_store') / date_str / f"permno={permno}" / 'factors.parquet'
    return expected_path.exists()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    start_time = time.time()
    
    # Parse dates
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    
    print(f"\n=== 开始因子计算 ===")
    print(f"时间范围: {start_date.date()} 到 {end_date.date()}")
    print(f"重平衡频率: {args.reb}")
    print(f"回溯期: {args.lookback} 年")
    print(f"并行任务数: {args.n_jobs}")
    print("=" * 50)
    
    # Create config
    config = Config(
        start_date=start_date,
        end_date=end_date,
        reb_freq=args.reb,
        lookback_years=args.lookback,
        n_jobs=args.n_jobs,
        debug=args.debug
    )
    
    # Ensure local cache is available
    print("\n1. 准备本地数据缓存...")
    dc.ensure_local_cache(start_date, end_date)
    print("✓ 数据缓存准备完成")
    
    # Load S&P 500 membership
    print("\n2. 加载股票池数据...")
    membership = dc.get_membership_df(start_date, end_date)
    print(f"✓ 加载了 {len(membership)} 只股票的历史数据")
    
    # Generate rebalancing dates
    reb_dates = pd.date_range(start_date, end_date, freq=args.reb)
    print(f"\n3. 生成重平衡日期...")
    print(f"✓ 生成了 {len(reb_dates)} 个重平衡日期")
    
    # Pre-scan existing factors to avoid redundant computation
    print("\n4. 扫描已计算的因子...")
    computed_factors = set()
    for date_dir in config.feature_store.glob('*'):
        if not date_dir.is_dir():
            continue
        for permno_dir in date_dir.glob('permno=*'):
            if not permno_dir.is_dir():
                continue
            if (permno_dir / 'factors.parquet').exists():
                date = date_dir.name
                permno = int(permno_dir.name.split('=')[1])
                computed_factors.add((date, permno))
    print(f"✓ 发现 {len(computed_factors)} 个已计算的因子组合")
    
    # Create expanded grid for processing
    print("\n5. 构建处理网格...")
    grid = []
    for reb_date in reb_dates:
        date_str = reb_date.strftime('%Y-%m-%d')
        # Filter members active on this date
        active = membership[
            (membership['start_date'] <= reb_date) & 
            (membership['end_date'] >= reb_date)
        ]
        
        # Add to grid, skipping already computed combinations
        for _, row in active.iterrows():
            if (date_str, row['permno']) not in computed_factors:
                grid.append({
                    'permno': row['permno'],
                    'ticker': row['ticker'],
                    'reb_date': reb_date
                })
    
    print(f"✓ 创建了 {len(grid)} 个待计算的股票-日期组合")
    
    if len(grid) == 0:
        print("\n所有因子已经计算完成，无需重新计算。")
        return
    
    # Process in parallel
    print("\n6. 开始并行计算因子...")
    with Parallel(n_jobs=config.n_jobs) as parallel:
        results = parallel(
            delayed(worker)(row, config, computed_factors) 
            for row in tqdm(grid, desc="计算进度", mininterval=5)
        )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    print(f"\n✓ 成功计算了 {len(results)} 个股票-日期组合的因子")
    
    # Calculate weekly ranks
    print("\n7. 计算周收益率排名...")
    ranked_results = calculate_weekly_ranks(results)
    print(f"✓ 完成了 {len(ranked_results)} 个股票-日期组合的排名计算")
    
    # Save results
    print("\n8. 保存因子数据...")
    saved = 0
    for permno, factors, reb_date in tqdm(ranked_results, desc="保存进度", mininterval=5):
        if persist_factors(factors, permno, reb_date, config):
            saved += 1
    
    print(f"\n✓ 成功保存了 {saved} 个股票-日期组合的因子")
    
    # Print summary
    total_time = time.time() - start_time
    print("\n=== 处理完成 ===")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每个组合: {total_time/len(grid):.2f} 秒")
    print("=" * 50)
    
    logger.info(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 


import pandas as pd
path = '/Users/coffeer/Desktop/cursor量化/feature_store/2021-01-04/permno=10104'

df = pd.read_parquet(path)

print(df.columns)

