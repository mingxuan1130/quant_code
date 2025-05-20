#!/usr/bin/env python
"""
Strategy implementations including buy-and-hold and tree model strategies
"""
# 1) __future__ imports must come first
from __future__ import annotations

# 2) standard libraries
from abc import ABC, abstractmethod
import argparse
import warnings
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Tuple, List

# 3) third-party libraries
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from data.data_loading.data_loader import DataLoader

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

# --------------------------------- Paths ---------------------------------
FEATURE_STORE = Path("data/feature_store")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# --------------------------------- Constants ---------------------------------
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

class BaseStrategy(ABC):
    """Base class for all strategies"""
    def __init__(self, data_loader: DataLoader, name: str):
        self.data = data_loader
        self.name = name
    
    @abstractmethod
    def build_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """Subclasses implement their own target building logic here"""
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **meta):
        """Subclasses implement their own model fitting logic here"""
        raise NotImplementedError
    
    def train(self, start: str, end: str, *, splitter=None, **kw):
        """Common training pipeline"""
        raw = self.data.load(start, end, dropna_cols=['weekly_rank','weekly_return'],
                           show_progress=kw.get("debug", False))
        X = raw[FACTOR_COLS].to_numpy(float)
        y, meta = self.build_target(raw)

        if splitter is None:                           # Default to fitting on all data
            self.fit(X, y, **meta)
            return

        for fold, (tr, va) in enumerate(splitter.split(raw.factor_date)):
            self.fit(X[tr], y[tr], **{k: v[tr] if isinstance(v, np.ndarray) else v
                                      for k, v in meta.items()})
            # You can implement early stopping / parameter tuning / validation score recording here
    
    @abstractmethod
    def predict(self, test_start: str, test_end: str) -> Tuple[np.ndarray, Dict]:
        """Make predictions"""
        raise NotImplementedError

class Evaluator:
    """Common evaluator for all strategies"""
    def __init__(self, top_n):
        self.top_n = top_n
    
    def period_returns(self, df_scores: pd.DataFrame):
        """Calculate period returns based on top N scores"""
        rets = []
        dates = []
        for d, g in df_scores.groupby('factor_date'):
            top = g.nlargest(self.top_n, 'score')
            rets.append(top.weekly_return.mean())
            dates.append(d)
        return dates, rets
    
    def summary(self, rets):
        """Calculate summary statistics"""
        arr = np.array(rets)
        cum = np.prod(1 + arr) - 1
        sharpe = np.sqrt(52) * arr.mean() / arr.std()  # weekly freq
        return cum, sharpe

class Rank01Strategy(BaseStrategy):
    """Tree model strategy implementation using normalized rank scores"""
    def __init__(self, lr: float = 0.05, max_depth: int = 5, top_n: int = 10):
        super().__init__(DataLoader(FEATURE_STORE), "Rank01 Strategy")
        self.lr = lr
        self.max_depth = max_depth
        self.top_n = top_n
        self.model = None
    
    def build_target(self, df):
        """map weekly rank to 0 to 1"""
        y = df["weekly_rank"].to_numpy(dtype=float)
        y = (y - 1) / (len(y) - 1)
        groups = df.groupby("factor_date").size().to_list()
        return y, {"groups": groups}
    
    def fit(self, X, y, **meta):
        """Fit the regression model"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=self.lr,
                max_depth=self.max_depth,
                random_state=42,
                verbose=-1
            )
            
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', regressor)
            ])
            
            self.model.fit(X, y)
    
    def predict(self, test_start: str, test_end: str) -> Tuple[np.ndarray, Dict]:
        """Make predictions using the trained model"""
        df = self.data.load(test_start, test_end)
        X = df[FACTOR_COLS].to_numpy(float)
        predictions = self.model.predict(X)
        df['score'] = predictions
        return predictions, {'df': df}

class BinaryTopNStrategy(BaseStrategy):
    """Binary classification strategy for top N stocks"""
    def __init__(self, N, lr=0.05, max_depth=5, top_n=10):
        super().__init__(DataLoader(FEATURE_STORE), "BinaryTopN")
        self.N = N
        self.top_n = top_n
        self.model = None
        self.lr = lr
        self.max_depth = max_depth
    
    def build_target(self, df):
        """Build binary classification target"""
        df = df.copy()
        df["y"] = (df.groupby("factor_date")["weekly_rank"]
                     .transform(lambda r: r <= self.N)).astype(int)
        return df["y"].to_numpy(int), {}
    
    def fit(self, X, y, **_):
        """Fit the binary classifier"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=self.lr,
                max_depth=self.max_depth,
                random_state=42,
                verbose=-1
            )
            
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            
            self.model.fit(X, y)
    
    def predict(self, test_start: str, test_end: str) -> Tuple[np.ndarray, Dict]:
        """Make predictions using the trained model"""
        df = self.data.load(test_start, test_end)
        X = df[FACTOR_COLS].to_numpy(float)
        predictions = self.model.predict_proba(X)[:, 1]
        df['score'] = predictions
        return predictions, {'df': df}

class LambdaRankNStrategy(BaseStrategy):
    """LambdaRank strategy for NDCG@N"""
    def __init__(self, N=10, lr=0.05, max_depth=5, top_n=10):
        super().__init__(DataLoader(FEATURE_STORE), "LambdaRank")
        self.N = N
        self.top_n = top_n
        self.model = None
        self.lr = lr
        self.max_depth = max_depth
    
    def build_target(self, df):
        """Build ranking target using normalized relevance scores"""
        # Calculate group sizes
        group_id = df["factor_date"].values  # Group ID for each row
        # Map ranks to [0,100] using relevance = 1 - (rank - 1) / (group_size - 1)
        y = df.groupby("factor_date").apply(
            lambda g: (1 - (g["weekly_rank"] - 1) / (len(g) - 1)) * 100
        ).reset_index(level=0, drop=True)
        y = y.round().astype(int)
        return y.to_numpy(), {"group_id": group_id}
    
    def fit(self, X, y, **meta):
        """Fit the ranker"""
        group_id = meta["group_id"]
        # Regroup to get group size list
        unique, group_sizes = np.unique(group_id, return_counts=True)
        groups = group_sizes.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranker = lgb.LGBMRanker(
                objective='lambdarank',
                metric='ndcg',
                ndcg_eval_at=[self.N],
                n_estimators=600,
                learning_rate=self.lr,
                max_depth=self.max_depth,
                random_state=42,
                verbose=-1,
                label_gain=[i for i in range(101)]  # Gain values from 0 to 100
            )
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('ranker', ranker)
            ])
            self.model.fit(X, y, ranker__group=groups)
    
    def predict(self, test_start: str, test_end: str) -> Tuple[np.ndarray, Dict]:
        """Make predictions using the trained model"""
        df = self.data.load(test_start, test_end)
        X = df[FACTOR_COLS].to_numpy(float)
        predictions = self.model.predict(X)
        df['score'] = predictions
        return predictions, {'df': df}

def plot_performance(dates, returns_dict, label_dict, filename_prefix="performance"):
    """Plot cumulative returns, drawdown, and stock price comparison"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # Plot cumulative returns
    for key, rets in returns_dict.items():
        cum = np.cumprod(1 + np.array(rets)) - 1
        ax1.plot(dates, cum * 100, label=label_dict.get(key, key))
    ax1.set_title("Cumulative Return")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot drawdown
    for key, rets in returns_dict.items():
        cum = np.cumprod(1 + np.array(rets))
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        ax2.plot(dates, dd * 100, label=label_dict.get(key, key))
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True)
    
    # Plot stock price comparison
    for key, rets in returns_dict.items():
        if key == "Buy & Hold":
            continue
        cum = np.cumprod(1 + np.array(rets))
        ax3.plot(dates, cum, label=f"{label_dict.get(key, key)} Strategy")
    
    # Add buy & hold as reference
    if "Buy & Hold" in returns_dict:
        cum_bh = np.cumprod(1 + np.array(returns_dict["Buy & Hold"]))
        ax3.plot(dates, cum_bh, label="Buy & Hold", linestyle='--', alpha=0.7)
    
    ax3.set_title("Strategy vs Buy & Hold")
    ax3.set_ylabel("Portfolio Value (Starting at 1)")
    ax3.legend()
    ax3.grid(True)
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_performance.png")
    plt.close()

def load_raw(start_date: str, end_date: str, *, debug=False) -> pd.DataFrame:
    """Return a DataFrame containing *raw* factor rows within [start, end]"""
    try:
        if debug:
            print(f"Loading data from {start_date} to {end_date}...")
            
        # Convert string dates to date objects for comparison
        d0, d1 = pd.to_datetime(start_date), pd.to_datetime(end_date)
        tables = []

        # Iterate over all parquet files in feature_store
        for file in tqdm(list(FEATURE_STORE.glob("*.parquet")), desc="Scanning files"):
            try:
                file_date_str = file.stem.split("-")[0:3]
                file_date = pd.to_datetime("-".join(file_date_str))
                if not (d0 <= file_date <= d1):
                    continue
            except Exception as e:
                if debug:
                    print(f"Skipping file {file} due to date parsing error: {e}")
                continue

            tbl = pq.read_table(file)
            tables.append(tbl)

        if not tables:
            raise RuntimeError("No parquet files matched the date range.")

        batches = pa.concat_tables(tables, promote=True)
        df = batches.to_pandas()
        
        if debug:
            print(f"Loaded {len(df)} samples")
            print(f"Unique dates: {df['factor_date'].nunique()}")
            print(f"Unique stocks: {df['permno'].nunique()}")
        
        # Handle NaN values
        # 1. Fill NaN in features with 0
        feature_cols = [col for col in df.columns if col not in ['permno', 'factor_date', 'weekly_return', 'weekly_rank']]
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # 2. Remove rows with NaN in weekly_rank or weekly_return
        df = df.dropna(subset=['weekly_rank', 'weekly_return'])
        
        if debug:
            print(f"After cleaning: {len(df)} samples")
            print(f"NaN in features: {df[feature_cols].isna().sum().sum()}")
            print(f"NaN in weekly_rank: {df['weekly_rank'].isna().sum()}")
            print(f"NaN in weekly_return: {df['weekly_return'].isna().sum()}")
        
        return df
    
    except Exception as e:
        print(f"Error loading samples: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    

def main():
    parser = argparse.ArgumentParser(description="Run strategy comparison")
    parser.add_argument("--train_start", type=str, default="2020-01-01", help="Training start date")
    parser.add_argument("--train_end", type=str, default="2023-12-31", help="Training end date")
    parser.add_argument("--test_start", type=str, default="2024-01-01", help="Test start date")
    parser.add_argument("--test_end", type=str, default="2024-03-31", help="Test end date")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for tree model")
    parser.add_argument("--max_depth", type=int, default=5, help="Max tree depth")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top stocks to select")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for data validation")
    args = parser.parse_args()
    
    # Initialize CV splitter
    import sys
    sys.path.append("data/processed/feature_store")
    from cv import PurgedWalkForwardCV
    
    splitter = PurgedWalkForwardCV(
        n_splits=6,          # 总共做6个walk-forward步
        train_period=52*2,   # 每步训练用2年数据
        test_period=4,       # 每步测试用4周数据
        embargo=1            # 设置1周的embargo期
    )
    
    # Initialize strategies
    strategies = {
        "Rank01": Rank01Strategy(lr=args.lr, max_depth=args.max_depth, top_n=args.top_n),
        "BinaryTopN": BinaryTopNStrategy(N=50, lr=args.lr, max_depth=args.max_depth, top_n=args.top_n),
        "LambdaRank": LambdaRankNStrategy(N=args.top_n, lr=args.lr, max_depth=args.max_depth, top_n=args.top_n)
    }
    
    results = {}
    all_period_returns = {}
    all_period_dates = None
    label_dict = {
        "Rank01": f"Rank01 Strategy Top{args.top_n}",
        "BinaryTopN": f"BinaryTopN Strategy Top{args.top_n}",
        "LambdaRank": f"LambdaRank Strategy Top{args.top_n}"
    }
    
    # Train and evaluate each strategy
    for name, strategy in strategies.items():
        print(f"\nTraining {name} strategy...")
        strategy.train(args.train_start, args.train_end, splitter=splitter, debug=args.debug)
        
        print(f"Backtesting {name} strategy...")
        predictions, metadata = strategy.predict(args.test_start, args.test_end)
        
        # Use common evaluator
        evaluator = Evaluator(top_n=args.top_n)
        dates, period_returns = evaluator.period_returns(metadata['df'])
        cum_return, sharpe = evaluator.summary(period_returns)
        
        results[name] = {
            'cumulative_return': cum_return,
            'sharpe_ratio': sharpe
        }
        all_period_returns[name] = period_returns
        if all_period_dates is None:
            all_period_dates = dates
    
    # Print results
    print("\nResults:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Cumulative Return: {metrics['cumulative_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot performance metrics
    plot_performance(all_period_dates, all_period_returns, label_dict)

if __name__ == "__main__":
    main() 


