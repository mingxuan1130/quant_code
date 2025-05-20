#!/usr/bin/env python
"""
Backtest script to compare model predictions with buy-and-hold strategy
"""
from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

from models.strategy import load_raw, Rank01Strategy, BinaryTopNStrategy, LambdaRankNStrategy, FACTOR_COLS
from backtest.backtest_analysis import BacktestAnalyzer

def compute_drawdown(cum: pd.Series) -> Tuple[pd.Series, float]:
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd, dd.min()

def annualize_ret(weekly_ret: np.ndarray) -> float:
    return (np.prod(1 + weekly_ret) ** (52/len(weekly_ret))) - 1

def annualize_vol(weekly_ret: np.ndarray) -> float:
    return np.std(weekly_ret, ddof=1) * np.sqrt(52)

def train_model(train_start: str, train_end: str, lr: float = 0.05, max_depth: int = 5):
    """Train model using specified parameters"""
    strategy = LGBMStrategy(lr=lr, max_depth=max_depth)
    strategy.train(train_start, train_end)
    return strategy.model, strategy.metadata

def calculate_returns(predictions: np.ndarray, actual_returns: np.ndarray, top_n: int = 10) -> float:
    """Calculate returns based on top N predictions"""
    # Get indices of top N predictions
    top_indices = np.argsort(predictions)[-top_n:]
    # Calculate average return of top N stocks
    return np.mean(actual_returns[top_indices])

def backtest_strategy(test_start: str, test_end: str, model, metadata: dict, top_n: int):
    """Backtest the strategy"""
    # Load test data
    df = load_raw(test_start, test_end)
    X_test = df[FACTOR_COLS].to_numpy(float)
    y_test = df['weekly_return'].to_numpy()
    
    # Make predictions
    preds = model.predict(X_test)
    df = pd.DataFrame({
        'date': df['factor_date'],
        'pred': preds,
        'ret': y_test
    })
    return df

def calculate_returns_df(df: pd.DataFrame, top_n: int):
    # Strategy: Each week, select top_n stocks with the highest pred, equally weighted
    strat = (
        df.groupby('date')
          .apply(lambda g: g.nlargest(top_n, 'pred')['ret'].mean())
          .sort_index()
    )
    # Benchmark: Each week, equally weight all stocks
    bench = df.groupby('date')['ret'].mean().sort_index()
    # Buy and Hold: Each week, buy all stocks with equal weight
    buy_hold = df.groupby('date')['ret'].mean().sort_index()
    return strat, bench, buy_hold

def plot_performance(strategy: pd.Series, bench: pd.Series, fname='perf.png'):
    cum_s = (1 + strategy).cumprod()
    cum_b = (1 + bench).cumprod()
    dd_s, mdd_s = compute_drawdown(cum_s)
    dd_b, mdd_b = compute_drawdown(cum_b)

    fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True,
                            gridspec_kw={'height_ratios':[2,1,1]})

    axs[0].plot(cum_s.index, cum_s*100, label='Strategy')
    axs[0].plot(cum_b.index, cum_b*100, label='Benchmark', ls='--')
    axs[0].set_ylabel('Cumulative %'); axs[0].legend(); axs[0].grid()

    axs[1].plot(dd_s.index, dd_s*100, label=f'Strat MDD {mdd_s:.1%}')
    axs[1].plot(dd_b.index, dd_b*100, label=f'Bench MDD {mdd_b:.1%}', ls='--')
    axs[1].set_ylabel('Drawdown %'); axs[1].legend(); axs[1].grid()

    # rolling 26w sharpe
    roll = strategy.rolling(26).apply(lambda x: np.mean(x)/np.std(x, ddof=1) if x.std(ddof=1)>0 else 0)
    axs[2].plot(roll.index, roll, label='Rolling Sharpe 26w')
    axs[2].axhline(0, color='grey')
    axs[2].set_ylabel('Sharpe'); axs[2].grid()
    plt.tight_layout(); plt.savefig(fname); plt.close()

def plot_dist(strategy: pd.Series, fname='hist.png'):
    plt.figure(figsize=(6,4))
    plt.hist(strategy*100, bins=30, alpha=.7)
    plt.title('Weekly Return Distribution'); plt.xlabel('%'); plt.ylabel('freq')
    plt.tight_layout(); plt.savefig(fname); plt.close()

def plot_feat_imp(model, fname='feat_importance.png', top_k=20):
    booster = model[-1].booster_
    imp = booster.feature_importance(importance_type='gain')
    cols = range(len(imp))
    s = pd.Series(imp, index=cols).nlargest(top_k)
    plt.figure(figsize=(6,6))
    s[::-1].plot(kind='barh')
    plt.title('LightGBM Feature Importance (gain)'); plt.tight_layout()
    plt.savefig(fname); plt.close()

def plot_returns(strategy_returns: float, buy_hold_returns: float, test_period: str):
    """Plot comparison of returns"""
    plt.figure(figsize=(10, 6))
    strategies = ['Model Strategy', 'Buy & Hold']
    returns = [strategy_returns, buy_hold_returns]
    
    plt.bar(strategies, returns)
    plt.title(f'Strategy Comparison ({test_period})')
    plt.ylabel('Returns')
    
    # Add return values on top of bars
    for i, v in enumerate(returns):
        plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
    
    plt.savefig('strategy_comparison.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Backtest strategy comparison")
    parser.add_argument("--train_start", type=str, default="2020-01-01", help="Training start date")
    parser.add_argument("--train_end", type=str, default="2023-12-31", help="Training end date")
    parser.add_argument("--test_start", type=str, default="2024-01-01", help="Test start date")
    parser.add_argument("--test_end", type=str, default="2024-03-31", help="Test end date")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=5, help="Max tree depth")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top stocks to select")
    parser.add_argument("--strategy", type=str, default="rank01", choices=["rank01", "binarytopn", "lambdarank"], help="Strategy to use")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for analysis plots")
    args = parser.parse_args()
    
    print(f"Training {args.strategy} model on {args.train_start} to {args.train_end}")
    if args.strategy == "rank01":
        strategy = Rank01Strategy(lr=args.lr, max_depth=args.max_depth, top_n=args.top_n)
    elif args.strategy == "binarytopn":
        strategy = BinaryTopNStrategy(N=50, lr=args.lr, max_depth=args.max_depth, top_n=args.top_n)
    elif args.strategy == "lambdarank":
        strategy = LambdaRankNStrategy(N=args.top_n, lr=args.lr, max_depth=args.max_depth, top_n=args.top_n)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    strategy.train(args.train_start, args.train_end)
    model = strategy.model
    
    print(f"\nBacktesting on {args.test_start} to {args.test_end}")
    df_week = backtest_strategy(args.test_start, args.test_end, model, None, args.top_n)
    strat, bench, buy_hold = calculate_returns_df(df_week, args.top_n)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer with both benchmark and buy & hold
    analyzer = BacktestAnalyzer(strat, buy_hold)  # Use buy & hold as benchmark
    
    # Generate all plots
    feature_importance = None
    if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        booster = model.named_steps['regressor'].booster_
        feature_importance = pd.Series(booster.feature_importance(importance_type='gain'))
    elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        booster = model.named_steps['classifier'].booster_
        feature_importance = pd.Series(booster.feature_importance(importance_type='gain'))
    elif hasattr(model, 'named_steps') and 'ranker' in model.named_steps:
        booster = model.named_steps['ranker'].booster_
        feature_importance = pd.Series(booster.feature_importance(importance_type='gain'))
    analyzer.generate_all_plots(
        output_dir=output_dir,
        feature_importance=feature_importance
    )
    
    # Print performance metrics
    metrics = analyzer.get_performance_metrics()
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric == 'Sharpe Ratio':
            print(f"{metric}: {value:.2f}")
        elif isinstance(value, float):
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main() 

