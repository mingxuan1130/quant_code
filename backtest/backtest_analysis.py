#!/usr/bin/env python
"""
Comprehensive backtest analysis module with advanced visualizations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from pathlib import Path

class BacktestAnalyzer:
    def __init__(self, strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize the analyzer with strategy returns and optional benchmark returns
        
        Args:
            strategy_returns: pd.Series with datetime index and strategy returns
            benchmark_returns: Optional pd.Series with datetime index and benchmark returns
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.cum_returns = (1 + strategy_returns).cumprod()
        if benchmark_returns is not None:
            self.benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """Plot cumulative returns (equity curve)"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.cum_returns.index, self.cum_returns, label='Strategy', linewidth=2)
        if self.benchmark_returns is not None:
            plt.plot(self.benchmark_cum_returns.index, self.benchmark_cum_returns, 
                    label='Benchmark', linestyle='--', linewidth=2)
        
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        """Plot drawdown curve"""
        roll_max = self.cum_returns.cummax()
        drawdown = self.cum_returns / roll_max - 1
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=2)
        
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_returns_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot returns distribution with statistical annotations"""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(self.strategy_returns, bins=50, kde=True)
        
        # Add statistical annotations
        mean = self.strategy_returns.mean()
        std = self.strategy_returns.std()
        
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2%}')
        plt.axvline(mean + std, color='green', linestyle=':', label=f'+1σ: {(mean + std):.2%}')
        plt.axvline(mean - std, color='green', linestyle=':', label=f'-1σ: {(mean - std):.2%}')
        
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_rolling_metrics(self, window: int = 52, save_path: Optional[str] = None) -> None:
        """Plot rolling annualized metrics"""
        # Calculate rolling metrics
        rolling_mean = self.strategy_returns.rolling(window).mean()
        rolling_std = self.strategy_returns.rolling(window).std()
        
        # Annualize
        rolling_ann_ret = rolling_mean * 52
        rolling_ann_vol = rolling_std * np.sqrt(52)
        rolling_sharpe = rolling_ann_ret / rolling_ann_vol
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot annualized returns
        ax1.plot(rolling_ann_ret.index, rolling_ann_ret, label='Annualized Returns')
        ax1.set_title('Rolling Annualized Returns')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot annualized volatility
        ax2.plot(rolling_ann_vol.index, rolling_ann_vol, label='Annualized Volatility', color='orange')
        ax2.set_title('Rolling Annualized Volatility')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot rolling Sharpe ratio
        ax3.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe Ratio', color='green')
        ax3.set_title('Rolling Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.Series, top_n: int = 20, 
                              save_path: Optional[str] = None) -> None:
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        
        # Get top N features
        top_features = feature_importance.nlargest(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_all_plots(self, output_dir: str, feature_importance: Optional[pd.Series] = None) -> None:
        """Generate all plots and save them to the specified directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_equity_curve(output_path / 'equity_curve.png')
        self.plot_drawdown(output_path / 'drawdown.png')
        self.plot_returns_distribution(output_path / 'returns_distribution.png')
        self.plot_rolling_metrics(save_path=output_path / 'rolling_metrics.png')
        
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, save_path=output_path / 'feature_importance.png')
    
    def get_performance_metrics(self) -> dict:
        """Calculate and return key performance metrics"""
        total_return = self.cum_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(self.strategy_returns)) - 1
        annualized_vol = self.strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Calculate drawdown metrics
        roll_max = self.cum_returns.cummax()
        drawdown = self.cum_returns / roll_max - 1
        max_drawdown = drawdown.min()
        
        # Calculate additional metrics
        win_rate = (self.strategy_returns > 0).mean()
        profit_factor = abs(self.strategy_returns[self.strategy_returns > 0].sum() / 
                          self.strategy_returns[self.strategy_returns < 0].sum())
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor
        } 