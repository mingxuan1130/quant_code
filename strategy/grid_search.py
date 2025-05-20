import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from data.data_loading.data_loader import DataLoader
from data.data_loading.split import PurgedWalkForwardSplit
from strategy.topN_tree import LambdaRankNStrategy, Evaluator

def evaluate_params(strategy, train_start, train_end, test_start, test_end, splitter, debug=False):
    """Evaluate the performance of a set of parameters"""
    # Train the model
    strategy.train(train_start, train_end, splitter=splitter, debug=debug)
    
    # Make predictions on test set
    predictions, metadata = strategy.predict(test_start, test_end)
    
    # Calculate evaluation metrics
    evaluator = Evaluator(top_n=strategy.top_n)
    dates, period_returns = evaluator.period_returns(metadata['df'])
    cum_ret, sharpe = evaluator.summary(period_returns)
    
    return {
        'cumulative_return': cum_ret,
        'sharpe_ratio': sharpe,
        'period_returns': period_returns,
        'dates': dates
    }

def grid_search(train_start, train_end, test_start, test_end, param_grid, debug=False):
    """Perform grid search"""
    # Initialize data loader
    data_loader = DataLoader(Path("/Users/coffeer/Desktop/cursor_quant/data/data storage /feature_store"))
    
    # Initialize cross-validation splitter
    splitter = PurgedWalkForwardSplit(
        n_splits=3,
        train_period=12,  # 12 weeks training
        test_period=4,    # 4 weeks testing
        embargo=1         # 1 week embargo
    )
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    results = []
    best_sharpe = -np.inf
    best_params = None
    best_returns = None
    best_dates = None
    
    print(f"\nStarting grid search with {len(param_combinations)} parameter combinations...")
    
    for params in tqdm(param_combinations, desc="Grid Search"):
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))
        
        # Initialize strategy
        strategy = LambdaRankNStrategy(**param_dict)
        
        # Evaluate parameters
        metrics = evaluate_params(
            strategy, train_start, train_end, 
            test_start, test_end, splitter, debug
        )
        
        # Record results
        result = {
            'params': param_dict,
            'cumulative_return': metrics['cumulative_return'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }
        results.append(result)
        
        # Update best parameters
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = param_dict
            best_returns = metrics['period_returns']
            best_dates = metrics['dates']
            
            if debug:
                print(f"\nFound new best parameters:")
                print(f"Parameters: {best_params}")
                print(f"Sharpe Ratio: {best_sharpe:.2f}")
                print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    
    # Sort results by Sharpe Ratio
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return {
        'all_results': results,
        'best_params': best_params,
        'best_sharpe': best_sharpe,
        'best_returns': best_returns,
        'best_dates': best_dates
    }

def main():
    # Define parameter grid
    param_grid = {
        'N': [5, 10, 20],           # Ranking target
        'lr': [0.01, 0.05, 0.1],    # Learning rate
        'max_depth': [3, 5, 7],     # Maximum tree depth
        'top_n': [5, 10, 20]        # Select top N stocks
    }
    
    # Set time range
    train_start = "2020-02-03"
    train_end = "2023-12-23"
    test_start = "2023-12-23"
    test_end = "2024-12-23"
    
    # Execute grid search
    results = grid_search(
        train_start, train_end,
        test_start, test_end,
        param_grid,
        debug=True
    )
    
    # Print results
    print("\nGrid Search Results:")
    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    print(f"\nBest Sharpe Ratio: {results['best_sharpe']:.2f}")
    
    print("\nTop 5 Parameter Combinations:")
    for i, result in enumerate(results['all_results'][:5], 1):
        print(f"\n{i}. Parameter Combination:")
        for param, value in result['params'].items():
            print(f"   {param}: {value}")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"   Cumulative Return: {result['cumulative_return']:.2%}")

if __name__ == "__main__":
    main() 