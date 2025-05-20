import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from data.data_loading.data_loader import DataLoader
from data.data_loading.split import PurgedWalkForwardSplit
from strategy.topN_tree import Rank01Strategy, Evaluator

def main():
    # Initialize data loader with correct path
    data_loader = DataLoader(Path("/Users/coffeer/Desktop/cursor_quant/data/data storage/feature_store"))
    
    # Initialize strategy
    strategy = Rank01Strategy(lr=0.05, max_depth=5, top_n=10)
    
    # Initialize splitter
    splitter = PurgedWalkForwardSplit(
        n_splits=3,
        train_period=12,  # 12 weeks training
        test_period=4,    # 4 weeks testing
        embargo=1         # 1 week embargo
    )
    
    # Training period - adjust to match your data
    train_start = "2020-02-03"  # Match the date in your parquet file
    train_end = "2023-12-23"
    
    # Train the strategy
    print("Training strategy...")
    strategy.train(train_start, train_end, splitter=splitter, debug=True)
    
    # Test period
    test_start = "2023-12-23"
    test_end = "2024-12-23"
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, meta = strategy.predict(test_start, test_end)
    
    # Evaluate results
    print("\nEvaluating results...")
    evaluator = Evaluator(top_n=10)
    dates, returns = evaluator.period_returns(meta['df'])
    cum_ret, sharpe = evaluator.summary(returns)
    
    print(f"\nResults:")
    print(f"Cumulative Return: {cum_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    main() 