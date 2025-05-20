import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from strategy.topN_tree import Rank01Strategy, BinaryTopNStrategy, LambdaRankNStrategy
from data.data_loading.data_loader import DataLoader
from data.data_loading.split import PurgedWalkForwardSplit

class TestStrategySmoke(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.data_loader = DataLoader(Path("data/data storage/feature_store"))
        self.splitter = PurgedWalkForwardSplit(
            n_splits=1,
            train_period=4,  # 4 weeks training
            test_period=1,   # 1 week testing
            embargo=0        # No embargo for quick test
        )
        
        # Test parameters
        self.train_start = "2023-01-01"
        self.train_end = "2023-01-31"
        self.test_start = "2023-02-01"
        self.test_end = "2023-02-07"
        
    def test_rank01_strategy(self):
        """Test Rank01Strategy basic functionality"""
        strategy = Rank01Strategy(lr=0.05, max_depth=3, top_n=5)
        
        # Train strategy
        strategy.train(self.train_start, self.train_end, splitter=self.splitter)
        
        # Make predictions
        predictions, metadata = strategy.predict(self.test_start, self.test_end)
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(metadata)
        self.assertIn('df', metadata)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(metadata['df']))
        
        # Check no NaN values
        self.assertFalse(np.isnan(predictions).any())
        
    def test_binary_strategy(self):
        """Test BinaryTopNStrategy basic functionality"""
        strategy = BinaryTopNStrategy(N=10, lr=0.05, max_depth=3, top_n=5)
        
        # Train strategy
        strategy.train(self.train_start, self.train_end, splitter=self.splitter)
        
        # Make predictions
        predictions, metadata = strategy.predict(self.test_start, self.test_end)
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(metadata)
        self.assertIn('df', metadata)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(metadata['df']))
        
        # Check predictions are probabilities
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))
        
    def test_lambda_rank_strategy(self):
        """Test LambdaRankNStrategy basic functionality"""
        strategy = LambdaRankNStrategy(N=10, lr=0.05, max_depth=3, top_n=5)
        
        # Train strategy
        strategy.train(self.train_start, self.train_end, splitter=self.splitter)
        
        # Make predictions
        predictions, metadata = strategy.predict(self.test_start, self.test_end)
        
        # Basic assertions
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(metadata)
        self.assertIn('df', metadata)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(metadata['df']))
        
        # Check no NaN values
        self.assertFalse(np.isnan(predictions).any())

if __name__ == '__main__':
    unittest.main() 