import pytest
import numpy as np
import pandas as pd
from strategy.topN_tree import Rank01Strategy, BinaryTopNStrategy, LambdaRankNStrategy

def test_rank01_strategy_initialization():
    strategy = Rank01Strategy(lr=0.05, max_depth=5, top_n=10)
    assert strategy.lr == 0.05
    assert strategy.max_depth == 5
    assert strategy.top_n == 10
    assert strategy.model is None

def test_binary_topn_strategy_initialization():
    strategy = BinaryTopNStrategy(N=50, lr=0.05, max_depth=5, top_n=10)
    assert strategy.N == 50
    assert strategy.lr == 0.05
    assert strategy.max_depth == 5
    assert strategy.top_n == 10
    assert strategy.model is None

def test_lambda_rank_strategy_initialization():
    strategy = LambdaRankNStrategy(N=10, lr=0.05, max_depth=5, top_n=10)
    assert strategy.N == 10
    assert strategy.lr == 0.05
    assert strategy.max_depth == 5
    assert strategy.top_n == 10
    assert strategy.model is None 