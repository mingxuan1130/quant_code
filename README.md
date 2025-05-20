# Quantitative Trading Strategy Framework

> **TL;DR**: clone → `make setup` → `make demo` → check `output/performance_*.png`

A flexible and extensible framework for implementing and backtesting quantitative trading strategies. The framework supports multiple strategy types including regression-based, binary classification, and ranking-based approaches.

## Features

- Multiple strategy implementations:
  - Rank01: Regression-based strategy
  - BinaryTopN: Binary classification strategy
  - LambdaRank: Learning to rank strategy
- Cross-validation with time series data
- Performance visualization and metrics
- Configurable parameters via environment variables
- Comprehensive logging system

## Directory Structure

```
.
├── config.py           # Configuration and environment variables
├── strategy/          # Strategy implementations
│   └── topN_tree.py   # Main strategy classes
├── data/             # Data loading and processing
├── tests/            # Unit tests
├── logs/             # Log files
├── models/           # Saved model files
└── output/           # Performance plots and results
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quant-strategy.git
cd quant-strategy
```

2. Set up environment:
```bash
cp .env.example .env  # Edit with your paths
make setup
```

3. Run demo:
```bash
make demo
```

4. Check results in `output/performance_*.png`

## Development

- Run tests: `make test`
- Clean cache: `make clean`

## Strategy Types

1. **Rank01 Strategy**
   - Regression-based approach
   - Predicts normalized ranks (0-1)
   - Good for continuous ranking predictions

2. **BinaryTopN Strategy**
   - Binary classification approach
   - Predicts whether a stock will be in top N
   - Good for clear top/bottom selection

3. **LambdaRank Strategy**
   - Learning to rank approach
   - Optimizes for NDCG metric
   - Good for maintaining relative rankings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 