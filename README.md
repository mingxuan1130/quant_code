# Quant Code

A quantitative trading strategy implementation with machine learning models for stock market prediction.

## Features

- Multiple ML-based trading strategies (Rank01, BinaryTopN, LambdaRank)
- Walk-forward cross-validation for time series data
- Technical factor computation and feature engineering
- Performance analysis and visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/mingxuan1130/quant_code.git
cd quant_code

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Download historical data:
```bash
python data/data_download/download_polygon_data.py
```

2. Build factors:
```bash
python data/data_download/build_factors.py --start 2018-01-01 --end 2024-04-30 --reb W-MON --lookback 1 --n_jobs 8
```

3. Run backtest:
```bash
python backtest/backtest.py --strategy rank01 --top_n 10
```

## Project Structure

```
.
├── backtest/           # Backtesting and analysis tools
├── data/              # Data processing and feature engineering
│   ├── data_download/ # Data downloading scripts
│   └── data_loading/  # Data loading utilities
├── docs/              # Documentation
├── scripts/           # Example scripts
├── strategy/          # Trading strategy implementations
└── tests/            # Unit tests
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
