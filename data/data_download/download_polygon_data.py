#!/usr/bin/env python
"""
Download historical price data from Polygon.io API
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm
import threading

# -------------------- Configuration --------------------
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    print("Error: POLYGON_API_KEY environment variable not set")
    sys.exit(1)

OUTPUT_DIR = Path("data/polygon_data")
START_DATE = (dt.date.today() - dt.timedelta(days=730)).isoformat()
END_DATE = dt.date.today().isoformat()

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Download historical price data for a single stock"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "OK" or not data["results"]:
            return None
            
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "transactions"
        })
        
        return df[["date", "open", "high", "low", "close", "volume", "transactions"]]
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def download_sp500_data(tickers: List[str], start_date: str, end_date: str):
    """Download data for all S&P 500 stocks"""
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        df = download_stock_data(ticker, start_date, end_date)
        if df is not None:
            output_file = OUTPUT_DIR / f"{ticker}.csv"
            df.to_csv(output_file, index=False)
        time.sleep(0.2)  # Rate limiting

def main():
    """Main entry point"""
    # Load S&P 500 tickers
    sp500_file = Path("data/sp500_tickers.txt")
    if not sp500_file.exists():
        print(f"Error: {sp500_file} not found")
        sys.exit(1)
        
    with open(sp500_file) as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    print(f"Downloading data for {len(tickers)} stocks")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    download_sp500_data(tickers, START_DATE, END_DATE)
    print("Download complete!")

if __name__ == "__main__":
    main() 