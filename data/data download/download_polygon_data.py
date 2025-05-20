#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download S&P 500 daily data for the past two years using multiple Polygon API tokens.
"""

# Requirements:
#   pip install pandas polygon-api-client tenacity pyarrow requests tqdm

import os
import time
import datetime as dt
import pandas as pd
from polygon import RESTClient
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# -------------------- Configuration --------------------
# Load tokens from local environment file
tokens_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
TOKENS = []

# Read tokens from the env file
try:
    with open(tokens_file, 'r') as f:
        for line in f:
            if line.strip().startswith('POLYGON_TOKEN'):
                # Extract the token value (assumes format like POLYGON_TOKEN=value)
                key, value = line.strip().split('=', 1)
                TOKENS.append(value.strip().strip('"').strip("'"))
    
    # Use only the first 3 tokens
    TOKENS = TOKENS[:3]
    
    if not TOKENS:
        raise ValueError("No Polygon tokens found in the env file")
except Exception as e:
    print(f"Error loading tokens: {e}")
    exit(1)

OUTPUT_DIR = "/Users/coffeer/Desktop/cursor_quant/polygon loading/polygon_data"
START_DATE = (dt.date.today() - dt.timedelta(days=730)).isoformat()
END_DATE = dt.date.today().isoformat()
CALLS_PER_MIN = 5  # per token
SLEEP_INTERVAL = 60.0 / CALLS_PER_MIN  # seconds between calls per token

# Test mode - only download 1 ticker per token (total 3 tickers)
TEST_MODE = False
TICKERS_PER_TOKEN = 1 if TEST_MODE else None  # None means all tickers

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Helper Functions --------
def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia and normalize symbols."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    df = tables[0]
    # Replace dots in symbols (e.g., BRK.B → BRK-B)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df["Symbol"].tolist()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def fetch_ticker_data(client: RESTClient, ticker: str) -> pd.DataFrame:
    aggs = client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=START_DATE,
        to=END_DATE,
        limit=50000,
        adjusted=True
    )
    bars = list(aggs)
    if not bars:
        raise ValueError(f"No data for ticker {ticker}")
    df = pd.DataFrame([{
        "date": pd.to_datetime(bar.timestamp, unit="ms").date(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    } for bar in bars])
    return df

# -------- Progress tracking --------
class ProgressTracker:
    def __init__(self, total_tickers):
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total_tickers, desc="Downloading data", unit="ticker")
        self.start_time = time.time()
        self.total_tickers = total_tickers
        self.completed = 0
    
    def update(self, n=1):
        with self.lock:
            self.completed += n
            self.pbar.update(n)
            
            # Calculate and display estimated time remaining
            if self.completed > 0:
                elapsed = time.time() - self.start_time
                tickers_per_sec = self.completed / elapsed
                remaining = (self.total_tickers - self.completed) / tickers_per_sec if tickers_per_sec > 0 else 0
                
                # Format remaining time
                remaining_str = ""
                if remaining > 3600:
                    remaining_str = f"{remaining/3600:.1f}h"
                elif remaining > 60:
                    remaining_str = f"{remaining/60:.1f}m"
                else:
                    remaining_str = f"{remaining:.0f}s"
                
                self.pbar.set_postfix({"est. remaining": remaining_str})
    
    def close(self):
        self.pbar.close()

# -------- Worker --------
def worker(token: str, tickers: list, progress_tracker):
    client = RESTClient(token)
    for ticker in tickers:
        try:
            df = fetch_ticker_data(client, ticker)
            filename = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
            df.to_parquet(filename, index=False)
            print(f"[{token[:4]}] Saved {ticker} ({len(df)} rows)")
            progress_tracker.update()
        except Exception as e:
            print(f"[{token[:4]}] Error fetching {ticker}: {e}")
            progress_tracker.update()
        time.sleep(SLEEP_INTERVAL)  # Respect API rate limit: 5 calls per minute

# -------- Main --------
def main():
    print(f"Using {len(TOKENS)} Polygon API tokens")
    tickers = get_sp500_tickers()
    
    if TEST_MODE:
        print(f"TEST MODE: Only downloading first {TICKERS_PER_TOKEN} tickers per token")
    
    print(f"Downloading data for {len(tickers)} S&P 500 stocks")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"API rate limit: {CALLS_PER_MIN} calls per minute per token")
    
    # Wait 20 seconds before starting (reduced from 60)
    print(f"Waiting 20 seconds before starting download...")
    for i in range(20, 0, -1):
        print(f"\rStarting in {i} seconds...", end="")
        time.sleep(1)
    print("\rDownload starting now!                ")
    
    # Split tickers into N groups (one per token)
    groups = [tickers[i::len(TOKENS)] for i in range(len(TOKENS))]
    
    # If in test mode, limit each group to specified number of tickers
    if TEST_MODE and TICKERS_PER_TOKEN:
        groups = [group[:TICKERS_PER_TOKEN] for group in groups]
    
    # Calculate total tickers
    total_tickers = sum(len(group) for group in groups)
    
    # Initialize progress tracker
    progress = ProgressTracker(total_tickers)
    
    with ThreadPoolExecutor(max_workers=len(TOKENS)) as executor:
        for token, group in zip(TOKENS, groups):
            executor.submit(worker, token, group, progress)
    
    progress.close()
    print("Download complete!")

if __name__ == "__main__":
    main() 