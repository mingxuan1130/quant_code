#!/usr/bin/env python
"""
Data caching module for S&P 500 ML factor pipeline.
Handles downloading and loading of price and membership data.
"""

from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
from functools import lru_cache

# Set up paths
HERE = Path(__file__).resolve().parent

# Based on the current script's directory, construct the data and prices directories
DATA_DIR   = HERE / 'polygon loading' / 'polygon_data'
PRICES_DIR = HERE / 'prices'
MEMBERSHIP_FILE = HERE / 'sp500_membership.parquet'

# Ensure directories exist
PRICES_DIR.mkdir(parents=True, exist_ok=True)

# Global price cache
WRDS_CUTOFF = pd.Timestamp('2024-12-31')
_price_cache: Dict[int, pd.DataFrame] = {}

def connect_wrds(username: str | None = None):
    """Connect to WRDS, using provided username or WRDS_USERNAME env var."""
    try:
        import wrds  # type: ignore
        user = username or os.getenv('WRDS_USERNAME')
        if not user:
            print("Warning: WRDS_USERNAME environment variable not set.")
            print("Please set it with: export WRDS_USERNAME=<your wrds uid>")
            print("Trying fallback username...")
            user = 'michael_ml'
        print("Connecting to WRDS database...", end="", flush=True)
        conn = wrds.Connection(wrds_username=user)
        print(" Connected!")
        return conn
    except ImportError:
        print("Error: wrds package not installed. Install with: pip install wrds")
        raise
    except Exception as e:
        print(f"\nError connecting to WRDS: {e}")
        raise

_db = None  # global cache

def db():
    """Get or create WRDS connection."""
    global _db
    if _db is None:
        _db = connect_wrds()
    return _db


def download_sp500_membership() -> pd.DataFrame:
    """Download S&P 500 membership data from WRDS."""
    print("\nDownloading S&P 500 membership data...")
    try:
        qry = """
            WITH latest_names AS (
                SELECT permno, ticker, namedt, nameendt,
                       ROW_NUMBER() OVER (PARTITION BY permno ORDER BY namedt DESC) as rn
                FROM crsp.dsenames
            )
            SELECT DISTINCT 
                a.permno, 
                a.mbrstartdt as start_date, 
                COALESCE(a.mbrenddt, '9999-12-31') as end_date,
                COALESCE(b.ticker, 'UNKNOWN') as ticker
            FROM crsp_a_indexes.dsp500list_v2 a
            LEFT JOIN latest_names b
                ON a.permno = b.permno
                AND b.rn = 1
            ORDER BY a.permno, a.mbrstartdt
        """
        df = db().raw_sql(qry, date_cols=['start_date', 'end_date'])
        print(f"Downloaded {len(df)} membership records")
        return df
    except Exception as e:
        print(f"Error downloading membership data: {e}")
        raise

def download_price_data(members: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """Download price data for S&P 500 members and save as partitioned parquet."""
    print("\nDownloading price data...")
    
    # Get unique permnos
    permnos = members['permno'].unique()
    print(f"Found {len(permnos)} unique permnos")
    
    years = range(start_date.year, (end_date + pd.DateOffset(days=10)).year + 1)
    batch_size = 2000  # Increased to 2000
    
    # Define price partitioning schema with year 
    price_schema = pa.schema([
        ('permno', pa.int64()),
        ('date', pa.date32()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('year', pa.int32())
    ])

    for yr in tqdm(years, desc="Year Progress", mininterval=5):
        y0 = f"{yr}-01-01"
        y1 = f"{yr}-12-31"
        total_batches = (len(permnos) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc=f"Batch Progress ({yr})", leave=False, mininterval=5):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(permnos))
            permnos_batch = permnos[start_idx:end_idx]

            print(f"\nDownloading {len(permnos_batch)} permnos for year {yr} (batch {batch_idx + 1}/{total_batches})")

            # Use parameterized query
            qry = """
                SELECT permno,
                    date,
                    ABS(prc / cfacpr)  AS close,
                    ABS(vol / cfacshr) AS volume
                FROM crsp.dsf
                WHERE permno = ANY(%(permnos)s)
                AND date BETWEEN %(start_date)s AND %(end_date)s
            """
            
            try:
                df = db().raw_sql(
                    qry,
                    params={
                        'permnos': permnos_batch.tolist(),
                        'start_date': y0,
                        'end_date': y1
                    },
                    date_cols=['date']
                )
                
                if df.empty:
                    print(f"No data found for batch {batch_idx + 1} in year {yr}")
                    continue

                df['year'] = yr
                table = pa.Table.from_pandas(df, schema=price_schema)

                # Batch write to Parquet
                ds.write_dataset(
                    table,
                    PRICES_DIR,
                    format="parquet",
                    partitioning=['year'],
                    existing_data_behavior='overwrite_or_ignore',
                    max_rows_per_file=128_000,
                    max_rows_per_group=128_000
                )
                
                print(f"Successfully processed batch {batch_idx + 1} for year {yr}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1} for year {yr}: {e}")
                continue

@lru_cache(maxsize=None)
def load_full_price(permno: int) -> pd.DataFrame:
    """Load full price history for a stock into memory cache."""
    try:
        # Get all parquet files for this permno
        permno_dir = PRICES_DIR / str(permno)
        if not permno_dir.exists():
            return pd.DataFrame(columns=['date', 'close', 'volume'])
            
        # Read all parquet files for this permno
        dfs = []
        for year_dir in permno_dir.iterdir():
            if not year_dir.is_dir():
                continue
            parquet_file = year_dir / 'part-0.parquet'
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
                
        if not dfs:
            return pd.DataFrame(columns=['date', 'close', 'volume'])
            
        # Combine all data
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure date column is datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # We're already using the volume column from the SQL query
        # If raw_vol exists, drop it as we're using the adjusted volume
        if 'raw_vol' in df.columns:
            df = df.drop(columns=['raw_vol'])
        
        return df[['date', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error loading full price data for permno {permno}: {e}")
        return pd.DataFrame(columns=['date', 'close', 'volume'])

def load_price_fast(permno: int, lookback_start: pd.Timestamp, future_date: pd.Timestamp) -> pd.DataFrame:
    """Load price data from memory cache with date filtering."""
    # Get full price history from cache
    df = load_full_price(permno)
    
    if df.empty:
        return df
        
    # Filter by date range
    mask = (df['date'] >= lookback_start) & (df['date'] <= future_date)
    return df[mask].copy()

def ensure_local_cache(start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """Ensure all required data is available in local cache."""
    print("\nChecking local data cache...")
    
    # Check membership data
    if not MEMBERSHIP_FILE.exists():
        print("Membership data not found, downloading...")
        membership = download_sp500_membership()
        membership.to_parquet(MEMBERSHIP_FILE)
        print(f"Saved membership data to {MEMBERSHIP_FILE}")
    else:
        print("Membership data found in cache")
        membership = pd.read_parquet(MEMBERSHIP_FILE)
    
    # Check price data
    print("\nChecking price data...")
    download_needed = False
    
    # Get required years
    future_date = end_date + pd.DateOffset(days=10)
    required_years = range(start_date.year, future_date.year + 1)
    
    # Check if we have data for all required years
    for permno in tqdm(membership['permno'].unique(), desc="Checking price data"):
        for year in required_years:
            year_dir = PRICES_DIR / f"year={year}"
            if not year_dir.exists() or not list(year_dir.glob("*.parquet")):
                download_needed = True
                break
        if download_needed:
            break
    
    if download_needed:
        print("\nDownloading missing price data...")
        download_price_data(membership, start_date, end_date)
    else:
        print("All required price data found in cache")
    
    # Preload all price data into memory
    print("\nPreloading price data into memory...")
    for permno in tqdm(membership['permno'].unique(), desc="Preloading prices"):
        load_full_price(permno)
    
    # Print cache size
    total_size = sum(f.stat().st_size for f in PRICES_DIR.rglob("*.parquet"))
    print(f"\nTotal cache size: {total_size / (1024*1024):.1f} MB")

def get_membership_df(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Get S&P 500 membership data for the specified date range."""
    if not MEMBERSHIP_FILE.exists():
        ensure_local_cache(start_date, end_date)
    
    membership = pd.read_parquet(MEMBERSHIP_FILE)
    return membership[
        (membership['start_date'] <= end_date) &
        (membership['end_date'] >= start_date)
    ]


