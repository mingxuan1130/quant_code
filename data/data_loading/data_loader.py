from pathlib import Path
from typing import Tuple
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

class DataLoader:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _files_in_range(self, d0: pd.Timestamp, d1: pd.Timestamp):
        for f in self.root.glob("*.parquet"):
            try:
                file_date = pd.to_datetime("-".join(f.stem.split("-")[:3]))
                if d0 <= file_date <= d1:
                    yield f
            except Exception:
                continue

    def load(self, start: str, end: str, *, fillna=0, dropna_cols=None,
             show_progress=False) -> pd.DataFrame:
        d0, d1 = pd.to_datetime(start), pd.to_datetime(end)
        files = list(self._files_in_range(d0, d1))
        if not files:
            raise FileNotFoundError("No files in range")

        tables = []
        it = tqdm(files, desc="Loading", disable=not show_progress)
        for f in it:
            tables.append(pq.read_table(f))

        df = pa.concat_tables(tables, promote=True).to_pandas()
        feature_cols = [c for c in df.columns
                        if c not in {'permno', 'factor_date',
                                     'weekly_return', 'weekly_rank'}]

        df[feature_cols] = df[feature_cols].fillna(fillna)
        if dropna_cols:
            df = df.dropna(subset=dropna_cols)
        return df 