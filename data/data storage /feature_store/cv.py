import numpy as np
import pandas as pd

class PurgedWalkForwardCV:
    """
    Purged walk-forward cross-validator for panel data.
    Parameters
    ----------
    n_splits : int
        总共做多少个 walk-forward 步。
    train_period : int
        每步训练用多少"日期单位"（周）。
    test_period : int
        每步测试用多少"日期单位"（周）。
    embargo : int, default 0
        purge 完后，再往后留多少个日期做 embargo，防止信息泄漏。
    """
    def __init__(self, n_splits, train_period, test_period, embargo=0):
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
        self.embargo = embargo

    def split(self, df_dates: pd.Series):
        """df_dates: 与 X 同长的日期 Series（YYYY-MM-DD）"""
        unique_dates = np.array(sorted(df_dates.unique()))
        for k in range(self.n_splits):
            train_start = k * self.test_period
            train_end   = train_start + self.train_period
            test_end    = train_end + self.test_period

            train_days = unique_dates[train_start:train_end]
            test_days  = unique_dates[train_end:test_end]

            # purge & embargo
            purge_start = train_end - self.embargo
            purge_end   = test_end + self.embargo
            purge_days  = unique_dates[purge_start:purge_end]

            train_mask = df_dates.isin(train_days)
            test_mask  = df_dates.isin(test_days)
            purge_mask = df_dates.isin(purge_days)

            yield (np.where(train_mask & ~purge_mask)[0],
                   np.where(test_mask)[0])

    def __repr__(self):
        return (f"PurgedWalkForwardCV("
                f"{self.train_period}-train/"
                f"{self.test_period}-test,"
                f"embargo={self.embargo}, "
                f"n_splits={self.n_splits})") 