import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedWalkForwardSplit(BaseCrossValidator):
    """
    Splits time-series indices into non-overlapping train/val folds
    with look-back, step, and embargo.
    """
    def __init__(self, n_splits, train_period, test_period, embargo=0):
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator"""
        return self.n_splits

    def split(self, dates: np.ndarray):
        # Assume dates are sorted in ascending order
        unique_dates = np.unique(dates)
        start = 0
        for _ in range(self.n_splits):
            train_start = start
            train_end = train_start + self.train_period
            test_start = train_end + self.embargo
            test_end = test_start + self.test_period

            if test_end > len(unique_dates):
                break

            train_mask = np.isin(dates,
                                 unique_dates[train_start:train_end])
            val_mask = np.isin(dates,
                               unique_dates[test_start:test_end])

            yield np.where(train_mask)[0], np.where(val_mask)[0]
            start += self.train_period  # Step forward in walk-forward 