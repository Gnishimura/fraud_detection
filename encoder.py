import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

class Encoder(TransformerMixin):
    def __init__(self, columns=None, thresh=0, sep='/', drop=True):
        self.columns = columns
        self.thresh = thresh
        self.encoding = None
        self.sep = sep
        self.drop = drop

    def fit(self, X, y=None):
        """Select categorical features to dummify, with the option of setting a threshold 
        on which values to include as new features"""

        #Specify columns to drop. If None, then drop all non-categorical columns
        c=self.columns
        if c is None: c=self.drop_numeric_columns(X).columns

        #Set threshold based on either a percentage of total or a raw number
        if isinstance(self.thresh, float):
            thresh = int(len(X) * self.thresh)
        else:
            thresh = self.thresh

        self.encoding = dict()
        for col in c:
            e = self._get_column_names(X[col], thresh)  #Find the values for creating dummy columns
            if len(e) > 0:
                self.encoding[col] = e
        return self

    def transform(self, X, y=None, inplace=False):
        """Create new dummy columns"""
        if not inplace: 
            X = X.copy()
        for col in self.encoding:
            self._apply_encoding(X, col)
        return X

    def _get_column_names(self, s, thresh):
        """Returns a list of the values in a given series that meet the threshold criteria"""
        output = []
        for val, n in s.value_counts().iteritems():
            if n >= thresh:
                output.append(val)
        if len(output) == len(s.value_counts()):
            output.pop()
        return output

    def _apply_encoding(self, X, col):
        """Create new columns according to values in self.encoding"""
        s = X[col]
        for val in self.encoding[col]:
            X[f'{col}_{val}'] = (s.astype(type(val)) == val).astype(int)
        if self.drop:
            X.drop(col, axis=1, inplace=True)

    @staticmethod
    def drop_numeric_columns(X):
        numeric_cols = [c for c,b in X.apply(Encoder.can_cast).items() if b]
        return X.drop(columns=numeric_cols)

    @staticmethod
    def can_cast(x, dtypes=(np.int64, np.float64, np.datetime64)):
        for dtype in dtypes:
            try:
                dtype(x)
                return True
            except (ValueError, TypeError):
                pass
        return False