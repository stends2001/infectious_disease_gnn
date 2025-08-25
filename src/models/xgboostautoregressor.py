import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm import tqdm
from .modelcore import ModelCore

class XGBoostAutoRegressor(ModelCore):
    def __init__(self, dataloader, name=None):
        super().__init__(dataloader, name=name)
        if not self.name:
            self.name = 'XGBoostAutoRegressor'
        self.model_color = '#1F77B4'
        self.model = None

    def prepare_features(self, df):
        # Identify lag and sin/cos features dynamically
        lag_cols = [c for c in df.columns if c.startswith(f'{self.target_column}_lag')]
        sin_cos_cols = [c for c in df.columns if (c.endswith('_sin') or c.endswith('_cos')) and c.startswith(self.temporal_column)]
        
        # Also include id column as categorical
        feature_cols = lag_cols + sin_cos_cols + [self.id_column]
        
        # Prepare X and y
        X = df[feature_cols].copy()
        # Convert id column to categorical codes (numeric)
        X[self.id_column] = X[self.id_column].astype('category').cat.codes
        y = df[self.target_column]
        return X, y

    def forecast(self):
        # Prepare training data
        X_train, y_train = self.prepare_features(self.dataloader.XYt_train)
        X_val, y_val = self.prepare_features(self.dataloader.XYt_val)
        X_test, y_test = self.prepare_features(self.dataloader.XYt_test)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",  # faster
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "verbosity": 1
        }

        evals = [(dtrain, 'train'), (dval, 'val')]

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=20,
            evals=evals,
            verbose_eval=20
        )

        # Predict on test set
        preds = self.model.predict(dtest)
        evaluation_df = self.dataloader.XYt_test.copy()
        evaluation_df['preds'] = preds
        self.evaluation_df = evaluation_df

        return self