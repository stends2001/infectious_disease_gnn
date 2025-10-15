from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Literal
import xgboost as xgb 
from ..base.basemodel import BaseModel

from ...dataloading.gnndataloader import GNNDataLoader


class SpatioTemporalXGBModel(BaseModel):

    """
    
    Examples:
    --------
    >>> dataloader_identity_graph = GNNDataLoader(disease_name, get_data_env(), nuts_level=nuts_level, min_date=min_date,max_date=max_date, include_population=False, horizon_size = horizon_size, horizon_leadtime = horizon_leadtime, sequence_length=sequence_length, split_berlin=split_berlin)
    >>> dataloader_identity_graph.add_time_features()
    >>> dataloader_identity_graph.log_transform_target()
    >>> dataloader_identity_graph.set_splits(split_trainval, split_valtest)
    >>> dataloader_identity_graph.normalize()
    >>> dataloader_identity_graph.add_lagged_features(lags = lags)
    >>> dataloader_identity_graph.finalize()
    >>> dataloader_identity_graph.retrieve_graph('identity_selfmax').construct_dataloaders()

    >>> xgbmodel_id = SpatioTemporalXGBModel(dataloader_identity_graph)
    >>> xgbmodel_id.set_global_hparams(lr = 0.01, n_epochs=500)
    >>> xgbmodel_id.set_model_hparams(n_estimators = 1000, max_depth = 8)
    >>> xgbmodel_id.train(verbose = 2)
    >>> xgbmodel_id.forecast('train')
    >>> xgbmodel_id.show_forecasts('train',26, timeframe = ['2009-01-01','2010-01-01'])    
    """


    def __init__(self, 
                 dataloader     : 'GNNDataLoader',
                 name           : Optional[str] = None):
        super().__init__(dataloader, name)
        
        if not self.name:
            self.name = f'SpatioTemporalXGB'
        
        # dataloader metadata
        self.gnn_dataloader     = dataloader
        self.sequence_length    = dataloader.sequence_length
        self.horizon_size       = dataloader.horizon_size
        self.models             = {}
        
        # Store graph structure if provided
        self.edge_index         = dataloader.edge_index
        self.neighbor_dict: Optional[Dict[int, list]] = None
        
        self._build_neighbor_dict()
        
        # XGBoost models (one per horizon if multi-horizon)
        
        # Data containers
        self.long_df_train = self._unpack_dataloader(dataloader.dataloader_train)
        self.long_df_val   = self._unpack_dataloader(dataloader.dataloader_val)
        self.long_df_test  = self._unpack_dataloader(dataloader.dataloader_test)    

        target_col = f"target_h{0}"
        feature_cols = [col for col in self.long_df_train.columns
                        if col not in ['t_idx', 'node','timestamp'] and not col.startswith('target')]
        # Prepare X and y as NumPy arrays (for sklearn API)
        self.X_train = self.long_df_train[feature_cols].values
        self.y_train = self.long_df_train[target_col].values

        self.X_val = self.long_df_val[feature_cols].values
        self.y_val = self.long_df_val[target_col].values

        self.X_test = self.long_df_test[feature_cols].values
        self.y_test = self.long_df_test[target_col].values

        self._map_timesteps()    
        
        self._state = {
            'model_initialized' : False,
            'trained'           : False,
            'forecasted'        : False,
        }
        
        self.evaluation_datasets    = {}
        self.train_losses           = []
        self.val_losses             = []
       
    def _check_state(self, required_states: list) -> None:
        """Validate that required setup steps have been completed."""
        missing = [s for s in required_states if not self._state.get(s, False)]
        if missing:
            raise ValueError(
                f"Missing required setup steps: {', '.join(missing)}. "
                f"Call the corresponding methods first."
            )
    
    def _build_neighbor_dict(self, taskbar = False):
        """
        Build dictionary mapping each node to its neighbors
        
        """
        
        neighbor_dict   = {}
        edge_list       = self.edge_index.t().tolist()
        
        iterator = tqdm(edge_list, desc='walking edge indices') if taskbar else edge_list

        for src, dst in iterator:
            if src not in neighbor_dict:
                neighbor_dict[src] = set()
            neighbor_dict[src].add(dst)
        
        
        self.neighbor_dict = {k: sorted(list(v)) for k, v in neighbor_dict.items()}
    
    def _unpack_dataloader(self, loader):
        """
        Extract data from GNN dataloader format into flat dataframe (vectorized),
        and augment with neighbor-aggregated features.
        """

        import numpy as np
        import pandas as pd
        from tqdm import tqdm

        num_nodes, num_features, sequence_length = loader[0].x.shape
        num_snapshots = len(loader)

        features = np.empty((num_snapshots, num_nodes, num_features, sequence_length), dtype=np.float32)
        targets = np.empty((num_snapshots, num_nodes, self.horizon_size), dtype=np.float32)

        for t_idx, snapshot in enumerate(loader):
            features[t_idx] = snapshot.x.numpy()
            targets[t_idx] = snapshot.y.numpy()

        # Reverse time axis to match naming convention (t, t-1, ...)
        features = features[..., ::-1]

        # Create feature column names
        feature_cols = []
        for feat_name in self.dataloader.feature_columns:
            for t in range(sequence_length):
                suffix = "_t" if t == 0 else f"_t-{t}"
                feature_cols.append(f"{feat_name}{suffix}")

        # Flatten arrays
        features_flat = features.reshape(num_snapshots * num_nodes, -1)
        targets_flat = targets.reshape(num_snapshots * num_nodes, self.horizon_size)

        # Create base dataframe
        t_idx_arr = np.repeat(np.arange(num_snapshots), num_nodes)
        node_arr = np.tile(np.arange(num_nodes), num_snapshots)

        df = pd.DataFrame(features_flat, columns=feature_cols)
        df['t_idx'] = t_idx_arr
        df['node'] = node_arr

        # Add target columns
        for h in range(self.horizon_size):
            df[f"target_h{h}"] = targets_flat[:, h]

        # === Vectorized Neighbor Aggregation ===
        # 1. Create neighbor mapping table — only valid neighbors
        rows = []
        for node, neighbors in self.neighbor_dict.items():
            for nbr in neighbors:
                if nbr != node:  # Optional: skip self if needed
                    rows.append({'node': node, 'neighbor': nbr})
        neighbor_map = pd.DataFrame(rows)  # [node, neighbor]

        # ⛔️ If the above DataFrame is empty (e.g. only self-loops), stop early
        if neighbor_map.empty:
            print("⚠️ No neighbors found — skipping aggregation.")
            return df

        # 2. Get node-level features only (no targets, no duplicate columns)
        base_feats = df.drop(columns=[c for c in df.columns if c.startswith('target')])

        # 3. Merge to assign neighbors
        df_neighbors = neighbor_map.merge(base_feats, left_on='neighbor', right_on='node')
        df_neighbors = df_neighbors.rename(columns={'node_x': 'node', 't_idx': 't_idx', 'node_y': 'neighbor'})

        # 4. Aggregate neighbor features
        agg_feature_cols = [col for col in base_feats.columns if col not in ['node', 't_idx']]

        neighbor_agg = (
            df_neighbors
            .groupby(['node', 't_idx'])[agg_feature_cols]
            .mean()
            .add_suffix('_neighbor')
            .reset_index()
        )

        # 5. Merge neighbor features back into original
        df_final = df.merge(neighbor_agg, on=['node', 't_idx'], how='left')

        return df_final
    
    def _map_timesteps(self):
        max_t = max(
            self.long_df_train["t_idx"].max(),
            self.long_df_val["t_idx"].max(),
            self.long_df_test["t_idx"].max()
        )

        # Create a mapping from t_idx to actual timestamp
        t_idx_to_ts = pd.date_range(start=self.dataloader.min_date, periods=max_t + 1, freq='W-MON')
        t_idx_to_ts = pd.Series(t_idx_to_ts, name="timestamp").reset_index().rename(columns={"index": "t_idx"})

        # Merge into each dataset
        for attr in ["long_df_train", "long_df_val", "long_df_test"]:
            df = getattr(self, attr)
            df = df.merge(t_idx_to_ts, on="t_idx", how="left")
            setattr(self, attr, df)        

    def forecast(self, 
                dataset: Literal['train', 'val', 'test'] = 'test'):
        """
        Generate forecasts for the specified dataset using trained XGBoost models.
        
        Parameters:
        -----------
        dataset : str
            Which dataset to forecast on ('train', 'val', 'test')

        Returns:
        --------
        self : SpatioTemporalXGBModel
        """
        horizon = 0
        model = self.models[f'horizon_{horizon}']
        
        # Select the appropriate dataframe
        X_map = {
            'train' : self.X_train,
            'val'   : self.X_val,
            'test'  : self.X_test
        }

        y_map = {
            'train' : self.y_train,
            'val'   : self.y_val,
            'test'  : self.y_test
        }

        eval_df_map = {
            'train' :  self.long_df_train[['timestamp','node','target_h0']],
            'val'   :  self.long_df_val[['timestamp'  ,'node','target_h0']],
            'test'  :  self.long_df_test[['timestamp' ,'node','target_h0']]                       
        }

        # Generate predictions per horizon
        preds = model.predict(X_map[dataset])

        horizon_prediction_dict = {}

        evaluation_df           = eval_df_map[dataset].rename(columns = {'target_h0': 'incidence'})
        evaluation_df['pred']   = preds

        horizon_prediction_dict['transformed']      = {f'horizon_{horizon}': evaluation_df}
        horizon_prediction_dict['nontransformed']   = {f'horizon_{horizon}': self._denorm_predictions(evaluation_df)}    
        self.evaluation_datasets[dataset] = horizon_prediction_dict    
        self._state['forecasted'] = True

    def set_model_hparams(self,
                        max_depth: int = 6,
                        subsample: float = 0.8,
                        colsample_bytree: float = 0.8,
                        reg_alpha: float = 0.0,
                        reg_lambda: float = 1.0,
                        **kwargs):
        """
        Set XGBoost-specific hyperparameters (model-level).
        Excludes training-related hyperparams like learning_rate, n_estimators, etc.
        """
        self._check_state(['global_hparams_set'])

        # Protect global hparams from being overwritten here
        forbidden_keys = {'learning_rate', 'n_estimators', 'early_stopping_rounds'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}

        self.model_hparams = {
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'tree_method': 'hist',
            'random_state': 42,
            **filtered_kwargs
        }

        self.config_info['model_hparams'] = self.model_hparams
        self._state['model_initialized'] = True

        # Initialize one model per horizon (with training params added later)
        self.models = {}
        for hh in range(self.horizon_size):
            self.models[f'horizon_{hh}'] = xgb.XGBRegressor(**self.model_hparams)

        print(f"✓ Initialized {self.horizon_size} XGBoost models")

    def set_global_hparams(self, 
                        lr: float = 0.001,
                        n_epochs: int = 5,
                        patience: int = 15,
                        min_delta: float = 1e-4):
        """
        Set global training hyperparameters (used across models).
        """
        

        self.global_hparams = {
            'lr': lr,
            'n_epochs': n_epochs,
            'patience': patience,
            'min_delta': min_delta
        }

        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.min_delta = min_delta

        self.config_info['global_hparams'] = self.global_hparams
        self._state['global_hparams_set'] = True

        # import xgboost as xgb

    def train(self, horizon=0, verbose=2):
        """
        Train XGBoost model for a given prediction horizon using global training hyperparameters.
        """
        self._check_state(['model_initialized', 'global_hparams_set'])

        model = self.models[f'horizon_{horizon}']

        # Inject global training hparams into the model
        model.set_params(
            learning_rate=self.lr,
            n_estimators=self.n_epochs,
            early_stopping_rounds=self.patience
        )

        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=verbose > 1
        )

        # Store losses
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        train_mse = np.mean((train_pred - self.y_train) ** 2)
        val_mse = np.mean((val_pred - self.y_val) ** 2)

        if verbose > 0:
            print(f"✓ Horizon {horizon} - Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
    
    def __repr__(self) -> str:
        """String representation for SpatioTemporalXGBModel"""
        
        # Calculate max length for alignment
        max_len = max(len(state) for state in self._state.keys())
        if self.config_info.get('model_hparams'):
            max_len = max(max_len, max(len(k) for k in self.config_info['model_hparams'].keys()))
        if self.config_info.get('global_hparams'):
            max_len = max(max_len, max(len(k) for k in self.config_info['global_hparams'].keys()))
        max_len = max(max_len, len('model name'), len('model class'), len('sequence length'), 
                    len('horizon size'), len('num neighbors'))
        
        # Status lines
        status_lines = [
            f'    {state:<{max_len}} : {"✓" if value else "✗"}'
            for state, value in self._state.items()
        ]
        
        # Model hyperparameter lines
        model_hparam_lines = []
        if self.config_info.get('model_hparams'):
            model_hparam_lines = [
                f'    {hparam:<{max_len}} : {value}' 
                for hparam, value in self.config_info['model_hparams'].items()
            ]
        
        # Global hyperparameter lines
        global_hparam_lines = []
        if self.config_info.get('global_hparams'):
            global_hparam_lines = [
                f'    {hparam:<{max_len}} : {value}' 
                for hparam, value in self.config_info['global_hparams'].items()
            ]
        
        # Get neighbor statistics
        num_neighbors = 'N/A'
        if self.neighbor_dict:
            neighbor_counts = [len(neighbors) for neighbors in self.neighbor_dict.values()]
            if neighbor_counts:
                avg_neighbors = sum(neighbor_counts) / len(neighbor_counts)
                num_neighbors = f"{avg_neighbors:.1f} (avg)"
        
        # Build the full representation
        lines = [
            '<SpatioTemporalXGBModel(',
            f"    {'model name':<{max_len}} : {self.name}",
            f"    {'model class':<{max_len}} : {self.model_class}",
            f"    {'sequence length':<{max_len}} : {self.sequence_length}",
            f"    {'horizon size':<{max_len}} : {self.horizon_size}",
            f"    {'num neighbors':<{max_len}} : {num_neighbors}",
            '',
            '    ----------- STATUS --------------',
            *status_lines,
            '',
            '    ----------- FORECASTS -----------',
            f"    {'forecasted':<{max_len}} : {list(self.evaluation_datasets.keys())}",
        ]
        
        # Add model hparams if they exist
        if model_hparam_lines:
            lines.extend([
                '',
                '    ----------- MODEL HPARAMS -------',
                *model_hparam_lines,
            ])
        
        # Add global hparams if they exist
        if global_hparam_lines:
            lines.extend([
                '',
                '    ----------- GLOBAL HPARAMS ------',
                *global_hparam_lines,
            ])
        
        # Add data statistics
        lines.extend([
            '',
            '    ----------- DATA STATS ----------',
            f"    {'train samples':<{max_len}} : {len(self.X_train) if hasattr(self, 'X_train') else 'N/A'}",
            f"    {'val samples':<{max_len}} : {len(self.X_val) if hasattr(self, 'X_val') else 'N/A'}",
            f"    {'test samples':<{max_len}} : {len(self.X_test) if hasattr(self, 'X_test') else 'N/A'}",
            f"    {'num features':<{max_len}} : {self.X_train.shape[1] if hasattr(self, 'X_train') else 'N/A'}",
        ])
        
        lines.append(')>')
        
        return '\n'.join(lines)    