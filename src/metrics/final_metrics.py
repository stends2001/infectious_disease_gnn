from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def return_metrics(model):
    prediction_df = model.preds_denorm
    
    AMSE = _return_AMSE(prediction_df)
    AMAE  = _return_AMAE(prediction_df)

    return AMSE, AMAE

def _return_AMSE(df):
    nodes = df['node'].unique()
    rmses = []
    for node in nodes:
        node_df = df[df['node'] == node]
        rmse = mean_squared_error(node_df['preds'], node_df['incidence'])
        rmses.append(rmse)
    return np.mean(rmses)

def _return_AMAE(df):
    nodes = df['node'].unique()
    maes = []
    for node in nodes:
        node_df = df[df['node'] == node]
        mae = mean_absolute_error(node_df['preds'], node_df['incidence'])
        maes.append(mae)
    return np.mean(maes)

