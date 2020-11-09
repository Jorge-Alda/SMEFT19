from parscanning.mlscan import MLScan
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from SMEFT19.SMEFTglob import likelihood_global
from SMEFT19.scenarios import rotBII

bf = [-0.11995206352339435, -0.07715992292268066, -1.207419259815296e-06, -0.07618023346979363, 0.8027006412644478]

def lh(x):
    return likelihood_global(x, rotBII)

def train():
    df = pd.read_csv('../samples/metropoints.dat', sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL']>10]
    features =  ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = df.logL
    model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, n_jobs=4, learning_rate=0.05)
    ML = MLScan(lh, list(df.min()[:5]), list(df.max()[:5]), 1000, bf)
    ML.init_ML(model)
    ML.train_pred(X, y, mean_absolute_error)

    model.save_model('xgb_lh.json')

def generate_points(N):
    model = XGBRegressor()
    model.load_model('xgb_lh.json')
    ML = MLScan(lh, list(df.min()[:5]), list(df.max()[:5]), N, bf)
    ML.init_ML(model)
    ML.run_mp(4)
    ML.write('../samples/mlsample.dat')
    
