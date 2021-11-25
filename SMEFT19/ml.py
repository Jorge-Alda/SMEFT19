'''
=================
ml
=================

This module contains the functions needed to train a Machine Learning-based
Montecarlo scan, and to assess its performance.
'''

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import shap
from SMEFT19.SMEFTglob import likelihood_global
from SMEFT19.scenarios import rotBII
from parscanning.mlscan import MLScan


def lh(x):
    return likelihood_global(x, rotBII)

def train(fMC, fval, fmodel, bf):
    r'''
Trains the Machine Learning algorithm with the previously computed Metropolis points

:Arguments:

    - fMC\: Path to the file containing the Montecarlo pre-computed points.
    - fval\: Path to the file where the validation points will be saved.
    - fmodel\: Path to the file where the XGBoost model will be saved.
    - bf\: Best fit point.

:Returns:

    - The Machine Learning scan module, already trained and ready to be used
    '''
    df = pd.read_csv(fMC, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    features = ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = df.logL
    model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, n_jobs=4, learning_rate=0.05)
    ML = MLScan(lh, list(df.min()[:5]), list(df.max()[:5]), 1000, bf)
    ML.init_ML(model)
    ML.train_pred(X, y, mean_absolute_error)
    model.save_model(fmodel)
    ML.save_validation(fval)
    return ML

def regr(ML, vpoints):
    r'''
Plots the predicted likelihod vs the actual likelihood and computes their regression coefficient

:Arguments:

    - ML:\ The Machine Learning scan module.
    - vpoints\: Path to the file containing the points in the validation dataset.

:Returns:

    - A tuple containing the Perason r coefficient and the p-value of the regression
    '''

    df = pd.read_csv(vpoints, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    features = ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = 2*df.logL
    pred = 2*np.array(list(map(ML.guess_lh, X.values)))

    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=0.7)
    plt.plot([20, 50], [20, 50], c='black', lw=1)
    plt.xlim([20, 50])
    plt.ylim([20, 50])
    plt.xlabel(r'Actual $\Delta \chi^2_\mathrm{SM}$')
    plt.ylabel(r'Predicted $\Delta \chi^2_\mathrm{SM}$')
    plt.tight_layout(pad=0.5)
    return pearsonr(y, pred)

def hist(ML, vpoints):
    r'''
Plots an histogram for the predicted and actual likelihoods,
and compares them to the chi-square distribution

:Arguments:

    - ML:\ The Machine Learning scan module.
    - vpoints\: Path to the file containing the points in the validation dataset.
    '''

    df = pd.read_csv(vpoints, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    features = ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = 2*df.logL
    pred = 2*np.array(list(map(ML.guess_lh, X.values)))

    plt.figure()
    plt.hist(2*max(pred)-2*np.array(pred), range=(0, 25), bins=50,
             density=True, alpha=0.5, label='Predicted histogram')
    plt.hist(2*max(y)-2*np.array(y), range=(0, 25), bins=50,
             density=True, alpha=0.5, label='Actual histogram')
    plt.plot(np.linspace(0, 25, 51), chi2(5).pdf(np.linspace(0, 25, 51)),
             lw=1.5, color='red', label=r'$\chi^2$ distribution')
    plt.xlabel(r'$\chi^2_\mathrm{bf} - \chi^2$')
    plt.ylabel('Normalized frequency')
    plt.legend()
    plt.tight_layout(pad=0.5)

def load_model(fmodel, vpoints, bf):
    r'''
Loads a XGBoost model previously saved

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - bf\: Best fit point.

:Returns:

    - Machine Learning scan.
    '''

    df = pd.read_csv(vpoints, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    model = XGBRegressor()
    model.load_model(fmodel)
    ML = MLScan(lh, list(df.min()[:5]), list(df.max()[:5]), 1000, bf)
    ML.init_ML(model)
    return ML

def SHAP_bf(fmodel, bf):
    r'''
Computes the SHAP values of the best fit point

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - bf\: Best fit point.
    '''

    model = XGBRegressor()
    model.load_model(fmodel)
    explainer = shap.TreeExplainer(model)
    print(f'Base value: {float(explainer.expected_value)}')
    bfs = pd.Series(bf)
    print(f'SHAP values: {explainer.shap_values(bfs)}')
    total = float(explainer.expected_value)+np.sum(explainer.shap_values(bfs))
    print(f'Total prediction: {total}')

def SHAP_summary(fmodel, points):
    r'''
Creates a summary plot of the average SHAP values on a dataset.

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - points\: Pandas Dataframe containing the dataset.
    '''

    model = XGBRegressor()
    model.load_model(fmodel)
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv(points, sep='\t', names=['$C$', '$\\alpha^\\ell$',
                                              '$\\beta^\\ell$', '$\\alpha^q$',
                                              '$\\beta^q$', 'logL'])
    features = ['$C$', '$\\alpha^\\ell$', '$\\beta^\\ell$', '$\\alpha^q$', '$\\beta^q$']
    X = df[features]
    sv = explainer.shap_values(X)
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout(pad=0.5)

def SHAP_param(fmodel, points, param):
    r'''
Creates an scatter plot displaying how the SHAP values change
as functions of each parameter of the fit.

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - points\: Pandas Dataframe containing the dataset.
    - param\: Fit parameter. 0 = C, 1 = al, 2 = bl, 3 = aq, 4 = bq.
    '''

    model = XGBRegressor()
    model.load_model(fmodel)
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv(points, sep='\t', names=['$C$', '$\\alpha^\\ell$',
                                              '$\\beta^\\ell$', '$\\alpha^q$',
                                              '$\\beta^q$', 'logL'])
    features = ['$C$', '$\\alpha^\\ell$', '$\\beta^\\ell$', '$\\alpha^q$', '$\\beta^q$']
    X = df[features]
    sv = explainer.shap_values(X)
    shap.dependence_plot(param, sv, X, show=False, interaction_index=None, dot_size=5)
    plt.ylim([-6, 3])
    plt.tight_layout(pad=0.5)
