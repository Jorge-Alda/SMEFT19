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
from parscanning.mlscan import MLScan
from SMEFT19.SMEFTglob import likelihood_global
from SMEFT19.scenarios import rotBII

plt.rcParams.update({'pgf.texsystem':'pdflatex'})


def lh(x):
    '''
Pickle-able function for the likelihood in scenario BII.
    '''
    return likelihood_global(x, rotBII)

def train(dataset, fval, fmodel, bf, headers=None):
    r'''
Trains the Machine Learning algorithm with the previously computed Metropolis points

:Arguments:

    - dataset\: Path to the file or list of files containing the Montecarlo pre-computed points.
    - fval\: Path to the file where the validation points will be saved.
    - fmodel\: Path to the file where the XGBoost model will be saved.
    - bf\: Best fit point.
    - headers\: Header lines in the dataset files.
      None if there is no header, 0 if the first line contains the header.
      Admits list if using several dataset files.

:Returns:

    - The Machine Learning scan module, already trained and ready to be used
    '''
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        if headers is None:
            headers = [None,]*len(dataset)
        if isinstance(headers, int):
            headers = [headers,]*len(dataset)
        dfs = [pd.read_csv(f, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'],
                           header=headers[i]) for i, f in enumerate(dataset)]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(dataset, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'],
                         header=headers)
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

def regr(ML, vpoints, fout):
    r'''
Plots the predicted likelihod vs the actual likelihood and computes their regression coefficient

:Arguments:

    - ML:\ The Machine Learning scan module.
    - vpoints\: Path to the file containing the points in the validation dataset.
    - fout\: Path to the output regression plot (pdf only).

:Returns:

    - A tuple containing the Perason r coefficient and the p-value of the regression
    '''

    df = pd.read_csv(vpoints, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    features = ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = 2*df.logL
    pred = 2*ML.model.predict(X)

    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=0.7)
    plt.plot([20, 60], [20, 60], c='black', lw=1)
    plt.xlim([20, 60])
    plt.ylim([20, 60])
    plt.xlabel(r'Actual $\Delta \chi^2_\mathrm{SM}$', fontsize=16)
    plt.ylabel(r'Predicted $\Delta \chi^2_\mathrm{SM}$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(pad=0.5)
    plt.savefig(fout + '.pdf')
    return pearsonr(y, pred)

def hist(ML, vpoints, fout):
    r'''
Plots an histogram for the predicted and actual likelihoods,
and compares them to the chi-square distribution

:Arguments:

    - ML:\ The Machine Learning scan module.
    - vpoints\: Path to the file containing the points in the validation dataset.
    - fout\: Path to save the histogram.
    '''

    df = pd.read_csv(vpoints, sep='\t', names=['C', 'al', 'bl', 'aq', 'bq', 'logL'])
    df = df.loc[df['logL'] > 10]
    features = ['C', 'al', 'bl', 'aq', 'bq']
    X = df[features]
    y = 2*df.logL
    pred = 2*ML.model.predict(X)

    plt.figure()
    plt.hist(max(pred)-np.array(pred), range=(0, 25), bins=50,
             density=True, alpha=0.5, label='Predicted histogram')
    plt.hist(max(y)-np.array(y), range=(0, 25), bins=50,
             density=True, alpha=0.5, label='Actual histogram')
    plt.plot(np.linspace(0, 25, 51), chi2(5).pdf(np.linspace(0, 25, 51)),
             lw=1.5, color='red', label=r'$\chi^2$ distribution')
    plt.xlabel(r'$\chi^2_\mathrm{bf} - \chi^2$', fontsize=18)
    plt.ylabel('Normalized frequency', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout(pad=0.5)
    plt.savefig(fout+'.pdf')
    plt.savefig(fout+'.pgf')

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

def SHAP_summary(fmodel, points, fout, header=None):
    r'''
Creates a summary plot of the average SHAP values on a dataset.

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - points\: Pandas Dataframe containing the dataset.
    - fout\: Path to save the plot (pdf only).
    - header\: If the data file contains headers in the first row, 0.
    '''

    model = XGBRegressor()
    model.load_model(fmodel)
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv(points, sep='\t', names=['$C$', '$\\alpha^\\ell$',
                                              '$\\beta^\\ell$', '$\\alpha^q$',
                                              '$\\beta^q$', 'logL'], header=header)
    features = ['$C$', '$\\alpha^\\ell$', '$\\beta^\\ell$', '$\\alpha^q$', '$\\beta^q$']
    X = df[features]
    sv = explainer.shap_values(X)
    shap.summary_plot(sv, X, show=False)
    plt.xticks(fontsize=16)
    plt.tight_layout(pad=0.5)
    plt.savefig(fout+'.pdf')

def SHAP_param(fmodel, points, param, header=None):
    r'''
Creates an scatter plot displaying how the SHAP values change
as functions of each parameter of the fit.

:Arguments:

    - fmodel\: Path to the file where the model was saved.
    - points\: Pandas Dataframe containing the dataset.
    - param\: Fit parameter. 0 = C, 1 = al, 2 = bl, 3 = aq, 4 = bq.
    - header\: If the data file contains headers in the first row, 0.
    '''

    model = XGBRegressor()
    model.load_model(fmodel)
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv(points, sep='\t', names=['$C$', '$\\alpha^\\ell$',
                                              '$\\beta^\\ell$', '$\\alpha^q$',
                                              '$\\beta^q$', 'logL'], header=header)
    features = ['$C$', '$\\alpha^\\ell$', '$\\beta^\\ell$', '$\\alpha^q$', '$\\beta^q$']
    X = df[features]
    sv = explainer.shap_values(X)
    shap.dependence_plot(param, sv, X, show=False, interaction_index=None, dot_size=5)
    plt.ylim([-6, 3])
    plt.tight_layout(pad=0.5)
