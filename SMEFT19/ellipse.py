'''
=========
ellipse
=========
Assuming that the likelihood of the fit follows a gaussian distribution (Central Limit Theorem),
and therefore the log-likelihood is characterized by a quadratic form around the minimum,
this script finds this quadratic form, and parametrizes (ellipsoidal)
sections of constant likelihood.

'''


import numpy as np
from numpy.linalg import svd
from numpy import sqrt
import yaml
from flavio.statistics.functions import pull, delta_chi2
from iminuit import Minuit
from SMEFT19.utils import texnumber


def minimum(fit, x0):

    r'''
Finds the minimum of the fit function and approximates its neighbourhood by an ellipsoid.

:Arguments:
    - fit\: function that takes one point in parameter space and returns
      its negative log-likelihhod.
      Example\: `-SMEFTglob.likelihood_global(x, scenarios.scVI)`.
    - x0\: list or `np.array` containing an initial guess.

:Returns:
    - bf\: np.array with the point in parameter space with the best fit.
    - v\: Unitary matrix containing the axes of the ellipse.
    - d\: diagonal matrix containing the inverse of the squares of the semiaxes.
    - Lmin\: Log-likelihood at the best fit point.
    '''

    print('Minimizing...')
    m = Minuit.from_array_func(fit, x0, error=0.01, errordef=0.5, print_level=0)
    m.migrad()
    m.hesse()
    bf = m.np_values()
    Lmin = m.fval
    dim = len(x0)
    L0 = fit(np.zeros(dim))
    p = pull(2*abs(Lmin-L0), dim)
    print('Pull: ' + str(p) + ' sigma')
    v, d, _ = svd(m.np_matrix())
    d_ellipse = np.diag(1/d)
    return bf, v, d_ellipse, Lmin


def parametrize(x, bf, v, d, nsigmas=1):
    r'''
Maps points on the unit hypersphere to points on the ellipsoid of constant likelihood.

:Arguments:
    - x\: `np.array` containing a point in the surface of the unit `n`-hypersphere.
    - bf\: `np.array` with the point in parameter space with the best fit.
    - v\: `np.matrix` containing the orientation of the axes of the ellipsoid.
    - d\: `np.array` containing the principal axes of the ellipsoid.
    - [nsigmas\: significance of the isoprobability hypersurface with respect to the best fit.]

:Returns:
    - xe: Projection of the point x in the ellipsoid of equal probability
    '''
    r = delta_chi2(nsigmas, len(bf))
    xp = x * sqrt(r/np.diag(d))
    xe = bf + v @ xp
    return np.array(xe).flatten()

def save(bf, v, d, L, filename, name=None, fit=None):
    r'''
Saves the results of the minimization in a `.yaml` file.

:Arguments:
    - bf\: `np.array` with the point in parameter space with the best fit.
    - v\: `np.matrix` containing the orientation of the axes of the ellipsoid.
    - d\: `np.array` containing the principal axes of the ellipsoid
    - filename\: Path to the `.yaml` file where the shape of the ellipse will be saved.
    - L\: Log-likelihood at the best fit point.
    - [name: Descriptive name of the fit.]
    - [fit\: scenario used to fit the data.]
    '''

    values = dict()
    if name is not None:
        values['name'] = name
    values['L'] = L
    if fit is not None:
        values['fit'] = fit
    values['bf'] = bf.tolist()
    values['v'] = v.tolist()
    values['d'] = d.tolist()
    with open(filename, 'wt', encoding='utf-8') as f:
        yaml.dump(values, f)

def load(filename):
    r'''
Loads a ellipse saved in a `.yaml` file to a python dictionary.

:Arguments:
    - filename\: Path to the `.yaml` file where the shape of
      the ellipse has been saved by the `save` method.

:Returns:
    A `python` dictionary containing:
        - bf\: `np.array` with the point in parameter space with the best fit.
        - v\: `np.matrix` containing the orientation of the axes of the ellipsoid.
        - d\: `np.array` containing the principal axes of the ellipsoid.
        - L\: Log-likelihood at the best fit point.
        - [name\: Name of the fit.]
        - [fit\: Scenario used in the fit.]
    '''
    with open(filename, 'rt', encoding='utf-8') as f:
        values = yaml.safe_load(f)
    values['bf'] = np.array(values['bf'])
    values['v'] = np.matrix(values['v'])
    values['d'] = np.array(values['d'])
    return values

def notablepoints(fin, fout, fit):
    r'''
Finds the extrema of the ellipse, the intersection with the coordinate axis
and the closest and furthest point from the origin.

:Arguments:
    - fin\: Path to `.yaml` file containing the information about the ellipse.
    - fout\: Path to `.tex` file to save a table with the coordinates of the notable points.
    - fit\: Function used in the minimization.
    '''
    dbf = load(fin)
    bf = dbf['bf']
    v = dbf['v']
    d = dbf['d']
    n = len(bf)
    p = delta_chi2(1, n)
    ex_p = []
    ex_m = []
    chi2_ex_p = []
    chi2_ex_m = []
    bestchi2 = fit(bf)
    H = v @ d @ v.T
    cross_p = []
    cross_m = []
    chi2_cross_p = []
    chi2_cross_m = []
    for i in range(0, n):
        # Moving along operator axes
        dC = float(np.sqrt(p/H[i, i]))
        delta = np.zeros(n)
        delta[i] = dC
        point_p = bf + delta
        point_m = bf - delta
        chi2_p = 2*(fit(point_p) - bestchi2)
        chi2_m = 2*(fit(point_m) - bestchi2)
        cross_p.append(point_p)
        cross_m.append(point_m)
        chi2_cross_p.append(chi2_p)
        chi2_cross_m.append(chi2_m)
        #Moving along ellipsoid axes
        delta[i] = 1
        point_p = parametrize(delta, bf, v, d)
        point_m = parametrize(-delta, bf, v, d)
        chi2_p = 2*(fit(point_p) - bestchi2)
        chi2_m = 2*(fit(point_m) - bestchi2)
        ex_p.append(point_p)
        ex_m.append(point_m)
        chi2_ex_p.append(chi2_p)
        chi2_ex_m.append(chi2_m)
    bfm = np.matrix(bf)
    dSM = float(np.sqrt(p/(bfm @ H @ bfm.T)))
    SM_p = bf*(1+dSM)
    SM_m = bf*(1-dSM)
    chi2_SM_p = 2*(fit(SM_p) - bestchi2)
    chi2_SM_m = 2*(fit(SM_m) - bestchi2)

    with open(fout, 'w', encoding='utf-8') as f:
        f.write(r'\begin{tabular}{|' + 'c|'*(n+3) + r'}\hline' + '\n')
        f.write(r'$j$ & $s$ & ' + ' & '*n + r'$\Delta \chi^2$\\\hline' + '\n')
        f.write(r'BF & & ')
        for i in range(0, n):
            f.write(texnumber(bf[i]) + ' & ')
        f.write(r'\\\hline' + '\n')
        for i in range(0, n):
            f.write(str(i+1) + ' & $+$ & ')
            for j in range(0, n):
                f.write(texnumber(ex_p[i][j]) + ' & ')
            f.write(texnumber(chi2_ex_p[i]) + r'\\\hline' + '\n')
            f.write(str(i+1) + ' & $-$ & ')
            for j in range(0, n):
                f.write(texnumber(ex_m[i][j]) + ' & ')
            f.write(texnumber(chi2_ex_m[i]) + r'\\\hline' + '\n')
        for i in range(0, n):
            f.write(' & $+$ & ')
            for j in range(0, n):
                f.write(texnumber(cross_p[i][j]) + ' & ')
            f.write(texnumber(chi2_cross_p[i]) + r'\\\hline' + '\n')
            f.write(' & $-$ & ')
            for j in range(0, n):
                f.write(texnumber(cross_m[i][j]) + ' & ')
            f.write(texnumber(chi2_cross_m[i]) + r'\\\hline' + '\n')
        f.write(r'SM & $+$ & ')
        for i in range(0, n):
            f.write(texnumber(SM_p[i]) + ' & ')
        f.write(texnumber(chi2_SM_p) +  r' \\\hline' + '\n')
        f.write(r'SM & $-$ & ')
        for i in range(0, n):
            f.write(texnumber(SM_m[i]) + ' & ')
        f.write(texnumber(chi2_SM_m) +  r' \\\hline' + '\n')
        f.write(r'\end{tabular}')
