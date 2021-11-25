'''
=========
plots
=========

This module contains functions to plot the results of the fits.
'''

import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors
import scipy.interpolate
import numpy as np
import yaml
from flavio.statistics.functions import delta_chi2
import flavio.plots.colors
from . import SMEFTglob
from .SMEFTglob import loadobslist
from .ellipse import load
from .comparepulls import pullevolution


plt.rcParams.update({'pgf.texsystem':'pdflatex'})
plt.rcParams['hatch.color'] = 'w'

hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


def listpoint(x):
    if len(x) == 2:
        if len(np.array(x).flat) == 2:
            return [x,]
        else:
            return x
    else:
        return x

def likelihood_plot(grid, xmin, xmax, ymin, ymax, axlabels, fout=None, locleg=0, n_sigma=(1, 2),
                    colors=None, styles=None, widths=None, ticks=0.5, bf=None):
    r'''
Plots a contour plot of the log-likelihood of the fit.

:Arguments:

    - grid\: List containing the x coordinates, y corrdinates
             and a dictionary for the likelihood values in the grid.
    - xmin\: Minimum value of the `x` coordinate.
    - xmax\: Maximum value of the `x` coordinate.
    - ymin\: Minimum value of the `y` coordinate.
    - ymax\: Maximum value of the `y` coordinate.
    - axlabels\: List containing two strings to label the `x` and `y` axes.
    - [fout\: Path to the files where the plots will be saved.
              Two files are created, one `.pdf` and one `.pgf` (to use in TeX).
              Extensions are added automatically.]
    - [locleg\: Position of the legend of the plot, using `matplotlib`'s syntaxis.
                Default=0 (best position).]
    - [n_sigma\: List containing the significance (in sigmas) of each contour. Default = (1,2).]
    - [colors\: List with the colors of each contour. Default: flavio palette.]
    - [styles\: List with the linestyles of each contour. Default: All solid.]
    - [widths\: List with the linewidths of each contour. Default: All 1pt.]
    - [ticks\: Interval between ticks in both axes. Default:0.5]
    - [bf\: Coordinates of the best fit point(s). It can be `None` (no point marked)
            or a list containing two floats (one point marked). Default: `None`.]
    '''

    fig = plt.figure(figsize=(6, 6))
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    nfits = len(grid[2])
    if colors is None:
        _cols = [i%9 for i in range(nfits)]
    else:
        _cols = colors
    if styles is None:
        lstyle = ['solid',]*nfits
    else:
        lstyle = styles
    if widths is None:
        lwidths = [1,]*nfits
    else:
        lwidths = widths
    x = grid[0]
    y = grid[1]
    zl = grid[2]
    for i, z in enumerate(zl.values()):
        chi = -2*(z.T-np.max(z))
        # get the correct values for 2D confidence/credibility contours for n sigma
        if isinstance(n_sigma, float) or isinstance(n_sigma, int):
            levels = [delta_chi2(n_sigma, dof=2)]
        else:
            levels = [delta_chi2(n, dof=2) for n in n_sigma]
        hatch_contour(x=x, y=y, z=chi, levels=levels, col=_cols[i], label=list(zl.keys())[i],
                      interpolation_factor=5, hatched=False,
                      contour_args={'linestyles':lstyle[i], 'linewidths':lwidths[i]})
        if bf is not None:
            plt.scatter(*bf, marker='x', color='black')
    plt.xlabel(axlabels[0], fontsize=18)
    plt.ylabel(axlabels[1], fontsize=18)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    ax = fig.gca()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks(np.arange(xmin, xmax+1e-5, ticks))
    ax.yaxis.set_ticks(np.arange(ymin, ymax+1e-5, ticks))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=locleg, fontsize=16)
    plt.tight_layout(pad=0.5)
    if fout is not None:
        fig.savefig(fout+'.pdf')
        fig.savefig(fout+'.pgf')

def hatch_contour(x, y, z, levels, interpolation_factor=1, interpolation_order=2, col=0,
                  label=None, hatched=True, contour_args=None, contourf_args=None):
    r"""
Plots coloured and hatched confidence contours (or bands) given numerical
input arrays.Based on the `flavio` function

:Arguments:

    - x, y\: 2D arrays containg x and y values as returned by numpy.meshgrid
    - z\: value of the function to plot. 2D array in the same shape as `x` and
      `y`. The lowest value of the function should be 0 (i.e. the best fit point).
    - levels\: list of function values where to draw the contours. They should
      be positive and in ascending order.
    - [interpolation factor\:: in between the points on the grid,
      the functioncan be interpolated to get smoother contours.
      This parameter sets the number of subdivisions (default: 1, i.e. no
      interpolation). It should be larger than 1.]
    - [col\: number between 0 and 9 to choose the color of the plot
      from a predefined palette.]
    - [label\: label that will be added to a legend created with `maplotlib.pyplot.legend()`.]
    - [contour_args\: dictionary of additional options that will be passed to
                      `matplotlib.pyplot.contour()` (that draws the contour lines).]
    - [contourf_args\: dictionary of additional options that will be passed to
                       `matplotlib.pyplot.contourf()` (that paints the contour filling).]
    """
    contour_args = contour_args or {}
    contourf_args = contour_args or {}
    if interpolation_factor > 1:
        x = scipy.ndimage.zoom(x, zoom=interpolation_factor, order=1)
        y = scipy.ndimage.zoom(y, zoom=interpolation_factor, order=1)
        z = scipy.ndimage.zoom(z, zoom=interpolation_factor, order=interpolation_order)
    if isinstance(col, int):
        _contour_args = {}
        _contourf_args = {}
        _contour_args['colors'] = [flavio.plots.colors.set1[col]]
        _contour_args['linewidths'] = 1.2
        N = len(levels)
        _contourf_args['colors'] = [flavio.plots.colors.pastel[col] +
                                    (max(1-n/(N+1), 0),) for n in range(1, N+1)]
    else:
        _contour_args = {}
        _contourf_args = {}
        _contour_args['colors'] = [darken_color(matplotlib.colors.to_rgb(col), 0.7)]
        _contour_args['linewidths'] = 1.2
        N = len(levels)
        _contourf_args['colors'] = [matplotlib.colors.to_rgb(col) +
                                    (max(1-n/(N+1), 0),) for n in range(1, N+1)]

    if hatched:
        hl = []
        for i in range(0, N):
            hl.append(hatches[col]*(N-i))
        hl.append(None)
        _contourf_args['hatches'] = hl
    _contour_args['linestyles'] = 'solid'
    _contour_args.update(contour_args)
    _contourf_args.update(contourf_args)
    # for the filling, need to add zero contour
    levelsf = [np.min(z)] + list(levels)
    ax = plt.gca()
    CF = ax.contourf(x, y, z, levels=levelsf, **_contourf_args)
    CS = ax.contour(x, y, z, levels=levels, **_contour_args)
    if label is not None:
        CS.collections[0].set_label(label)
    return (CS, CF)


def error_plot(fout, plottype, flist, flist2=None, legend=0):
    r'''
Plots the uncertainty intervals for several observables in NP scenarios, SM and experimental values.

:Arguments:
    - fout\: Path to the files where the plots will be saved.
             Two files are created, one `.pdf` and one `.pgf` (to use in TeX).
             Extensions are added automatically.
    - plottype\: Selects the observables to be plotted\:

        - 'RK'\: Plots RK in the [1.1,6.0] bin and RK\* in the [0.045,1.1] and [1.1,6] bins.
        - 'RD'\: Plots RD, and RD\* using only muons or muons+electrons.
    - flist\: List of paths to files created by `obsuncert.calculate`.
    - flist2\: Additional list of paths to files created by `obsuncert.calculate`.
    - legend\: 0 for no legend, 1 for legend next to the plot and 2 for legend inside the plot.
    '''
    if legend < 2:
        fig = plt.figure(figsize=(5.7+2.3*legend, 5))
    else:
        fig = plt.figure()
    if plottype == 'RD':
        observables = ['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)']
        texlabels = [r'$R_D^\ell$', r'$R_{D^*}^\ell$', r'$R_{D^*}^\mu$']
        #legloc = 1
    elif plottype == 'RK':
        observables = [('<Rmue>(B+->Kll)', 1.1, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1),
                       ('<Rmue>(B0->K*ll)', 1.1, 6.0)]
        texlabels = [r'$R_K^{[1.1,6]}$', r'$R_{K^*}^{[0.045, 1.1]}$', r'$R_{K^*}^{[1.1, 6]}$']
        #legloc = 3
    nobs = len(texlabels)
    nhyp = len(flist)
    ax = plt.gca()
    plt.xlim([0, nobs])
    #plt.ylim([-0.055, 0.015])
    markers = ['o', '^', 's', '*', 'D']

    #data = np.zeros([nhyp, nobs,2])
    #smdata = np.zeros([nobs,2])
    #expdata = np.zeros([nobs,2])
    data = [[{'central':0, 'uncert':0} for i in range(nobs)] for j in range(nhyp)]
    data2 = [[{'central':0, 'uncert':0} for i in range(nobs)] for j in range(nhyp)]
    smdata = [{'central': 0, 'uncert': 0} for i in range(nobs)]
    expdata = [{'central': 0, 'uncert': 0} for i in range(nobs)]
    leglabels = []
    leglabels2 = []
    hyp = 0
    for fin in flist:
        f = open(fin, 'rt', encoding='utf-8')
        values = yaml.safe_load(f)
        f.close()
        try:
            leglabels.append(values['name'])
        except KeyError:
            leglabels.append(fin[:-5])

        o = 0
        for obs in observables:
            data[hyp][o]['central'] = values[str(obs)]['NP']['central']
            data[hyp][o]['uncert'] = values[str(obs)]['NP']['uncert']
            smdata[o]['central'] = values[str(obs)]['SM']['central']
            smdata[o]['uncert'] = values[str(obs)]['SM']['uncert']
            expdata[o]['central'] = values[str(obs)]['exp']['central']
            expdata[o]['uncert'] = values[str(obs)]['exp']['uncert']
            o += 1
        hyp += 1
    hyp = 0
    if flist2 is not None:
        for fin in flist2:
            f = open(fin, 'rt', encoding='utf-8')
            values = yaml.safe_load(f)
            f.close()
            try:
                leglabels2.append(values['name'])
            except KeyError:
                leglabels2.append(fin[:-5])
            o = 0
            for obs in observables:
                data2[hyp][o]['central'] = values[str(obs)]['NP']['central']
                data2[hyp][o]['uncert'] = values[str(obs)]['NP']['uncert']
                o += 1
            hyp += 1

    for o in range(0, nobs):
        for i in range(0, nhyp):
            if o == 0:
                plt.plot(o+(i+1)/(nhyp+1), data[i][o]['central'],
                         marker=markers[i], color='b', label=leglabels[i], zorder=3)
            else:
                plt.plot(o+(i+1)/(nhyp+1), data[i][o]['central'],
                         marker=markers[i], color='b', zorder=3)
            plt.errorbar(o+(i+1)/(nhyp+1), data[i][o]['central'],
                         yerr=data[i][o]['uncert'], color='b', zorder=3)
            if flist2 is not None:
                if o == 0:
                    plt.plot(o+(i+1.5)/(nhyp+1), data2[i][o]['central'],
                             marker=markers[i], color='r', label=leglabels2[i], zorder=3)
                else:
                    plt.plot(o+(i+1.5)/(nhyp+1), data2[i][o]['central'],
                             marker=markers[i], color='r', zorder=3)
                plt.errorbar(o+(i+1.5)/(nhyp+1), data2[i][o]['central'],
                             yerr=data2[i][o]['uncert'], color='r', zorder=3)
        if isinstance(smdata[o]['uncert'], list):
            smleft = smdata[o]['uncert'][0]
            smrange = smdata[o]['uncert'][0] + smdata[o]['uncert'][1]
        else:
            smleft = smdata[o]['uncert']
            smrange = 2*smdata[o]['uncert']
        if isinstance(expdata[o]['uncert'], list):
            expleft = expdata[o]['uncert'][0]
            exprange = expdata[o]['uncert'][0] + expdata[o]['uncert'][1]
        else:
            expleft = expdata[o]['uncert']
            exprange = 2*expdata[o]['uncert']
        if o == 0:
            ax.add_patch(Rectangle((o+0.05, smdata[o]['central']-smleft), 0.9,
                                   smrange, color='orange', alpha=0.5, label='SM', lw=0))
            ax.add_patch(Rectangle((o+0.05, expdata[o]['central']-expleft), 0.9,
                                   exprange, color='green', alpha=0.5, label='Experimental', lw=0))
        else:
            ax.add_patch(Rectangle((o+0.05, expdata[o]['central']-expleft), 0.9,
                                   exprange, color='green', alpha=0.5, lw=0))
            ax.add_patch(Rectangle((o+0.05, smdata[o]['central']-smleft), 0.9,
                                   smrange, color='orange', alpha=0.5, lw=0))
        plt.plot([o+0.05, o+0.95], [smdata[o]['central'], smdata[o]['central']],
                 lw=1, color='orange', ls='dashed')
        plt.plot([o+0.05, o+0.95], [expdata[o]['central'], expdata[o]['central']],
                 lw=1, color='green', ls='dashed')


    ax.set_xticks(np.linspace(0.5, nobs-0.5, nobs))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xticklabels(texlabels + [''])
    if legend == 1:
        plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    elif legend > 1:
        plt.legend(fontsize=14)
    plt.tight_layout(pad=0.5)
    fig.savefig(fout + '.pdf')
    fig.savefig(fout + '.pgf')

def binerrorbox(binmin, binmax, central, error, centralline=False, **kwargs):
    ax = plt.gca()
    if isinstance(error, float):
        errormin = error
        errormax = error
    else:
        errormin = error[0]
        errormax = error[1]

    ax.add_patch(Rectangle((binmin, central-errormin), binmax-binmin, errormin+errormax, **kwargs))
    if centralline:
        plt.plot([binmin, binmax], [central, central], **kwargs)

def compare_plot(wfun, fin, fout, sigmas=1):
    r'''
Plots the pull of each observable in the SM and in the NP hypothesis.

:Arguments:

    - wfun\: Function that takes a point in parameter space
             and returns a dictionary of Wilson coefficents.
    - fin\: Path to the `.yaml` file where the ellipsoid is saved.
    - fout\: Path to the files where the plots will be saved.
             Two files are created, one `.pdf` and one `.pgf` (to use in TeX).
             Extensions are added automatically.
    '''
    dbf = load(fin)
    bf = dbf['bf']

    w = wfun(bf)
    gl = SMEFTglob.gl
    glNP = gl.parameter_point(w)
    glSM = gl.parameter_point({}, scale=1000)
    obsSM = glSM.obstable()
    obsNP = glNP.obstable()
    obscoll = loadobslist()

    NP = []
    SM = []
    for obs in obscoll:
        NP.append(float(obsNP.loc[[obs], 'pull exp.']))
        SM.append(float(obsSM.loc[[obs], 'pull exp.']))

    fig = plt.figure()
    plt.plot(NP, label='New Physics')
    plt.plot(SM, label='Standard Model')
    vertplus = 0
    vertminus = 0
    for i, (SMi, NPi) in enumerate(zip(SM, NP)):
        if (NPi-SMi) > sigmas:
            v = 0.3 + vertplus
            vertplus += 0.1
            plt.annotate(str(i), xy=(i, NPi), xytext=(i, NPi+v), fontsize=6,
                         horizontalalignment='right',
                         arrowprops=dict(facecolor='black', arrowstyle='->'))
        elif (SMi-NPi) > sigmas:
            v = 0.3 + vertminus
            #vertminus += 0.1
            plt.annotate(str(i), xy=(i, NPi), xytext=(i, NPi-v), fontsize=6,
                         horizontalalignment='left',
                         arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.xlabel('Observable')
    plt.ylabel(r'$|$Pull$|$')
    plt.legend(loc=1)
    plt.tight_layout(pad=0.5)
    fig.savefig(fout+'.pdf')
    fig.savefig(fout+'.pgf')

def evolution_plot(obscodes, wfun, fin, direction, fout, obsnames=None):
    r'''
Plots the vairation of the pull of several observables along a line
connecting two opposite  notable points of the ellipsoid.

:Arguments:

    - obscodes\: List of ID-Numbers of the observables, as returned by `comparepulls.pointpull`
    - wfun\: Function that takes a point in parameter space
             and returns a dictionary of Wilson coefficents.
    - fin\: Path to the `.yaml` file where the ellipsoid is saved.
    - direction\: string with the following format\:

            - 'ax' + str(i)\: for the i-th principal axis of the ellipsoid.
            - 'sm'\: for the direction joining the bf and sm points.

    - fout\: Path to the files where the plots will be saved.
             Two files are created, one `.pdf` and one `.pgf` (to use in TeX).
             Extensions are added automatically.
    '''
    fig = plt.figure()
    j = 0
    for o in obscodes:
        ev = pullevolution(o, wfun, fin, direction)
        if obsnames is None:
            n = ''
        else:
            n = '\t' + obsnames[j]
            j += 1
        plt.plot(np.linspace(-1, 1, 200), ev, label='Obs. ' + str(o)+n)
    if direction[:2] == 'ax':
        i = direction[2:]
        plt.xlabel(r'$\delta C_{' + i + '}/a_{' + i + '}$')
    if direction[:2] == 'sm':
        plt.xlabel(r'$C_\mathrm{SM}/a_\mathrm{SM}$')
    plt.ylabel('Pull')
    plt.axvline(0, color='black', linewidth=0.5)
    ax = fig.gca()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.legend()
    plt.tight_layout(pad=0.5)
    fig.savefig(fout+'.pdf')
    fig.savefig(fout+'.pgf')

def darken_color(color, amount=0.5):
    """
    Darkens the given color by multiplying luminosity by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = matplotlib.colors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], amount * c[1], c[2])
