from . import SMEFTglob
from .SMEFTglob import likelihood_fits, loadobslist
from .comparepulls import pullevolution
import warnings
import matplotlib.pyplot as plt
import texfig # https://github.com/knly/texfig
plt.rcParams['hatch.color'] = 'w'
from matplotlib.patches import Rectangle
from flavio.statistics.functions import delta_chi2, confidence_level
import flavio.plots.colors
import scipy.interpolate
import numpy as np
import yaml
from parscanning import GridScan


hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


def listpoint(x):
	if len(x) == 2:
		if len(np.array(x).flat) == 2:
			return [x,]
		else:
			return x
	else:
		return x

def likelihood_plot(wfun, xmin, xmax, ymin, ymax, fits, axlabels, fout=None, locleg=0, n_sigma=(1,2), steps=55, hatched=False, threads=1, bf=None):
	fitcodes = {'RK':'likelihood_lfu_fcnc.yaml', 'RD':'likelihood_rd_rds.yaml', 'EW':'likelihood_ewpt.yaml', 'LFV':'likelihood_lfv.yaml', 'ZLFV':'likelihood_zlfv.yaml', 'global':'global'}
	labels = {'RK':r'$R_{K^{(*)}}$', 'RD':r'$R_{D^{(*)}}$', 'EW': 'EW precission', 'LFV':'LFV', 'ZLFV':r'$Z$ LFV',  'global':'Global'}
	fig=plt.figure(figsize=(4,4))
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	colors = [0,1,2,4,5,6,7]
	xmargin = 0.02*(xmax-xmin)
	ymargin = 0.02*(ymax-ymin)
	GS = GridScan(likelihood_fits, [xmin-xmargin, ymin-ymargin], [xmax+xmargin, ymax+ymargin], steps)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		if threads == 1:
			GS.run(wfun)
		else:
			GS.run_mp(threads, wfun)


	for i, f in enumerate(fits):
		(x, y, z) = GS.meshdata(fitcodes[f])
		chi = -2*(z-np.max(z))
		# get the correct values for 2D confidence/credibility contours for n sigma
		if isinstance(n_sigma, float) or isinstance(n_sigma, int):
			levels = [delta_chi2(n_sigma, dof=2)]
		else:
			levels = [delta_chi2(n, dof=2) for n in n_sigma]
		hatch_contour(x=x, y=y, z=chi, levels=levels, col=colors[i], label=labels[f], interpolation_factor=5, hatched=hatched)


	if bf is not None:
		for p in listpoint(bf):
			plt.scatter(*p, marker='x', s=15, c='black')

	plt.xlabel(axlabels[0])
	plt.ylabel(axlabels[1])
	plt.axhline(0, color='black', linewidth=0.5)
	plt.axvline(0, color='black', linewidth=0.5)
	ax = fig.gca()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.legend(loc = locleg)
	plt.tight_layout(pad=0.5)
	if fout is not None:
		texfig.savefig(fout)

def hatch_contour(x, y, z, levels,
              interpolation_factor=1,
              interpolation_order=2,
              col=0, label=None,
              hatched=True,
              contour_args={}, contourf_args={}):
    r"""Plot coloured and hatched confidence contours (or bands) given numerical input
    arrays. Based on the flavio function

    Parameters:

    - `x`, `y`: 2D arrays containg x and y values as returned by numpy.meshgrid
    - `z` value of the function to plot. 2D array in the same shape as `x` and
      `y`. The lowest value of the function should be 0 (i.e. the best fit
      point).
    - levels: list of function values where to draw the contours. They should
      be positive and in ascending order.
    - `interpolation factor` (optional): in between the points on the grid,
      the functioncan be interpolated to get smoother contours.
      This parameter sets the number of subdivisions (default: 1, i.e. no
      interpolation). It should be larger than 1.
    - `col` (optional): number between 0 and 9 to choose the color of the plot
      from a predefined palette
    - `label` (optional): label that will be added to a legend created with
       maplotlib.pyplot.legend()
    - `contour_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contour() (that draws the contour lines)
    - `contourf_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contourf() (that paints the contour filling).
    """
    if interpolation_factor > 1:
        x = scipy.ndimage.zoom(x, zoom=interpolation_factor, order=1)
        y = scipy.ndimage.zoom(y, zoom=interpolation_factor, order=1)
        z = scipy.ndimage.zoom(z, zoom=interpolation_factor, order=interpolation_order)
    if not isinstance(col, int):
        _col = 0
    else:
        _col = col
    _contour_args = {}
    _contourf_args = {}
    _contour_args['colors'] = [flavio.plots.colors.set1[_col]]
    _contour_args['linewidths'] = 1.2
    N = len(levels)
    _contourf_args['colors'] = [flavio.plots.colors.pastel[_col] # RGB
                                       + (max(1-n/(N+1), 0),) # alpha, decreasing for contours
                                       for n in range(1,N+1)]
    if hatched:
        hl = []
        for i in range(0, N):
            hl.append(hatches[_col]*(N-i))
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


def error_plot(flist, plottype, fout):
	fig = texfig.figure()
	if plottype == 'RD':
		observables = ['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)']
		texlabels = [r'$R_D^\ell$', r'$R_{D^*}^\ell$', r'$R_{D^*}^\mu$']
		#legloc = 1
	elif plottype == 'RK':
		observables = [('<Rmue>(B+->Kll)', 1.1, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0)]
		texlabels = [r'$R_K^{[1.1,6]}$', r'$R_{K^*}^{[0.045, 1.1]}$', r'$R_{K^*}^{[1.1, 6]}$']
		#legloc = 3
	nobs = len(texlabels)
	nhyp = len(flist)
	ax=plt.gca()
	plt.xlim([0, nobs])
	#plt.ylim([-0.055, 0.015])
	markers = ['o', '^', 's', '*', 'D']

	data = np.zeros([nhyp, nobs,2])
	smdata = np.zeros([nobs,2])
	expdata = np.zeros([nobs,2])
	leglabels = []
	hyp = 0
	for fin in flist:
		f = open(fin, 'rt')
		values = yaml.load(f)
		f.close()
		try:
			leglabels.append(values['name'])
		except:
			leglabels.append(fin[:-5])

		o = 0
		for obs in observables:
			data[hyp][o][0] = values[str(obs)]['NP']['central']
			data[hyp][o][1] = values[str(obs)]['NP']['uncert']
			smdata[o][0] = values[str(obs)]['SM']['central']
			smdata[o][1] = values[str(obs)]['SM']['uncert']
			expdata[o][0] = values[str(obs)]['exp']['central']
			expdata[o][1] = values[str(obs)]['exp']['uncert']
			o += 1
		hyp += 1

	for o in range(0, nobs):
		for i in range(0, nhyp):
			if o==0:
				plt.plot(o+(i+1)/(nhyp+1), data[i][o][0], marker=markers[i], color='b', label=leglabels[i])
			else:
				plt.plot(o+(i+1)/(nhyp+1), data[i][o][0], marker=markers[i], color='b')
			plt.errorbar(o+(i+1)/(nhyp+1), data[i][o][0], yerr=data[i][o][1], color='b')

		if o==0:
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7, label='SM'))
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7, label='Experimental'))
		else:
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7))
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7))


	ax.set_xticks(np.linspace(0.5, nobs-0.5, nobs) )
	ax.set_xticklabels(texlabels + [''])
	plt.legend()
	texfig.savefig(fout)

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

def compare_plot(wfun, fin, fout):
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

	plt.figure()
	plt.plot(NP, label='New Physics')
	plt.plot(SM, label='Standard Model')
	vertplus = 0
	vertminus = 0
	for i in range(0, len(SM)):
		if (NP[i]-SM[i]) > 1:
			v = 0.3 + vertplus
			vertplus += 0.1
			plt.annotate(str(i), xy=(i, NP[i]), xytext=(i, NP[i]+v), fontsize=6, horizontalalignment='right', arrowprops = dict(facecolor = 'black',  arrowstyle='->') )
		elif (SM[i]-NP[i]) > 1:
			v = 0.3 + vertminus
			#vertminus += 0.1
			plt.annotate(str(i), xy=(i, NP[i]), xytext=(i, NP[i]-v), fontsize=6, horizontalalignment='left', arrowprops = dict(facecolor = 'black',  arrowstyle='->') )
	plt.xlabel('Observable')
	plt.ylabel(r'$|$Pull$|$')
	plt.legend(loc=1)
	plt.tight_layout(pad=0.5)
	texfig.savefig(fout)

def evolution_plot(obscodes, wfun, fin, direction, fout):
	fig = plt.figure()
	for o in obscodes:
		ev = pullevolution(o, wfun, fin, direction)
		plt.plot(np.linspace(-1, 1, 200), ev, label='Obs. ' + str(o))
	if direction[:2] == 'ax':
		i = direction[2:]
		plt.xlabel('$\delta C_{' + i + '}/a_{' + i + '}$')
	if direction[:2] == 'sm':
		plt.xlabel(r'$C_\mathrm{SM}/a_\mathrm{SM}$')
	plt.ylabel('Pull')
	plt.axvline(0, color='black', linewidth=0.5)
	ax = fig.gca()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.legend()
	plt.tight_layout(pad=0.5)
	texfig.savefig(fout)
