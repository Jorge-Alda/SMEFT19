import flavio
import flavio.statistics.fits
import flavio.plots
from flavio.statistics.functions import pull
import matplotlib.pyplot as plt
import numpy as np
import smelli
from math import isinf
import warnings


fits = ['likelihood_lfu_fcnc.yaml', 'likelihood_rd_rds.yaml', 'likelihood_ewpt.yaml', 'global']
labels = {'likelihood_lfu_fcnc.yaml':r'$R_{K^{(*)}}$', 'likelihood_rd_rds.yaml':r'$R_{D^{(*)}}$', 'likelihood_ewpt.yaml': 'EW precission', 'global':'Global'}


gl = smelli.GlobalLikelihood()		

cache = dict()
def update_cache(x, wfun):
	if x not in cache.keys():
		cache[x] = dict()
		with warnings.catch_warnings():	
			warnings.simplefilter('ignore')
			glpp = gl.parameter_point(wfun(x))
			gldict = glpp.log_likelihood_dict()
			for f in fits:
				g = gldict[f]
				if isinf(g):
					if f == 'global':
						g = 0
						for f2 in fits[:-1]:
							g += cache[x][f2]
					else: 
						g = -68
				cache[x][f] = g
				

def clearcache():
	global cache
	cache = dict()

def likelihood_fit_cached(x, wfun, f):
	xt = tuple(x)
	update_cache(xt, wfun)
	return cache[xt][f]

def likelihood_global(x, wfun):
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		glpp = gl.parameter_point(wfun(x))
		return glpp.log_likelihood_global()

def plot(wfun, xmin, xmax, ymin, ymax, axlabels, fout, locleg=0):
	import texfig # https://github.com/knly/texfig
	fig=plt.figure(figsize=(4,4))
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])

	i=0
	for f in fits:
		print('Plotting ' + f) 
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			loglike = lambda x: likelihood_fit_cached(x, wfun, f)
			flavio.plots.likelihood_contour(loglike , 1.1*xmin, 1.1*xmax, 1.1*ymin, 1.1*ymax, col=i, label=labels[f], interpolation_factor=5, n_sigma=(1,2), steps=55)
		i+=1
	plt.xlabel(axlabels[0])
	plt.ylabel(axlabels[1])
	plt.axhline(0, color='black', linewidth=0.5)
	plt.axvline(0, color='black', linewidth=0.5)
	ax = fig.gca()
	ax.xaxis.set_ticks_position('both')
	ax.yaxis.set_ticks_position('both')
	plt.legend(loc = locleg)
	plt.tight_layout(pad=0.5)
	texfig.savefig(fout)
	clearcache()

def fastmeas(obs):
	obsm = gl.obstable_sm[obs]
	lhname = obsm['lh_name']
	return lhname[:4]=='fast'

def prediction(obs, x, wfun):
	obsm = gl.obstable_sm[obs]
	lhname = obsm['lh_name']
	wc = wfun(x)
	if fastmeas(obs):
		lh = gl.fast_likelihoods[lhname]
		m = lh.pseudo_measurement
		ml = lh.likelihood.measurement_likelihood
		pred = ml.get_predictions_par(gl.par_dict, wc)
		return pred[obs]
	else:
		lh = gl.likelihoods[lhname]
		ml = lh.measurement_likelihood
		pred = ml.get_predictions_par(gl.par_dict, wc)
		return pred[obs]

def pull_obs(obs, x, wfun):
	obsm = gl.obstable_sm[obs]
	lhname = obsm['lh_name']
	wc = wfun(x)
	pred = prediction(obs, x, wfun)
	ll_central = obsm['ll_central']
	if fastmeas(obs):
		lh = gl.fast_likelihoods[lhname]
		m = lh.pseudo_measurement
		ll = m.get_logprobability_single(obs, pred)
	else:
		p_comb = obsm['exp. PDF']
		ll = p_comb.logpdf([pred])
	return pull(-2*(ll-ll_central), 1)
