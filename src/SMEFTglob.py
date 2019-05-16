import flavio
import flavio.statistics.fits
import flavio.plots
from flavio.statistics.functions import pull
import matplotlib.pyplot as plt
import numpy as np
import smelli
from math import isinf
import warnings

'''
WARNING: To use the latest experimental values of RK(*), you must work with the developement (github) version of flavio and modify some files of smelli
'''

fits = ['likelihood_lfu_fcnc.yaml', 'likelihood_rd_rds.yaml', 'likelihood_ewpt.yaml', 'global']
labels = {'likelihood_lfu_fcnc.yaml':r'$R_{K^{(*)}}$', 'likelihood_rd_rds.yaml':r'$R_{D^{(*)}}$', 'likelihood_ewpt.yaml': 'EW precission', 'global':'Global'}

gl = smelli.GlobalLikelihood()

cache = dict()
plotting = False
point = 0
def update_cache(x, wfun):
	if x not in cache.keys():
		cache[x] = dict()
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				glpp = gl.parameter_point(wfun(x))
				gldict = glpp.log_likelihood_dict()
				for f in fits:
					cache[x][f] = gldict[f]

def likelihood_fit_cached(x, wfun, f):
	xt = tuple(x)
	update_cache(xt, wfun)
	return cache[xt][f]

def likelihood_global(x, wfun):
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		glpp = gl.parameter_point(wfun(x))
		return glpp.log_likelihood_global()

def plot(wfun, xmin, xmax, ymin, ymax):
	import texfig # https://github.com/knly/texfig
	fig=plt.figure(figsize=(4,4))
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])

	i=0
	for f in fits:
		print('Plotting ' + f) 
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			loglike = lambda x: likelihood_fit(x, wfun, f)
			flavio.plots.likelihood_contour(loglike , 1.1*xmin, 1.1*xmax, 1.1*ymin, 1.1*ymax, col=i, label=labels[f], interpolation_factor=5, n_sigma=(1,2))
		i+=1
	plt.xlabel(r'$C_{\ell q(1)}^{\mu} = C_{\ell q(3)}^{\mu}$')
	plt.ylabel(r'$C_{\ell q(1)}^{\tau} = C_{\ell q(3)}^{\tau}$')
	plt.legend(loc=4)
	texfig.savefig('SMEFT_C13_mutau')
