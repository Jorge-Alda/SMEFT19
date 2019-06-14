import flavio
from flavio.statistics.functions import pull
import numpy as np
import smelli
from math import isinf



#fits = ['likelihood_lfu_fcnc.yaml', 'likelihood_rd_rds.yaml', 'likelihood_ewpt.yaml', 'global']
#labels = {'likelihood_lfu_fcnc.yaml':r'$R_{K^{(*)}}$', 'likelihood_rd_rds.yaml':r'$R_{D^{(*)}}$', 'likelihood_ewpt.yaml': 'EW precission', 'global':'Global'}
fits = ['likelihood_lfu_fcnc.yaml', 'likelihood_rd_rds.yaml','likelihood_lfv.yaml','global']
labels = {'likelihood_lfu_fcnc.yaml':r'$R_{K^{(*)}}$', 'likelihood_rd_rds.yaml':r'$R_{D^{(*)}}$', 'likelihood_lfv.yaml':'LFV', 'likelihood_ewpt.yaml': 'EW precission', 'global':'Global'}


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


def loadobslist():
	try:
		fyaml = open('observables.yaml', 'rt')
		obscoll = yaml.load(fyaml)
		fyaml.close()
	except:	
		gl = SMEFTglob.gl
		glSM = gl.parameter_point({}, scale=1000)
		obsSM = glSM.obstable()
		obscoll = list(obsSM['pull exp.'].keys())
		fyaml = open('observables.yaml', 'wt')
		yaml.dump(obscoll, fyaml)
		fyaml.close()
	return obscoll
