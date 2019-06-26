import flavio
from flavio.statistics.functions import pull
import numpy as np
import smelli
from math import isinf
import warnings



gl = smelli.GlobalLikelihood()		

def likelihood_fits(x, wfun):
	res = dict()
	with warnings.catch_warnings():	
		warnings.simplefilter('ignore')
		glpp = gl.parameter_point(wfun(x))
		gldict = glpp.log_likelihood_dict()
		for f in gldict.keys():
			g = gldict[f]
			if isinf(g):
				if f == 'global':
					g = 0
					for f2 in list(gldict.keys())[:-1]:
						g += res[f2]
				else: 
					g = -68
			res[f] = g
	return res


def likelihood_global(x, wfun):
	return likelihood_fits(x, wfun)['global']

def fastmeas(obs):
	obsm = gl.obstable_sm[obs]
	lhname = obsm['lh_name']
	return lhname[:4]=='fast'

def prediction(x, obs, wfun):
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

def pull_obs(x, obs, wfun):
	obsm = gl.obstable_sm[obs]
	lhname = obsm['lh_name']
	wc = wfun(x)
	pred = prediction(x, obs, wfun)
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
