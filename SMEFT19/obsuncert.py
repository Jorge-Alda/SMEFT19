from . import obs_bctaunu
from . import ellipse
from wilson import Wilson
import flavio
from math import sqrt
import numpy as np
import yaml
from parscanning import MontecarloScan
from .SMEFTglob import likelihood_global, prediction


obslist = [('<Rmue>(B+->Kll)', 1.1, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), 'Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)', 'Rtaul(Bc->J/psilnu)']

def distrsphere(dim):
	vect = np.random.randn(dim)
	return vect/np.linalg.norm(vect)

def _variance(x, obs, wfun, central):
	return (prediction(x, obs, wfun)-central)**2

def calculate(wfun, minx, maxx, fout, bf, num=50, cores=1):
	values = dict() 
	inf = float('Inf')
	uncert = []
	maxval = [-inf]*len(obslist)
	minval = [inf]*len(obslist)

	w = wfun(bf)
	for obs in obslist:
		values[str(obs)] = dict()
		if isinstance(obs, str):
			values[obs]['SM'] = dict()
			values[obs]['exp'] = dict()
			values[obs]['NP'] = dict()
			values[obs]['NP']['central'] = float(flavio.np_prediction(obs, w))
			uncert.append(flavio.np_uncertainty(obs, w))
			values[obs]['SM']['central'] = float(flavio.sm_prediction(obs))
			values[obs]['SM']['uncert'] = float(flavio.sm_uncertainty(obs))
			dist = flavio.combine_measurements(obs)
			values[obs]['exp']['central'] = float(dist.central_value)
			values[obs]['exp']['uncert'] = float((dist.error_left + dist.error_right)/2)
		else:
			values[str(obs)]['SM'] = dict()
			values[str(obs)]['exp'] = dict()
			values[str(obs)]['NP'] = dict()
			values[str(obs)]['NP']['central'] = float(flavio.np_prediction(obs[0], w, obs[1], obs[2]))
			uncert.append(flavio.np_uncertainty(obs[0], w, obs[1], obs[2]))
			values[str(obs)]['SM']['central'] = float(flavio.sm_prediction(obs[0], obs[1], obs[2]))
			values[str(obs)]['SM']['uncert'] = float(flavio.np_uncertainty(obs[0], w, obs[1], obs[2]))
			dist = flavio.combine_measurements(obs)
			values[str(obs)]['exp']['central'] = float(dist.central_value)
			values[str(obs)]['exp']['uncert'] = float((dist.error_left + dist.error_right)/2)

	MS = MontecarloScan(likelihood_global, minx, maxx, num, bf, 0.1, wfun)
	if cores==1:
		MS.run(wfun)
		for obsnum, obs in enumerate(obslist):
			var = MS.expectedvalue(_variance, obs, wfun, values[str(obs)]['NP']['central'])
			values[str(obs)]['NP']['uncert'] = sqrt(uncert[obsnum]**2 + var )
	else:
		MS.run_mp(cores, wfun)		
		for obsnum, obs in enumerate(obslist):
			var = MS.expectedvalue_mp(_variance, cores, obs, wfun, values[str(obs)]['NP']['central'])
			values[str(obs)]['NP']['uncert'] = sqrt(uncert[obsnum]**2 + var )

	with open(fout, 'wt') as f:
		yaml.dump(values, f)
