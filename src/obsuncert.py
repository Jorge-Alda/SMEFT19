import ellipse
from wilson import Wilson
import flavio
from math import sqrt
import numpy as np
import yaml


obslist = [('<Rmue>(B+->Kll)', 1.1, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0), 'Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)']

def distrsphere(dim):
	vect = np.random.randn(dim)
	return vect/np.linalg.norm(vect)

def calculate(wfun, fin, fout, num=1000):
	bf, v, d = ellipse.load(fin)

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

	for i in range(0, num):
		x = distrsphere(len(bf))
		xe = ellipse.parametrize(x, bf, v, d)
		w = wfun(xe)
		obsnum = 0
		for obs in obslist:
			if isinstance(obs, str):
				val = flavio.np_prediction(obs, w)
			else:
				val = flavio.np_prediction(obs[0], w, obs[1], obs[2])
			minval[obsnum] = min(minval[obsnum], val)
			maxval[obsnum] = max(maxval[obsnum], val)
			obsnum += 1

	obsnum = 0
	for obs in obslist:
		values[str(obs)]['NP']['uncert'] = sqrt(uncert[obsnum]**2 + (maxval[obsnum] - minval[obsnum])**2/4 )
		obsnum += 1

	f = open(fout, 'wt')
	yaml.dump(values, f)
	f.close()
