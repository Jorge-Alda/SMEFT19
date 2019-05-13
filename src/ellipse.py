'''
Assuming that the likelihood of the fit follows a gaussian distribution (Central Limit Theorem), and therefore the log-likelihood is characterized by a quadratic form around the minimum, this script finds this quadratic form, and parametrizes (ellipsoidal) sections of constant likelihood.

'''


import numpy as np
from numpy.linalg import svd
from numpy import sqrt
import yaml
from flavio.statistics.functions import pull, delta_chi2
from iminuit import Minuit

def roundsig(x, num=4):
	l = int(np.log10(abs(x)))
	return round(x, -l+num)

def minimum(fit, x0):
		
	'''
bf, Lmin = minimum(fit, x0)

Arguments
	- fit: function that takes one point in parameter space and returns its log-likelihhod
		example: -2*SMEFTglob.likelihood(x, 'global')
	- x0: list or np.array containing an initial guess

Return:
	- bf: np.array with the point in parameter space with the best fit
	- Lmin: value of fit(bf)
	'''
	global bf
	global Lmin
	global dim
	print('Minimizing...')
	m = Minuit.from_array_func(fit, x=x0, errordef=0.5, print_level=0)
	bf = m.values[0]
	Lmin = m.get_fmin()
	dim = len(x0)
	L0 = fit(np.zeros(dim))
	p = pull(abs(Lmin-L0), dim)
	print('Pull: ' + str(p) + ' sigma')
	return bf, Lmin
	
def ellipseform(bf, fit, step=1e-3):
	'''
v, d = ellipseform(bf, fit, step=1e-3)
		
Arguments:
	- bf: np.array with the point in parameter space with the best fit
	- fit: function that takes one point in parameter space and returns its log-likelihhod
		example: -2*SMEFTglob.likelihood(x, 'global')
	- step (optional): distance from the best fit used to compute the quadratic form

Returns:
	- v: np.matrix containing the orientation of the axes of the ellipsoid
	- d: np.array containing the principal axes of the ellipsoid
	'''
	dim = len(bf)
	Lmin = fit(bf)
	Q = np.matrix(np.zeros([dim, dim]))
	for i in range(0, dim):
		delta = np.zeros(dim)
		delta[i] = step
		Q1 = fit(bf+delta)-Lmin
		Q2 = fit(bf-delta)-Lmin
		Q[i,i] = 0.5*(Q1+Q2)/step**2
	for i in range(0, dim):
		for j in range(0, i):
			delta = np.zeros(dim)
			delta[i] = delta[j] = step
			Q1 = fit(bf+delta)-Lmin
			Q2 = fit(bf-delta)-Lmin
			Q[i,j] = Q[j,i] = (Q1+Q2)/(4*step**2) - Q[i,i]/2 - Q[j,j]/2
	v, d, vt = svd(Q)
	return v, d

def parametrize(x, bf, v, d, nsigmas=1):
	'''
xe = parametrize(x, bf, v, d, nsigmas=1)

Arguments:
	- x: np.array containing a point in the surface of the unit n-hypersphere
	- bf: np.array with the point in parameter space with the best fit
	- v: np.matrix containing the orientation of the axes of the ellipsoid
	- d: np.array containing the principal axes of the ellipsoid
	- nsigmas (optional): significance of the isoprobability hypersurface wrt the best fit

Returns:
	- xe: Projection of the point xe in the ellipsoid of equal probability
	'''
	from flavio.statistics.functions import delta_chi2
	deltaL = delta_chi2(nsigmas, len(x))
	xp = x * sqrt(deltaL/d)
	xe = bf + v @ xp
	return np.array(xe).flatten()

def save(bf, v, d, filename):
	'''
save(bf, v, d, filename)

Arguments:
	- bf: np.array with the point in parameter space with the best fit
	- v: np.matrix containing the orientation of the axes of the ellipsoid
	- d: np.array containing the principal axes of the ellipsoid
	- filename: Path to the YAML file where the shape of the ellipse will be saved
	'''
	f = open(filename, 'wt')
	values = dict()
	values['bf'] = bf.tolist()
	values['v'] = v.tolist()
	values['d'] = d.tolist()
	yaml.dump(values, f)
	f.close()
	
def load(filename):
	'''
bf, v, d = load(filename)

Arguments:
	- filename: Path to the YAML file where the shape of the ellipse has been saved by the "save" method
		WARNING: this method doesn't check the integrity of the file

Returns:
	- bf: np.array with the point in parameter space with the best fit
	- v: np.matrix containing the orientation of the axes of the ellipsoid
	- d: np.array containing the principal axes of the ellipsoid	
	'''
	f = open(filename, 'rt')
	values = yaml.load(f)
	f.close()
	bf = np.array(values['bf'])
	v = np.matrix(values['v'])
	d = np.array(values['d'])
	return bf, v, d

def notablepoints(fin, fout, fit):
	bf, v, d = load(fin)
	n = len(bf)
	p = delta_chi2(1, n)
	ex_p = []
	ex_m = []
	chi2_ex_p = []
	chi2_ex_m = []
	a = np.sqrt(p/d)
	bestchi2 = fit(bf)
	for i in range(0,n):
		loop = True
		while loop:
			ex_p0 = np.array(bf + v[i,:] * a[i]).flatten()
			ex_m0 = np.array(bf - v[i,:] * a[i]).flatten()
			chi2_p0 = fit(ex_p0) - bestchi2
			chi2_m0 = fit(ex_m0) - bestchi2
			chi2_mean = (chi2_p0 + chi2_m0)/2
			if abs(chi2_mean/p - 1) > 0.1:
				d[i] = d[i] * chi2_mean/p
				a = np.sqrt(p/d)
			else:
				loop = False
				ex_p.append(ex_p0)
				ex_m.append(ex_m0)
				chi2_ex_p.append(chi2_p0)
				chi2_ex_m.append(chi2_m0)
	save(bf, v, d, fin.replace('.yaml', '_c.yaml'))
	H = v @ np.diag(d) @ v.T
	cross_p = []
	cross_m = []
	chi2_cross_p = []
	chi2_cross_m = []
	for i in range(0,n):
		dC = float(np.sqrt(p/H[i,i]))
		delta = np.zeros(n)
		delta[i] = dC
		point_p = bf + delta
		point_m = bf - delta
		chi2_p = fit(point_p) - bestchi2
		chi2_m = fit(point_m) - bestchi2
		cross_p.append(point_p)
		cross_m.append(point_m)
		chi2_cross_p.append(chi2_p)
		chi2_cross_m.append(chi2_m)
	bfm = np.matrix(bf)
	dSM = float(np.sqrt(p/(bfm @ H @ bfm.T ) ))
	SM_p = bf*(1+dSM)
	SM_m = bf*(1-dSM)
	chi2_SM_p = fit(SM_p) - bestchi2
	chi2_SM_m = fit(SM_m) - bestchi2
	from comparepulls import texnumber
	f = open(fout, 'w')
	f.write(r'\begin{tabular}{|' + 'c|'*(n+3) + r'}\\\hline' + '\n'  )
	f.write(r'$j$ & $s$ & ' + ' & '*n + r'$\Delta \chi^2$\\\hline' + '\n' )
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
		f.write( ' & $+$ & ')
		for j in range(0, n):
			f.write(texnumber(cross_p[i][j]) + ' & ')
		f.write(texnumber(chi2_cross_p[i]) + r'\\\hline' + '\n') 
		f.write( ' & $-$ & ')
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
	f.close()

def findellipse(vect, fin, fit):
	bf, v, d = load(fin)
	dim = len(bf)
	delta = delta_chi2(1, dim)
	bestchi2 = fit(bf)
	loop = True
	while loop:
		chi2 = fit(bf + vect) - bestchi2
		if abs(chi2/delta -1) > 0.1:
			vect = vect * np.sqrt(delta/chi2)
		else:
			loop = False
			print(bf + vect)
