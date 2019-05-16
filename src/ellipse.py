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
bf, v, d, Lmin = minimum(fit, x0)

Arguments
	- fit: function that takes one point in parameter space and returns its negative log-likelihhod
		example: -SMEFTglob.likelihood_global(x, scenarios.scVI)
	- x0: list or np.array containing an initial guess

Return:
	- bf: np.array with the point in parameter space with the best fit
	- v: Unitary matrix containing the axes of the ellipse
	- d: diagonal matrix containing the inverse of the squares of the semiaxes
	- Lmin: value of fit(bf)
	'''
	global bf
	global Lmin
	global dim
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
	v, d, vt = svd(m.np_matrix())
	d_ellipse = np.diag(1/d)
	return bf, v, d_ellipse, Lmin
	

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
	r = 2*delta_chi2(nsigmas, len(bf))
	xp = x * sqrt(r/np.diag(d))
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
	a = np.sqrt(2*p/np.diag(d))
	bestchi2 = fit(bf)
	H = v @ d @ v.T
	cross_p = []
	cross_m = []
	chi2_cross_p = []
	chi2_cross_m = []
	for i in range(0,n):
		# Moving along operator axes
		dC = float(np.sqrt(2*p/H[i,i]))
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
		#Moving along ellipsoid axes
		delta[i] = 1
		point_p = parametrize(delta, bf, v, d)
		point_m = parametrize(-delta, bf, v, d)
		chi2_p = fit(point_p) - bestchi2
		chi2_m = fit(point_m) - bestchi2
		ex_p.append(point_p)
		ex_m.append(point_m)
		chi2_ex_p.append(chi2_p)
		chi2_ex_m.append(chi2_m)
	bfm = np.matrix(bf)
	dSM = float(np.sqrt(2*p/(bfm @ H @ bfm.T ) ))
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
