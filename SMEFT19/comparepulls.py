'''
================
comparepulls
================

This module contains several functions used to compare between different NP scenarios and the Standard Model.
'''

from .ellipse import load, parametrize
from . import SMEFTglob
from .SMEFTglob import loadobslist
import flavio
import re
import numpy as np
from flavio.statistics.functions import delta_chi2

def sign(x, y):
	if float(x) < float(y):
		return -1
	else:
		return 1

def tex(obs):
	if isinstance(obs, str):
		text = flavio.Observable[obs].tex
	else:
		text = flavio.Observable[obs[0]].tex[:-1] + '^{' + str(list(obs[1:])).replace(',', ',\\ ') + '}$'
	return text.replace('text', 'mathrm')

def texnumber(x, prec=3):
	texn = ('{:.'+str(prec)+'g}').format(float(x))
	match = re.match(r'(-?[0-9]+(\.[0-9]+)?)e(.[0-9]+)', texn)
	if match:
		texn = '$' + match.group(1) + '\\times 10^{' + str(int(match.group(3))) + '}$'
	return texn


def compare(wfun, fin, fout):
	r'''
Lists the comparison between the pull of each observable in the NP hypothesis and the SM.

:Arguments:
	- wfun\: Function that takes a point in parameter space and returns a dictionary of Wilson coefficents.
	- fin\: Path to the file `.yaml` where the ellipsoid is saved.
	- fout\: Path to the `.tex` file where the comaparison table will be written. The observables are ordered by their SM pull, and are shaded in green if the NP improves this pull and in red otherwise.
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

	f = open(fout+'.tex', 'wt')
	obsnum = 0
	f.write('\\begin{longtable}{|c|c|c|c|c|}\\hline\n & Observable &\t NP prediction &\t NP pull & SM pull\\endhead\\hline\n')
	for obs in obscoll:
		NPpull = float(obsNP.loc[[obs], 'pull exp.'])
		SMpull = float(obsSM.loc[[obs], 'pull exp.'])
		if NPpull > SMpull:
			col = int(min(50, 50*(NPpull-SMpull)))
			f.write('{} &\t {} &\t {} &\t {} $ \\sigma$ &\t {} $ \\sigma$ \\\\ \\hline\n'.format(obsnum, tex(obs), texnumber(obsNP.loc[[obs], 'theory'], 5), r'\cellcolor{red!' + str(col) +'} ' + texnumber(NPpull, 2), texnumber(SMpull, 2)) )
		elif SMpull > NPpull:
			col = int(min(50, 50*(SMpull-NPpull)))
			f.write('{} &\t {} &\t {} &\t {} $ \\sigma$ &\t {} $ \\sigma$ \\\\ \\hline\n'.format(obsnum, tex(obs), texnumber(obsNP.loc[[obs], 'theory'], 5), r'\cellcolor{green!' + str(col) +'} ' + texnumber(NPpull, 2), texnumber(SMpull, 2)) )
		else:
			f.write('{} &\t {} &\t {} &\t {} $ \\sigma$ &\t {} $ \\sigma$ \\\\ \\hline\n'.format(obsnum, tex(obs), texnumber(obsNP.loc[[obs], 'theory'], 5), texnumber(NPpull, 2), texnumber(SMpull, 2)) )
		obsnum += 1
	f.write('\\end{longtable}')
	f.close()


def pointpull(x, wfun, bf, printlevel=1, numres=5):
	r'''
Determines the observable whose pull changes the most between two NP hypothesis.

:Arguments:
	- x\: Point in space parameter of the tested NP hypothesis.
	- wfun\: Function that takes a point in parameter space and returns a dictionary of Wilson coefficents.
	- bf\: Point in space parameter of the reference NP hypothesis (e.g. the best fit).
	- [printlevel\: 0 for silent mode, 1 for verbose mode.]
	- [numres\: Number of observables displayed. Default=5.]

:Returns:
	- A multi-line string. Each line contains the id number of the observable, its name and the squared difference of the pulls.
	'''
	w = wfun(bf)
	wx = wfun(x)
	gl = SMEFTglob.gl
	glNP = gl.parameter_point(w)
	glx = gl.parameter_point(wx)
	obsNP = glNP.obstable()
	obsx = glx.obstable()
	obscoll = loadobslist()
	dicpull = dict()
	i = 0
	for obs in obscoll:
		pull0 = float(obsNP.loc[[obs], 'pull exp.'])*sign(obsNP.loc[[obs], 'theory'], obsNP.loc[[obs], 'experiment'] )
		pullx = float(obsx.loc[[obs], 'pull exp.'])*sign(obsx.loc[[obs], 'theory'], obsx.loc[[obs], 'experiment'] )
		dicpull[i] = (pullx-pull0)**2
		i += 1
	sortdict = sorted(dicpull, key=dicpull.get, reverse=True)[0:numres]
	results = ''
	for obs in sortdict:
		results += str(obs) + '\t' + str(obscoll[obs]) + '\t' + str(dicpull[obs]) + '\n'
	if printlevel:
		print(results)
	return results

def notablepulls(wfun, fin, fout):
	r'''
Determines the observables whose pull changes the most between the best fit and the notable points of the ellipsoid.

:Arguments:
	- wfun\: Function that takes a point in parameter space and returns a dictionary of Wilson coefficents.
 	- fin\: Path to the file `.yaml` where the ellipsoid is saved.
	- fout\: Path to the `.tex` file where the comaparison table will be written.
	'''
	f = open(fout, 'wt')
	dbf = load(fin)
	bf = dbf['bf']
	v = dbf['v']
	d = dbf['d']
	n = len(bf)
	p = delta_chi2(1, n)
	H = v @ d @ v.T
	for i in range(0,n):
		# Moving along operator axes
		dC = float(np.sqrt(p/H[i,i]))
		delta = np.zeros(n)
		delta[i] = dC
		f.write('Operator ' + str(i+1) + '+\n**********************\n')
		f.write(pointpull(bf + delta, wfun, bf, 0))
		f.write('\n\n')
		f.write('Operator ' + str(i+1) + '-\n**********************\n')
		f.write(pointpull(bf - delta, wfun, bf, 0))
		f.write('\n\n')
	for i in range(0,n):
		#Moving along ellipsoid axes
		delta = np.zeros(n)
		delta[i] = 1
		f.write('Axis ' + str(i+1) + '+\n**********************\n')
		f.write(pointpull(parametrize(delta, bf, v, d), wfun, bf, 0))
		f.write('\n\n')
		f.write('Axis ' + str(i+1) + '-\n**********************\n')
		f.write(pointpull(parametrize(-delta, bf, v, d), wfun, bf, 0))
		f.write('\n\n')
	bfm = np.matrix(bf)
	dSM = float(np.sqrt(p/(bfm @ H @ bfm.T ) ))
	f.write('SM+\n**********************\n')
	f.write(pointpull(bf*(1+dSM), wfun, bf, 0))
	f.write('\n\n')
	f.write('SM-\n**********************\n')
	f.write(pointpull(bf*(1-dSM), wfun, bf, 0))
	f.close()

def pullevolution(obscode, wfun, fin, direction):
	r'''
Calculates the variation of the pull along a line connecting two opposite notable points of the ellipsoid.

:Arguments:
	- obscode\: ID-Number of the observable, as returned by comparepulls.pointpull
	- wfun\: Function that takes a point in parameter space and returns a dictionary of Wilson coefficents.
 	- fin\: Path to the file .yaml where the ellipsoid is saved.
	- direction\: string with the following format\:

		- 'wc' + str(i)\: for the i-th Wilson coefficient.
		- 'ax' + str(i)\: for the i-th principal axis of the ellipsoid.
		- 'sm'\: for the direction joining the bf and sm points.
	'''
	dbf = load(fin)
	bf = dbf['bf']
	v = dbf['v']
	d = dbf['d']
	n = len(bf)
	p = delta_chi2(1, n)
	H = v @ d @ v.T
	pull_list = []
	obscoll = loadobslist()
	obs = obscoll[obscode]
	for c in np.linspace(-1, 1, 200):
		if direction[:2] == 'wc':
			i = int(direction[2:])-1
			dC = float(np.sqrt(p/H[i,i]))
			delta = np.zeros(n)
			delta[i] = dC
			point = bf + c * delta
		if direction[:2] == 'ax':
			i = int(direction[2:])-1
			delta = np.zeros(n)
			delta[i] = c
			point = parametrize(delta, bf, v, d)
		if direction[:2] == 'sm':
			bfm = np.matrix(bf)
			dSM = float(np.sqrt(p/(bfm @ H @ bfm.T ) ))
			point = bf*(1+c*dSM)
		pull_list.append(SMEFTglob.pull_obs(point, obs, wfun) )
	return pull_list
