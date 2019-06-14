from .ellipse import load, parametrize
from . import SMEFTglob
from .SMEFTglob import loadobslist
from wilson import Wilson
import flavio
import re
import numpy as np
import yaml
from flavio.statistics.functions import delta_chi2, pull

def sign(x, y):
	if float(x) < float(y):
		return -1
	else:
		return 1

def tex(obs):
	if isinstance(obs, str):
		text = flavio.Observable[obs].tex
	else:
		text = flavio.Observable[obs[0]].tex[:-1] + '^{[' + str(obs[1]) + ',\\ ' + str(obs[2]) + ']}$'
	return text.replace('text', 'mathrm')

def texnumber(x, prec=3):
	texn = ('{:.'+str(prec)+'g}').format(float(x))
	match = re.match(r'(-?[0-9]+(\.[0-9]+)?)e(.[0-9]+)', texn)
	if match:
		texn = '$' + match.group(1) + '\\times 10^{' + str(int(match.group(3))) + '}$'
	return texn

	
def compare(wfun, fin, fout):
	bf, v, d = load(fin)

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


def pointpull(x, wfun, fin, printlevel=1, numres=5):
	bf, v, d = load(fin)
	w = wfun(bf)
	wx = wfun(x)
	gl = SMEFTglob.gl
	glNP = gl.parameter_point(w)
	glx = gl.parameter_point(wx)
	obsSM = gl.obstable_sm
	obsNP = glNP.obstable()
	obsx = glx.obstable()
	obscoll = loadobslist()
	dicpull = dict()
	i = 0
	for obs in obscoll:
		pull = float(obsNP.loc[[obs], 'pull exp.'])*sign(obsNP.loc[[obs], 'theory'], obsNP.loc[[obs], 'experiment'] )
		pullx = float(obsx.loc[[obs], 'pull exp.'])*sign(obsx.loc[[obs], 'theory'], obsx.loc[[obs], 'experiment'] )
		dicpull[i] = (pullx-pull)**2
		i += 1
	sortdict = sorted(dicpull, key=dicpull.get, reverse=True)[0:numres]
	results = ''
	for obs in sortdict:
		results += str(obs) + '\t' + str(obscoll[obs]) + '\t' + str(dicpull[obs]) + '\n'
	if printlevel:
		print(results)
	return results

def notablepulls(wfun, fin, fout):
	f = open(fout, 'wt')
	bf, v, d = load(fin)
	n = len(bf)
	p = delta_chi2(1, n)
	a = np.sqrt(p/np.diag(d))
	H = v @ d @ v.T
	for i in range(0,n):
		# Moving along operator axes
		dC = float(np.sqrt(p/H[i,i]))
		delta = np.zeros(n)
		delta[i] = dC
		f.write('Operator ' + str(i+1) + '+\n**********************\n')
		f.write(pointpull(bf + delta, wfun, fin, 0))
		f.write('\n\n')
		f.write('Operator ' + str(i+1) + '-\n**********************\n')
		f.write(pointpull(bf - delta, wfun, fin, 0))
		f.write('\n\n')
	for i in range(0,n):
		#Moving along ellipsoid axes
		delta = np.zeros(n)
		delta[i] = 1
		f.write('Axis ' + str(i+1) + '+\n**********************\n')
		f.write(pointpull(parametrize(delta, bf, v, d), wfun, fin, 0))
		f.write('\n\n')
		f.write('Axis ' + str(i+1) + '-\n**********************\n')
		f.write(pointpull(parametrize(-delta, bf, v, d), wfun, fin, 0))
		f.write('\n\n')
	bfm = np.matrix(bf)
	dSM = float(np.sqrt(p/(bfm @ H @ bfm.T ) ))
	f.write('SM+\n**********************\n')
	f.write(pointpull(bf*(1+dSM), wfun, fin, 0))
	f.write('\n\n')
	f.write('SM-\n**********************\n')
	f.write(pointpull(bf*(1-dSM), wfun, fin, 0))
	f.close()

def pullevolution(obscode, wfun, fin, direction):
	'''
	direction: string with the following format:
		'wc' + str(i): for the i-th Wilson coefficient
		'ax' + str(i): for the i-th principal axis of the ellipsoid
		'sm': for the direction joining the bf and sm points 
	'''
	bf, v, d = load(fin)	
	n = len(bf)
	p = delta_chi2(1, n)
	a = np.sqrt(2*p/np.diag(d))
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
		pull_list.append(SMEFTglob.pull_obs(obs, point, wfun) )
	return pull_list
