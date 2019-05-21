from ellipse import load, parametrize
import SMEFTglob
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
	obsSM = gl.obstable_sm
	obsNP = glNP.obstable()
	try:
		fyaml = open('observables.yaml', 'rt')
		obscoll = yaml.load(fyaml)
		fyaml.close()
	except:	
		obscoll = list(obsSM['pull exp.'].keys())
		fyaml = open('observables.yaml', 'wt')
		yaml.dump(obscoll, fyaml)
		fyaml.close()
	
	#TeX table
	f = open(fout+'.tex', 'wt')
	obsnum = 0
	f.write('\\begin{longtable}{|c|c|c|c|c|}\\hline\n & Observable &\t NP prediction &\t NP pull & SM pull\\endhead\\hline\n')
	for obs in obscoll:
		f.write('{} &\t {} &\t {} &\t {} $ \\sigma$ &\t {} $ \\sigma$ \\\\ \\hline\n'.format(obsnum, tex(obs), texnumber(obsNP.loc[[obs], 'theory'], 5), texnumber(obsNP.loc[[obs], 'pull exp.'], 2), texnumber(obsSM.loc[[obs], 'pull exp.'], 2)) )
		obsnum += 1
	f.write('\\end{longtable}')
	f.close()

	#Plots
	import texfig # https://github.com/knly/texfig
	import matplotlib.pyplot as plt
	NP = []
	SM = []
	for obs in obscoll:
		NP.append(float(obsNP.loc[[obs], 'pull exp.']))
		SM.append(float(obsSM.loc[[obs], 'pull exp.']))
			
	plt.figure()
	plt.plot(NP, label='New Physics')
	plt.plot(SM, label='Standard Model')
	vertplus = 0
	vertminus = 0
	for i in range(0, len(SM)):
		if (NP[i]-SM[i]) > 1:
			v = 0.3 + vertplus
			vertplus += 0.1		
			plt.annotate(str(i), xy=(i, NP[i]), xytext=(i, NP[i]+v), fontsize=6, horizontalalignment='right', arrowprops = dict(facecolor = 'black',  arrowstyle='->') )
		elif (SM[i]-NP[i]) > 1:
			v = 0.3 + vertminus
			#vertminus += 0.1		
			plt.annotate(str(i), xy=(i, NP[i]), xytext=(i, NP[i]-v), fontsize=6, horizontalalignment='left', arrowprops = dict(facecolor = 'black',  arrowstyle='->') )
	plt.xlabel('Observable')
	plt.ylabel(r'$|$Pull$|$')
	plt.legend(loc=1)
	plt.tight_layout(pad=0.5)
	texfig.savefig(fout)

def pointpull(x, wfun, fin, printlevel=1, numres=5):
	bf, v, d = load(fin)
	w = wfun(bf)
	wx = wfun(x)
	gl = SMEFTglob
	glNP = gl.parameter_point(w)
	glx = gl.parameter_point(wx)
	obsSM = gl.obstable_sm
	obsNP = glNP.obstable()
	obsx = glx.obstable()
	try:
		fyaml = open('observables.yaml', 'rt')
		obscoll = yaml.load(fyaml)
		fyaml.close()
	except:	
		obscoll = list(obsSM['pull exp.'].keys())
		fyaml = open('observables.yaml', 'wt')
		yaml.dump(obscoll, fyaml)
		fyaml.close()
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
	a = np.sqrt(2*p/np.diag(d))
	H = v @ d @ v.T
	for i in range(0,n):
		# Moving along operator axes
		dC = float(np.sqrt(2*p/H[i,i]))
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
	dSM = float(np.sqrt(2*p/(bfm @ H @ bfm.T ) ))
	f.write('SM+\n**********************\n')
	f.write(pointpull(bf*(1+dSM), wfun, fin, 0))
	f.write('\n\n')
	f.write('SM-\n**********************\n')
	f.write(pointpull(bf*(1-dSM), wfun, fin, 0))
	f.close()
