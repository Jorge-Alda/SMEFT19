import texfig # https://github.com/knly/texfig
from ellipse import load
import smelli
from wilson import Wilson
import flavio
import matplotlib.pyplot as plt
import re
import numpy as np

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
	gl = smelli.GlobalLikelihood()
	glSM = gl.parameter_point({}, scale=1e3)
	glNP = gl.parameter_point(w)
	obsSM = glSM.obstable()
	obsNP = glNP.obstable()
	obscoll = obsSM['pull'].keys()
	
	#TeX table
	f = open(fout+'.tex', 'wt')
	obsnum = 0
	f.write('\\begin{longtable}{|c|c|c|c|c|}\\hline\n & Observable &\t NP prediction &\t NP pull & SM pull\\endhead\\hline\n')
	for obs in obscoll:
		f.write('{} &\t {} &\t {} &\t {} $ \\sigma$ &\t {} $ \\sigma$ \\\\ \\hline\n'.format(obsnum, tex(obs), texnumber(obsNP.loc[[obs], 'theory'], 5), texnumber(obsNP.loc[[obs], 'pull'], 2), texnumber(obsSM.loc[[obs], 'pull'], 2)) )
		obsnum += 1
	f.write('\\end{longtable}')
	f.close()

	#Plots
	NP = []
	SM = []
	for obs in obscoll:
		NP.append(float(obsNP.loc[[obs], 'pull']))
		SM.append(float(obsSM.loc[[obs], 'pull']))
			
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
	
	texfig.savefig(fout)

def pointpull(x, wfun, fin):
	from ellipse import parametrize
	bf, v, d = load(fin)
	w = wfun(bf)
	wx = wfun(x)
	gl = smelli.GlobalLikelihood()
	glSM = gl.parameter_point({}, scale=1e3)
	glNP = gl.parameter_point(w)
	glx = gl.parameter_point(wx)
	obsSM = glSM.obstable()
	obsNP = glNP.obstable()
	obsx = glx.obstable()
	obscoll = obsSM['pull'].keys()
	dicpull = dict()
	for obs in obscoll:
		pull = float(obsNP.loc[[obs], 'pull'])*sign(obsNP.loc[[obs], 'theory'], obsNP.loc[[obs], 'experiment'] )
		pullx = float(obsx.loc[[obs], 'pull'])*sign(obsx.loc[[obs], 'theory'], obsx.loc[[obs], 'experiment'] )
		dicpull[str(obs)] = (pullx-pull)**2
	sortdict = sorted(dicpull, key=dicpull.get, reverse=True)[0:5]
	for obs in sortdict:
		print(obs + '\t' + str(dicpull[obs]))
