import texfig
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def errorplot(flist, plottype, fout):
	fig = texfig.figure()
	if plottype == 'RD':
		observables = ['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)']
		texlabels = [r'$R_D^\ell$', r'$R_{D^*}^\ell$', r'$R_{D^*}^\mu$']
		#legloc = 1
	elif plottype == 'RK':
		observables = [('<Rmue>(B+->Kll)', 1.1, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0)]
		texlabels = [r'$R_K^{[1.1,6]}$', r'$R_{K^*}^{[0.045, 1.1]}$', r'$R_{K^*}^{[1.1, 6]}$']
		#legloc = 3
	nobs = len(texlabels)
	nhyp = len(flist)
	ax=plt.gca()
	plt.xlim([0, nobs])
	#plt.ylim([-0.055, 0.015])
	markers = ['o', '^', 's', '*', 'D']

	data = np.zeros([nhyp, nobs,2])
	smdata = np.zeros([nobs,2])
	expdata = np.zeros([nobs,2])
	leglabels = []
	hyp = 0
	for fin in flist:
		f = open(fin, 'rt')
		values = yaml.load(f)
		f.close()
		try:
			leglabels.append(values['name'])
		except:
			leglabels.append(fin[:-5])
		
		o = 0
		for obs in observables:
			data[hyp][o][0] = values[str(obs)]['NP']['central']
			data[hyp][o][1] = values[str(obs)]['NP']['uncert']
			smdata[o][0] = values[str(obs)]['SM']['central']
			smdata[o][1] = values[str(obs)]['SM']['uncert']
			expdata[o][0] = values[str(obs)]['exp']['central']
			expdata[o][1] = values[str(obs)]['exp']['uncert']
			o += 1
		hyp += 1
		
	for o in range(0, nobs):
		for i in range(0, nhyp):
			if o==0:
				plt.plot(o+(i+1)/(nhyp+1), data[i][o][0], marker=markers[i], color='b', label=leglabels[i])
			else:
				plt.plot(o+(i+1)/(nhyp+1), data[i][o][0], marker=markers[i], color='b')
			plt.errorbar(o+(i+1)/(nhyp+1), data[i][o][0], yerr=data[i][o][1], color='b')
			
		if o==0:
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7, label='SM'))		
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7, label='Experimental'))
		else:
			ax.add_patch(Rectangle( (o, expdata[o][0]-expdata[o][1]), 1, 2*expdata[o][1], color='green', alpha=0.7))
			ax.add_patch(Rectangle( (o, smdata[o][0]-smdata[o][1]), 1, 2*smdata[o][1], color='orange', alpha=0.7))		
			
		
	ax.set_xticks(np.linspace(0.5, nobs-0.5, nobs) )
	ax.set_xticklabels(texlabels + [''])
	plt.legend()
	texfig.savefig(fout)
