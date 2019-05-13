import texfig # https://github.com/knly/texfig
import yaml
import matplotlib.pyplot as plt
import numpy as np

def plot(fin, fout, obstype):
	f = open(fin, 'rt')
	values = yaml.load(f)
	f.close()
	
	if obstype == 'RD':
		observables = ['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'Rtaumu(B->D*lnu)']
		texlabels = [r'$R_D^\ell$', r'$R_{D^*}^\ell$', r'$R_{D^*}^\mu$']
		legloc = 1
	elif obstype == 'RK':
		observables = [('<Rmue>(B+->Kll)', 1.0, 6.0), ('<Rmue>(B0->K*ll)', 0.045, 1.1), ('<Rmue>(B0->K*ll)', 1.1, 6.0)]
		texlabels = [r'$R_K^{[1,6]}$', r'$R_{K^*}^{[0.045, 1.1]}$', r'$R_{K^*}^{[1.1, 6]}$']
		legloc = 3
	elif obstype == 'RZ':
		observables = ['R_e', 'R_mu', 'R_tau']
		texlabels = [r'$R_e$', r'$R_\mu$', r'$R_\tau$']
		legloc = 1

	x = []
	y = []
	erry = []
	z = [0, 0.99999, 0.5]*len(observables)

	obsnum = 0
	for obs in observables:
		x.append(obsnum)
		y.append(values[str(obs)]['NP']['central'])
		erry.append(values[str(obs)]['NP']['uncert'])
		x.append(obsnum+0.2)
		y.append(values[str(obs)]['SM']['central'])
		erry.append(values[str(obs)]['SM']['uncert'])
		x.append(obsnum+0.4)
		y.append(values[str(obs)]['exp']['central'])
		erry.append(values[str(obs)]['exp']['uncert'])
		obsnum += 1

	plt.figure()
	cm=plt.get_cmap('brg')
	plt.scatter(x, y, c=z, s=15, cmap=cm, zorder=10)
	for i, (xval, yval, y_error_val, zval) in enumerate(zip(x, y, erry, z)):
		plt.errorbar(xval, yval, yerr=y_error_val, linestyle='', c=cm(zval))
	
	if obstype == 'RK':
		plt.ylim([0.5, 1.05])
	plt.xticks(np.arange(0.2, 0.2+len(observables), 1), texlabels)
	plt.tick_params(axis='x', length=0)
	npArtist = plt.Line2D((0,1),(0,0), color = cm(z[0]) )
	smArtist = plt.Line2D((0,1),(0,0), color = cm(z[1]))
	expArtist = plt.Line2D((0,1),(0,0), color = cm(z[2]))
	plt.legend([npArtist, smArtist, expArtist], ['New Physics', 'Standard Model', 'Measurement'], loc=legloc)

	texfig.savefig(fout)
