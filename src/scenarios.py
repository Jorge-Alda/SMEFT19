from wilson import Wilson
import numpy as np

def scI(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scII(x):
	return Wilson({'lq1_2233': x[0]*1e-6, 'lq3_2233': x[0]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scIII(x):
	return Wilson({'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scIV(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[1]*1e-6, 'lq3_2233': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scV(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_3333': x[1]*1e-6, 'lq3_3333': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scVI(x):
	return Wilson({'lq1_2233': x[0]*1e-6, 'lq3_2233': x[0]*1e-6, 'lq1_3333': x[1]*1e-6, 'lq3_3333': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scVII(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[1]*1e-6, 'lq3_2233': x[1]*1e-6, 'lq1_3333': x[2]*1e-6, 'lq3_3333': x[2]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scVIII(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[0]*1e-6, 'lq3_2233': x[0]*1e-6, 'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scIX(x):
	return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': -x[0]*1e-6, 'lq3_2233': -x[0]*1e-6, 'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def idemp(a,b):
	return np.matrix([[a**2, a*b, a], [a*b, b**2, b], [a, b, 1]])/(1+ a**2 + b**2)

def matrixwc(num, C, ll, lq):
	wc = dict()  
	for il in range(0,3):
		for jl in range(0, il+1):
			for iq in range(0, 3):
				if il == jl:
					jmax = iq+1
				else:
					jmax = 3
				for jq in range(0, jmax):
					w = C * ll[jl, il] * lq[jq, iq] 
					if abs(w) > 1e-12:
						wcname = 'lq' + num + '_' + str(jl+1) + str(il+1) + str(jq+1) + str(iq+1)
						wc[wcname] = w*1e-6
	return wc

def massrotation(x):
	C1 = x[0]
	C3 = x[1]
	al = x[2]
	bl = x[3]
	ll = idemp(al, bl)
	aq = x[4]
	bq = x[5]
	lq = idemp(aq, bq)
	w1 = matrixwc('1', C1, ll, lq)
	w3 = matrixwc('3', C3, ll, lq)
	return Wilson({**w1,**w3}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scbI(x):
	a = x[0]/x[2]
	b = x[1]/x[2]
	C = x[2]*(1+a**2+b**2)
	return massrotation([C, C, a, b, 0, 0])

def scbII(x):
	# x = [C1333, C2333, C3313, C3323, C3333]
	al = x[0]/x[4]
	bl = x[1]/x[4]
	aq = x[2]/x[4]
	bq = x[3]/x[4]
	C = x[4]*(1+al**2+bl**2)*(1+aq**2+bq**2)
	return massrotation([C, C, al, bl, aq, bq])

def Feruglio(x):
	bl = x[0]/x[2]
	bq = x[1]/x[2]
	C = x[2]*(1+bl**2)*(1+bq**2)
	return massrotation([C, C, 0, bl, 0, bq])
