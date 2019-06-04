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
	return np.matrix([[abs(a)**2, a*np.conj(b), a], [np.conj(a)*b, abs(b)**2, b], [np.conj(a), np.conj(b), 1]])/(1+ abs(a)**2 + abs(b)**2)

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

def rotBI(x):
	#x = [C, beta_l, beta_q])
	return massrotation([x[0], x[0], 0, x[1], 0, x[2]])

def rotBII(x):
	#x = [C, alpha_l, beta_l, alpha_q, beta_q])
	return massrotation([x[0], x[0], x[1], x[2], x[3], x[4]])

def rot2lqU1(x, M=1.5):
	if len(x)==3:
		al = 0
		aq = 0
		bl = x[1]
		bq = x[2]
	elif len(x)==5:
		al = x[1]
		bl = x[2]
		aq = x[3]
		bq = x[4]
	C = x[0]
	ll = idemp(al, bl)
	lq = idemp(aq, bq)
	xL = np.matrix(np.zeros([3,3]))
	for i in range(0,3):
		for j in range(0,3):
			xL[i,j] = np.sqrt(-2*M**2*C * ll[i,i] * lq[j,j])*np.exp(1j*np.angle(lq[j,2])-1j*np.angle(ll[i,2]))
	return xL
