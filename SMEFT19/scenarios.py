'''
=================
scenarios
=================

This module contains all the NP hypothesis that we have considered
and some auxiliar functions to implement them.
'''

from wilson import Wilson
import numpy as np

def scI(x):
    r'''
Scenario I\: NP only affects to electrons.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scII(x):
    r'''
Scenario II\: NP only affects to muons.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_2233': x[0]*1e-6, 'lq3_2233': x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scIII(x):
    r'''
Scenario III\: NP only affects to taus.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scIV(x):
    r'''
Scenario IV\: NP only affects to electrons and muons.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[1]*1e-6,
                   'lq3_2233': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scV(x):
    r'''
Scenario V\: NP only affects to electrons and taus.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_3333': x[1]*1e-6,
                   'lq3_3333': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scVI(x):
    r'''
Scenario VI\: NP only affects to muons and taus.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_2233': x[0]*1e-6, 'lq3_2233': x[0]*1e-6, 'lq1_3333': x[1]*1e-6,
                   'lq3_3333': x[1]*1e-6}, eft='SMEFT', basis='Warsaw', scale=1e3)

def scVII(x):
    r'''
Scenario VII\: NP affects to electrons, muons and taus.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[1]*1e-6,
                   'lq3_2233': x[1]*1e-6, 'lq1_3333': x[2]*1e-6, 'lq3_3333': x[2]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scVIII(x):
    r'''
Scenario VIII\: NP affects to electrons, muons and taus equally.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': x[0]*1e-6,
                   'lq3_2233': x[0]*1e-6, 'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scIX(x):
    r'''
Scenario IX\: NP affects to electrons and taus equally, and to muons by an opposite ammount.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': -x[0]*1e-6,
                   'lq3_2233': -x[0]*1e-6, 'lq1_3333': x[0]*1e-6, 'lq3_3333': x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scX(x):
    r'''
Scenario X\: NP affects to electrons and muons by an opposite ammount.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6,
                   'lq1_2233': -x[0]*1e-6, 'lq3_2233': -x[0]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def scXI(x):
    r'''
Scenario XI\: NP affects to electrons and to muons by an opposite ammount,
              and to taus independently.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return Wilson({'lq1_1133': x[0]*1e-6, 'lq3_1133': x[0]*1e-6, 'lq1_2233': -x[0]*1e-6,
                   'lq3_2233': -x[0]*1e-6, 'lq1_3333': x[1]*1e-6, 'lq3_3333': x[1]*1e-6},
                  eft='SMEFT', basis='Warsaw', scale=1e3)

def idemp(a, b):
    r'''
Creates an idempotent hermitic 3x3 matrix using to parameters.

:Arguments:

    - a, b\: Parameter of the matrix.

:Returns:

    - A `np.matrix`.
    '''

    return np.matrix([[abs(a)**2, a*np.conj(b), a], [np.conj(a)*b, abs(b)**2, b],
                      [np.conj(a), np.conj(b), 1]])/(1+ abs(a)**2 + abs(b)**2)

def matrixwc(num, C, ll, lq):
    r'''
Returns the Wilson coefficients for Clq1 or Clq3 given the parameters of the idemp matrix.
    '''
    wc = dict()
    for il in range(0, 3):
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
    r'''
NP affects only the third generation in the interaction basis
and then is rotated to the mass basis.
Couplings to the first generation not negligible.
C1 and C3 can be different.

:Arguments:

    -x \: Coordinates in the parameter space of the fit.
          x = [C1, C3, alpha_l, beta_l, alpha_q, beta_q].

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
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
    return Wilson({**w1, **w3}, eft='SMEFT', basis='Warsaw', scale=1e3)

def rotBI(x):
    r'''
Scenario BI\: NP affects only the third generation in the interaction basis
              and then is rotated to the mass basis.
              Couplings to the first generation negligible.

:Arguments:

    -x \: Coordinates in the parameter space of the fit. x = [C, beta_l, beta_q].

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return massrotation([x[0], x[0], 0, x[1], 0, x[2]])

def rotBII(x):
    r'''
Scenario BII\: NP affects only the third generation in the interaction basis
                and then is rotated to the mass basis.
                Couplings to the first generation not negligible.

:Arguments:

    -x \: Coordinates in the parameter space of the fit. x = [C, alpha_l, beta_l, alpha_q, beta_q].

:Returns:

    - A dictionary containing the SMEFT Wilson Coefficients of the fit.
    '''
    return massrotation([x[0], x[0], x[1], x[2], x[3], x[4]])

def rot2lqU1(x, M=1.5):
    r'''
Coupling of the U(1) leptoquarks obtained from the Wilson Coefficients.

:Arguments:

    - x\: Coordinates in the parameter space of the fit.
          It assumes scenario BI if len(x)==3 or scenario BII if len(x)==5.
    - [M\: Mass of the leptoquark, in TeV. Default=1.5.]

:Returns:

    - A `np.matrix` containing the couplings.
    '''
    if len(x) == 3:
        al = 0
        aq = 0
        bl = x[1]
        bq = x[2]
    elif len(x) == 5:
        al = x[1]
        bl = x[2]
        aq = x[3]
        bq = x[4]
    C = x[0]
    ll = idemp(al, bl)
    lq = idemp(aq, bq)
    xL = np.matrix(np.zeros([3, 3]))
    for i in range(0, 3):
        for j in range(0, 3):
            xL[i, j] = np.sqrt(-2*M**2*C * ll[i, i] * lq[j, j])* \
                       np.exp(1j*np.angle(lq[j, 2])-1j*np.angle(ll[i, 2]))
    return xL.T
