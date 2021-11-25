'''
================
utils
================

This module contains some auxiliary functions
'''


import re
import numpy as np
import flavio


def sign(x, y):
    r'''
Returns -1 if the first argument is strictly less than the second one, and 1 otherwise.    
    '''
    if float(x) < float(y):
        return -1
    else:
        return 1

def tex(obs):
    r'''
Returns the TeX representation for a given flavio observable.
If the observable includes arguments (e. g. q2), they are represented as superindex.    
    '''
    if isinstance(obs, str):
        text = flavio.Observable[obs].tex
    else:
        text = (flavio.Observable[obs[0]].tex[:-1] + '^{' +
                str(list(obs[1:])).replace(',', ',\\ ') + '}$')
    return text.replace('text', 'mathrm')

def texnumber(x, prec=3):
    r'''
Returns the TeX representation of a number in scientific notation.    
    '''
    texn = ('{:.'+str(prec)+'g}').format(float(x))
    match = re.match(r'(-?[0-9]+(\.[0-9]+)?)e(.[0-9]+)', texn)
    if match:
        texn = '$' + match.group(1) + '\\times 10^{' + str(int(match.group(3))) + '}$'
    return texn

def roundsig(x, num=4):
    r'''
Rounds a number to a fixed number of significative digits.    
    '''
    l = int(np.log10(abs(x)))
    return round(x, -l+num)
    
def listpoint(x):
    r'''
If passed a single 2-tuple (representing one point), it returns a list containing the tuple.
If passed more than 1 2-tuple (representing several points), returns the argument.
TODO: Rewrite in python 3.10 using match&case syntax.    
    '''
    if len(x) == 2:
        if len(np.array(x).flat) == 2:
            return [x,]
        else:
            return x
    else:
        return x
        

def distrsphere(dim):
    '''
Returns a random vector with norm 1.

:Arguments:
    - dim\: Dimension of the vector
    '''
    vect = np.random.randn(dim)
    return vect/np.linalg.norm(vect)
