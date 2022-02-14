'''
==========
SMEFTglob
==========

Common functions used to calculate likelihood values and pulls of the fits.
'''

import os
from math import isinf
import warnings
from flavio.statistics.functions import pull
import smelli
import yaml
import SMEFT19

sm_cov_folder = os.path.join(SMEFT19.__path__[0], 'smcovariances')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    gl = smelli.GlobalLikelihood()
gl.load_sm_covariances(sm_cov_folder)
gl.make_measurement()

def restart_smelli(include_likelihoods=None, add_measurements=None,
                   remove_measurements=None, custom_likelihoods=None):
    '''
Re-starts smelli's Global Likelihood with new parameters.

:Arguments:
    - include_likelihoods\: If not None, only the specified likelihoods will be included.
    - add_measurements\: Adds more experimental measurements not included by smelli.
    - remove_measurements\: Removes more experimental measurements not included by smelli.
    - custom_likelihoods\: Adds new likelihoods.
    '''
    global gl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gl = smelli.GlobalLikelihood(include_likelihoods=include_likelihoods,
                                 add_measurements=add_measurements,
                                 remove_measurements=remove_measurements,
                                 custom_likelihoods=custom_likelihoods)
    gl.load_sm_covariances(sm_cov_folder)
    gl.make_measurement()

def likelihood_fits(x, wfun):
    '''
Calculates the log-likelihood of a NP hypothesis for several classes of observables.

:Arguments:
    - x\: Point in parameter space to be evaluated.
    - wfun\: Function that takes a point in parameter space and
      returns a dictionary of Wilson coefficents.

:Returns:
    - A dictionary of log-likelihoods, for each of the classes of observables defined by `smelli`.
    '''

    res = dict()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        glpp = gl.parameter_point(wfun(x))
        gldict = glpp.log_likelihood_dict()
        for f in gldict.keys():
            g = gldict[f]
            if isinf(g):
                if f == 'global':
                    g = 0
                    for f2 in list(gldict.keys())[:-1]:
                        g += res[f2]
                else:
                    g = -68
            res[f] = g
    return res


def likelihood_global(x, wfun):
    '''
Calculates the global log-likelihood of a NP hypothesis.

:Arguments:
    - x\: Point in parameter space to be evaluated.
    - wfun\: Function that takes a point in parameter space
      and returns a dictionary of Wilson coefficents.

:Returns:
    - The global log-likelihood.
    '''

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        glpp = gl.parameter_point(wfun(x))
        return glpp.log_likelihood_global()

def fastmeas(obs):
    '''
Checks if the observable is part of a <<fast-measurement>> in smelli.
    '''
    if obs not in gl.obstable_sm.keys():
        raise KeyError(f"The observable '{obs}' is not available in smelli")
    obsm = gl.obstable_sm[obs]
    lhname = obsm['lh_name']
    return lhname[:4] == 'fast'

def prediction(x, obs, wfun):
    '''
Interfaces `flavio` to compute the NP prediction of a given observable.

:Arguments:
    - x\: Point in parameter space to be evaluated.
    - obs\: observable, as defined by flavio, whose prediction will be computed.
      If the observable does not depend on any parameter, obs is a string.
      If the observable depends on numerical parameters (such as q2), obs is
      a list containing a string and one or more floats.
    - wfun\: Function that takes a point in parameter space and
      returns a dictionary of Wilson coefficents.

:Returns:
    - The prediction of the observable.
    '''
    if obs not in gl.obstable_sm.keys():
        raise KeyError(f"The observable '{obs}' is not available in smelli")
    obsm = gl.obstable_sm[obs]
    lhname = obsm['lh_name']
    wc = wfun(x)
    if fastmeas(obs):
        lh = gl.fast_likelihoods[lhname]
        ml = lh.likelihood.measurement_likelihood
        pred = ml.get_predictions_par(gl.par_dict_default, wc)
        return pred[obs]
    else:
        lh = gl.likelihoods[lhname]
        ml = lh.measurement_likelihood
        pred = ml.get_predictions_par(gl.par_dict_default, wc)
        return pred[obs]

def pull_obs(x, obs, wfun):
    '''
Calculates the pull, in sigmas, of the prediction of a given observable
in NP with respect to its experimental value.

:Arguments:
    - x\: Point in parameter space to be evaluated.
    - obs\: observable, as defined by `flavio`, whose prediction will be computed.
      If the observable does not depend on any parameter, obs is a string.
      If the observable depends on numerical parameters (such as q2), obs is
      a list containing a string and one or more floats.
    - wfun\: Function that takes a point in parameter space and
      returns a dictionary of Wilson coefficents.

:Returns:
    - The pull of the observable.
    '''
    if obs not in gl.obstable_sm.keys():
        raise KeyError(f"The observable '{obs}' is not available in smelli")
    obsm = gl.obstable_sm[obs]
    lhname = obsm['lh_name']
    pred = prediction(x, obs, wfun)
    ll_central = obsm['ll_central']
    if fastmeas(obs):
        lh = gl.fast_likelihoods[lhname]
        m = lh.pseudo_measurement
        ll = m.get_logprobability_single(obs, pred)
    else:
        p_comb = obsm['exp. PDF']
        ll = p_comb.logpdf([pred])
    return pull(-2*(ll-ll_central), 1)


def newlist():
    '''
Creates a `.yaml` file with a list of all observables available, ordered by their pull in the SM.
    '''
    glSM = gl.parameter_point({}, scale=1000)
    obsSM = glSM.obstable()
    obscoll = list(obsSM['pull exp.'].keys())
    for o in obscoll:
        if isinstance(o, tuple):
            o = list(o)
    with open(SMEFT19.__path__[0] + '/observables.yaml', 'wt', encoding='utf-8') as fyaml:
        yaml.dump(obscoll, fyaml)

def loadobslist(new=False):
    '''
Loads from a `.yaml` file a list of all observables available, ordered by their pull in the SM.
If the file does not exist, this functions creates it.

:Returns:
    - A list with all observables available.
    '''
    if new:
        newlist()
    else:
        try:
            with open(SMEFT19.__path__[0] + '/observables.yaml', 'rt', encoding='utf-8') as fyaml:
                obscoll = yaml.safe_load(fyaml)
        except (OSError, IOError):
            newlist()
    with open(SMEFT19.__path__[0] + '/observables.yaml', 'rt', encoding='utf-8') as fyaml:
        obscoll = yaml.safe_load(fyaml)
        for o in obscoll:
            if isinstance(o, list):
                o = tuple(o)
    return obscoll
