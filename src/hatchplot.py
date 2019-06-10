import matplotlib.pyplot as plt
from flavio.plots.plotfunctions import likelihood_contour_data
from flavio.statistics.functions import delta_chi2, confidence_level
import flavio.plots.colors
import scipy.interpolate
import numpy as np

plt.rcParams['hatch.color'] = 'w'

hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

def likelihood_hatch_contour(log_likelihood, x_min, x_max, y_min, y_max,
              n_sigma=1, steps=20, threads=1,
              **kwargs):
    r"""Plot coloured confidence contours (or bands) given a log likelihood
    function.

    Parameters:

    - `log_likelihood`: function returning the logarithm of the likelihood.
      Can e.g. be the method of the same name of a FastFit instance.
    - `x_min`, `x_max`, `y_min`, `y_max`: data boundaries
    - `n_sigma`: plot confidence level corresponding to this number of standard
      deviations. Either a number (defaults to 1) or a tuple to plot several
      contours.
    - `steps`: number of grid steps in each dimension (total computing time is
      this number squared times the computing time of one `log_likelihood` call!)

    All remaining keyword arguments are passed to the `contour` function
    and allow to control the presentation of the plot (see docstring of
    `flavio.plots.plotfunctions.contour`).
    """
    data = likelihood_contour_data(log_likelihood=log_likelihood,
                                x_min=x_min, x_max=x_max,
                                y_min=y_min, y_max=y_max,
                                n_sigma=n_sigma, steps=steps, threads=threads)
    data.update(kwargs) #  since we cannot do **data, **kwargs in Python <3.5
    return hatch_contour(**data)

def hatch_contour(x, y, z, levels,
              interpolation_factor=1,
              interpolation_order=2,
              col=0, label=None,
              contour_args={}, contourf_args={}):
    r"""Plot coloured and hatched confidence contours (or bands) given numerical input
    arrays. Based on the flavio function

    Parameters:

    - `x`, `y`: 2D arrays containg x and y values as returned by numpy.meshgrid
    - `z` value of the function to plot. 2D array in the same shape as `x` and
      `y`. The lowest value of the function should be 0 (i.e. the best fit
      point).
    - levels: list of function values where to draw the contours. They should
      be positive and in ascending order.
    - `interpolation factor` (optional): in between the points on the grid,
      the functioncan be interpolated to get smoother contours.
      This parameter sets the number of subdivisions (default: 1, i.e. no
      interpolation). It should be larger than 1.
    - `col` (optional): number between 0 and 9 to choose the color of the plot
      from a predefined palette
    - `label` (optional): label that will be added to a legend created with
       maplotlib.pyplot.legend()
    - `contour_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contour() (that draws the contour lines)
    - `contourf_args`: dictionary of additional options that will be passed
       to matplotlib.pyplot.contourf() (that paints the contour filling).
    """
    if interpolation_factor > 1:
        x = scipy.ndimage.zoom(x, zoom=interpolation_factor, order=1)
        y = scipy.ndimage.zoom(y, zoom=interpolation_factor, order=1)
        z = scipy.ndimage.zoom(z, zoom=interpolation_factor, order=interpolation_order)
    if not isinstance(col, int):
        _col = 0
    else:
        _col = col
    _contour_args = {}
    _contourf_args = {}
    _contour_args['colors'] = [flavio.plots.colors.set1[_col]]
    _contour_args['linewidths'] = 1.2
    N = len(levels)
    _contourf_args['colors'] = [flavio.plots.colors.pastel[_col] # RGB
                                       + (max(1-n/(N+1), 0),) # alpha, decreasing for contours
                                       for n in range(1,N+1)]
    hl = []
    for i in range(0, N):
        hl.append(hatches[_col]*(N-i))
    hl.append(None)
    _contourf_args['hatches'] = hl
    _contour_args['linestyles'] = 'solid'
    _contour_args.update(contour_args)
    _contourf_args.update(contourf_args)
    # for the filling, need to add zero contour
    levelsf = [np.min(z)] + list(levels)
    ax = plt.gca()
    CF = ax.contourf(x, y, z, levels=levelsf, **_contourf_args)
    CS = ax.contour(x, y, z, levels=levels, **_contour_args)
    if label is not None:
        CS.collections[0].set_label(label)
    return (CS, CF)

