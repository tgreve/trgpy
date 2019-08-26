import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# === Plot data  ==============================================================
def make_plot(axes, x, ex, y, ey, xtitle=False, ytitle=False, color='k',
              markerfacecolor=False, marker='o', markeredgecolor='black',
              markersize=11, label='', new_plot=False, show_errors=True,
              show_data=True):

    if new_plot:
        axes.tick_params(which='minor', length=8)
        axes.tick_params(which='major', length=13)
        axes.tick_params(axis='x', pad=10)

        axes.set_xlim(5.E-5,10)
        axes.set_ylim(0,1.2)
        plt.xscale('log')

        axes.set_xlabel(xtitle, fontsize=23, labelpad=15)
        axes.set_ylabel(ytitle, fontsize=23, labelpad=15)

    if show_data:
        # --- Remove data points undefined in either x or y
        indices_not_undefined_x = np.where(x != -999)
        if len(indices_not_undefined_x) != 0:
            x = x[indices_not_undefined_x]
            ex = ex[indices_not_undefined_x]
            y = y[indices_not_undefined_x]
            ey = ey[indices_not_undefined_x]
        indices_not_undefined_y = np.where(y != -999)
        if len(indices_not_undefined_y) != 0:
            x = x[indices_not_undefined_y]
            ex = ex[indices_not_undefined_y]
            y = y[indices_not_undefined_y]
            ey = ey[indices_not_undefined_y]

        # --- Get indices for detections in x *and* y
        indices_not_upper_limits_x = np.where(ex != 99)
        indices_not_upper_limits_y = np.where(ey != 99)

        indices_not_lower_limits_x = np.where(ex != -99)
        indices_not_lower_limits_y = np.where(ey != -99)

        indices_not_upper_limits_x_y = np.intersect1d(indices_not_upper_limits_x,
                                                indices_not_upper_limits_y)
        indices_not_lower_limits_x_y = np.intersect1d(indices_not_lower_limits_x,
                                                indices_not_lower_limits_y)

        indices_detections_x_y = np.intersect1d(indices_not_upper_limits_x_y,
                                                indices_not_lower_limits_x_y)

        # --- Make plot with error bars and limit-arrows
        if show_errors:

            # --- Plot x upper limits and y detections
            indices_upper_limits_x = np.where(ex == 99)

            indices_upper_limits_x_not_upper_limits_y = np.intersect1d(indices_upper_limits_x,
                                                                 indices_not_upper_limits_y)

            indices_upper_limits_x_detections_y = np.intersect1d(indices_upper_limits_x_not_upper_limits_y,
                                                                 indices_not_lower_limits_y)

            if len(indices_upper_limits_x_detections_y) != 0:
                axes.errorbar(x[indices_upper_limits_x_detections_y],
                              y[indices_upper_limits_x_detections_y], linestyle='',
                              xerr=0.2*x[indices_upper_limits_x_detections_y],
                              xuplims=True,
                              yerr=ey[indices_upper_limits_x_detections_y],
                              color=color, marker='', markersize=5,
                              markerfacecolor=markerfacecolor,
                              markeredgecolor=markeredgecolor)

            # --- Plot x detections and y upper limits
            indices_upper_limits_y = np.where(ey == 99)

            indices_detections_x = np.intersect1d(indices_not_upper_limits_x,
                                                  indices_not_lower_limits_x)

            indices_detections_x_upper_limits_y = np.intersect1d(indices_detections_x,
                                                                 indices_upper_limits_y)

            if len(indices_detections_x_upper_limits_y) != 0:
                axes.errorbar(x[indices_detections_x_upper_limits_y],
                              y[indices_detections_x_upper_limits_y], linestyle='',
                              xerr=ex[indices_detections_x_upper_limits_y],
                              yerr=0.05,
                              uplims=True,
                              color=color, marker='', markersize=5,
                              markerfacecolor=markerfacecolor)

            # --- Plot x upper limits and y upper limits
            indices_upper_limits_x_y = np.intersect1d(indices_upper_limits_x,
                                                      indices_upper_limits_y)

            if len(indices_upper_limits_x_y) != 0:
                axes.errorbar(x[indices_upper_limits_x_y],
                              y[indices_upper_limits_x_y], linestyle='',
                              xerr=0.2*x[indices_upper_limits_x_y],
                              xuplims=True,
                              yerr=0.05,
                              uplims=True, color=color)


            # --- Plot x detection and y lower limit
            indices_lower_limits_y = np.where(ey == -99)

            indices_detections_x = np.intersect1d(indices_not_upper_limits_x,
                                                  indices_not_lower_limits_x)

            indices_detections_x_lower_limits_y = np.intersect1d(indices_lower_limits_y,
                                                                 indices_detections_x)

            if len(indices_detections_x_lower_limits_y) != 0:
                axes.errorbar(x[indices_detections_x_lower_limits_y],
                              y[indices_detections_x_lower_limits_y], linestyle='',
                              xerr=0.5*ex[indices_detections_x_lower_limits_y],
                              yerr=0.05, lolims=True,
                              color=color, marker='', markersize=5,
                              markerfacecolor=markerfacecolor,
                              markeredgecolor=markeredgecolor)

            # --- Plot x lower limits and y detections
            indices_lower_limits_x = np.where(ex == -99)

            indices_detections_y = np.intersect1d(indices_not_upper_limits_y,
                                                  indices_not_lower_limits_y)

            indices_lower_limits_x_detections_y = np.intersect1d(indices_lower_limits_x,
                                                                 indices_detections_y)

            if len(indices_lower_limits_x_detections_y) != 0:
                axes.errorbar(x[indices_lower_limits_x_detections_y],
                              y[indices_lower_limits_x_detections_y], linestyle='',
                              xerr=0.2*x[indices_lower_limits_x_detections_y],
                              xlolims=True,
                              yerr=ey[indices_lower_limits_x_detections_y],
                              color=color, marker='', markersize=5,
                              markerfacecolor=markerfacecolor,
                              markeredgecolor=markeredgecolor)

            # --- Plot x upper limits and y lower limits
            indices_upper_limits_x_lower_limits_y = np.intersect1d(indices_upper_limits_x,
                                                      indices_lower_limits_y)

            if len(indices_upper_limits_x_lower_limits_y) != 0:
                axes.errorbar(x[indices_upper_limits_x_lower_limits_y],
                              y[indices_upper_limits_x_lower_limits_y], linestyle='',
                              xerr=0.2*x[indices_upper_limits_x_lower_limits_y],
                              xuplims=True,
                              yerr=0.05,
                              lolims=True, color=color)

            # --- Plot x lower limits and y lower limits
            indices_lower_limits_x_y = np.intersect1d(indices_lower_limits_x,
                                                      indices_lower_limits_y)

            if len(indices_lower_limits_x_y) != 0:
                axes.errorbar(x[indices_lower_limits_x_y],
                              y[indices_lower_limits_x_y], linestyle='',
                              xerr=0.2*x[indices_lower_limits_x_y],
                              xlolims=True,
                              yerr=0.05,
                              lolims=True, color=color)

            # --- Plot x lower limits and y upper limits
            indices_lower_limits_x_upper_limits_y = np.intersect1d(indices_lower_limits_x,
                                                      indices_upper_limits_y)

            if len(indices_lower_limits_x_upper_limits_y) != 0:
                axes.errorbar(x[indices_lower_limits_x_upper_limits_y],
                              y[indices_lower_limits_x_upper_limits_y], linestyle='',
                              xerr=0.2*x[indices_lower_limits_x_upper_limits_y],
                              xlolims=True,
                              yerr=0.05,
                              uplims=True, color=color)


            # --- Plot x and y detections
            if len(indices_detections_x_y) != 0:
                axes.errorbar(x[indices_detections_x_y],
                              y[indices_detections_x_y], linestyle='',
                              xerr=ex[indices_detections_x_y],
                              yerr=ey[indices_detections_x_y], color=color,
                              marker=marker, markersize=markersize,
                              markerfacecolor=markerfacecolor,
                              markeredgecolor=markeredgecolor, label=label)
        else:
            # --- Plot x and y detections
            if len(indices_detections_x_y) != 0:
                axes.errorbar(x[indices_detections_x_y],
                              y[indices_detections_x_y], linestyle='',
                              color=color, marker=marker, markersize=markersize,
                              markerfacecolor=markerfacecolor,
                              markeredgecolor=markeredgecolor, label=label)
# =============================================================================















##
##
##
#def latexify(fig_width=None, fig_height=None, columns=1):
#    """Set up matplotlib's RC params for LaTeX plotting.
#    Call this before plotting a figure.
#
#    Parameters
#    ----------
#    fig_width : float, optional, inches
#    fig_height : float,  optional, inches
#    columns : {1, 2}
#    """
#
#    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
#
#    # Width and max height in inches for IEEE journals taken from
#    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
#
#    assert(columns in [1,2])
#
#    if fig_width is None:
#        fig_width = 3.39 if columns==1 else 6.9 # width in inches
#
#    if fig_height is None:
#        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
#        fig_height = fig_width*golden_mean # height in inches
#
#    MAX_HEIGHT_INCHES = 8.0
#    if fig_height > MAX_HEIGHT_INCHES:
#        print("WARNING: fig_height too large:" + fig_height +
#              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
#        fig_height = MAX_HEIGHT_INCHES
#
#    params = {'backend': 'ps',
#              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
#              'axes.titlesize': 18,
#              'font.size': 8, # was 10
#              'legend.fontsize': 8, # was 10
#              'xtick.labelsize': 8,
#              'ytick.labelsize': 8,
#              'text.usetex': True,
#              'figure.figsize': [fig_width,fig_height],
#              'font.family': 'serif'
#    }
#
#    matplotlib.rcParams.update(params)
#
#
##
##
##
#def format_axes(ax):
#
#    for spine in ['top', 'right']:
#        ax.spines[spine].set_visible(False)
#
#    for spine in ['left', 'bottom']:
#        ax.spines[spine].set_color(SPINE_COLOR)
#        ax.spines[spine].set_linewidth(0.5)
#
#    ax.xaxis.set_ticks_position('bottom')
#    ax.yaxis.set_ticks_position('left')
#
#    for axis in [ax.xaxis, ax.yaxis]:
#        axis.set_tick_params(direction='out', color=SPINE_COLOR)
#
#    return ax
#
#
#
##
##
##
#def setup_graph(title='', x_label='', y_label='',fig_size=None):
#    """Graphing helper function"""
#
#    latexify()
#    fig = plt.figure()
#    if fig_size != None:
#        fig.set_size_inches(fig_size[0], fig_size[1])
#    ax = fig.add_subplot(111)
#    ax.set_title(title,fontsize=20)
#    ax.set_xlabel(x_label,fontsize=25)
#    ax.set_ylabel(y_label,fontsize=25)
#    ax.tick_params(axis='both', which='major', labelsize=20)
#    plt.subplots_adjust(left=0.14,right=0.95,top=0.95,bottom=0.13)
#    ax.tick_params('both', length=10, width=1, which='major')
#    ax.tick_params('both', length=5, width=1, which='minor')
#
#
#
## === Plot data  ==============================================================
#def make_plot(axes, x, ex, y, ey, xtitle=False, ytitle=False, color='k',
#              markerfacecolor=False, marker='o', markeredgecolor='black',
#              markersize=11, label='', new_plot=False, show_errors=True,
#              show_data=True):
#
#    if new_plot:
#        axes.tick_params(which='minor', length=8)
#        axes.tick_params(which='major', length=13)
#        axes.tick_params(axis='x', pad=10)
#
#        axes.set_xlim(1.E7,5.E13)
#        axes.set_ylim(0.,1.2)
#        plt.xscale('log')
#        axes.set_xlabel(xtitle, fontsize=23, labelpad=15)
#        axes.set_ylabel(ytitle, fontsize=23, labelpad=15)
#
#    if show_data:
#        # --- Remove data points undefined in either x or y
#        indices_not_undefined_x = np.where(x != -999)
#        if len(indices_not_undefined_x) != 0:
#            x = x[indices_not_undefined_x]
#            ex = ex[indices_not_undefined_x]
#            y = y[indices_not_undefined_x]
#            ey = ey[indices_not_undefined_x]
#        indices_not_undefined_y = np.where(y != -999)
#        if len(indices_not_undefined_y) != 0:
#            x = x[indices_not_undefined_y]
#            ex = ex[indices_not_undefined_y]
#            y = y[indices_not_undefined_y]
#            ey = ey[indices_not_undefined_y]
#
#        # --- Get indices for detections in x *and* y
#        indices_not_upper_limits_x = np.where(ex != 99)
#        indices_not_upper_limits_y = np.where(ey != 99)
#
#        indices_not_lower_limits_x = np.where(ex != -99)
#        indices_not_lower_limits_y = np.where(ey != -99)
#
#        indices_not_upper_limits_x_y = np.intersect1d(indices_not_upper_limits_x,
#                                                indices_not_upper_limits_y)
#        indices_not_lower_limits_x_y = np.intersect1d(indices_not_lower_limits_x,
#                                                indices_not_lower_limits_y)
#
#        indices_detections_x_y = np.intersect1d(indices_not_upper_limits_x_y,
#                                                indices_not_lower_limits_x_y)
#
#        # --- Make plot with error bars and limit-arrows
#        if show_errors:
#
#            # --- Plot x upper limits and y detections
#            indices_upper_limits_x = np.where(ex == 99)
#
#            indices_upper_limits_x_not_upper_limits_y = np.intersect1d(indices_upper_limits_x,
#                                                                 indices_not_upper_limits_y)
#
#            indices_upper_limits_x_detections_y = np.intersect1d(indices_upper_limits_x_not_upper_limits_y,
#                                                                 indices_not_lower_limits_y)
#
#            if len(indices_upper_limits_x_detections_y) != 0:
#                axes.errorbar(x[indices_upper_limits_x_detections_y],
#                              y[indices_upper_limits_x_detections_y], linestyle='',
#                              xerr=0.2*x[indices_upper_limits_x_detections_y],
#                              xuplims=True,
#                              yerr=ey[indices_upper_limits_x_detections_y],
#                              color=color, marker='', markersize=5,
#                              markerfacecolor=markerfacecolor,
#                              markeredgecolor=markeredgecolor)
#
#            # --- Plot x detections and y upper limits
#            indices_upper_limits_y = np.where(ey == 99)
#
#            indices_detections_x = np.intersect1d(indices_not_upper_limits_x,
#                                                  indices_not_lower_limits_x)
#
#            indices_detections_x_upper_limits_y = np.intersect1d(indices_detections_x,
#                                                                 indices_upper_limits_y)
#
#            if len(indices_detections_x_upper_limits_y) != 0:
#                axes.errorbar(x[indices_detections_x_upper_limits_y],
#                              y[indices_detections_x_upper_limits_y], linestyle='',
#                              xerr=ex[indices_detections_x_upper_limits_y],
#                              yerr=0.05,
#                              uplims=True,
#                              color=color, marker='', markersize=5,
#                              markerfacecolor=markerfacecolor)
#
#            # --- Plot x upper limits and y upper limits
#            indices_upper_limits_x_y = np.intersect1d(indices_upper_limits_x,
#                                                      indices_upper_limits_y)
#
#            if len(indices_upper_limits_x_y) != 0:
#                axes.errorbar(x[indices_upper_limits_x_y],
#                              y[indices_upper_limits_x_y], linestyle='',
#                              xerr=0.2*x[indices_upper_limits_x_y],
#                              xuplims=True,
#                              yerr=0.05,
#                              uplims=True, color=color)
#
#
#            # --- Plot x detection and y lower limit
#            indices_lower_limits_y = np.where(ey == -99)
#
#            indices_detections_x = np.intersect1d(indices_not_upper_limits_x,
#                                                  indices_not_lower_limits_x)
#
#            indices_detections_x_lower_limits_y = np.intersect1d(indices_lower_limits_y,
#                                                                 indices_detections_x)
#
#            if len(indices_detections_x_lower_limits_y) != 0:
#                axes.errorbar(x[indices_detections_x_lower_limits_y],
#                              y[indices_detections_x_lower_limits_y], linestyle='',
#                              xerr=ex[indices_detections_x_lower_limits_y],
#                              yerr=0.05, lolims=True,
#                              color=color, marker='', markersize=5,
#                              markerfacecolor=markerfacecolor,
#                              markeredgecolor=markeredgecolor)
#
#            # --- Plot x lower limits and y detections
#            indices_lower_limits_x = np.where(ex == -99)
#
#            indices_detections_y = np.intersect1d(indices_not_upper_limits_y,
#                                                  indices_not_lower_limits_y)
#
#            indices_lower_limits_x_detections_y = np.intersect1d(indices_lower_limits_x,
#                                                                 indices_detections_y)
#
#            if len(indices_lower_limits_x_detections_y) != 0:
#                axes.errorbar(x[indices_lower_limits_x_detections_y],
#                              y[indices_lower_limits_x_detections_y], linestyle='',
#                              xerr=0.2*x[indices_lower_limits_x_detections_y],
#                              xlolims=True,
#                              yerr=ey[indices_lower_limits_x_detections_y],
#                              color=color, marker='', markersize=5,
#                              markerfacecolor=markerfacecolor,
#                              markeredgecolor=markeredgecolor)
#
#            # --- Plot x upper limits and y lower limits
#            indices_upper_limits_x_lower_limits_y = np.intersect1d(indices_upper_limits_x,
#                                                      indices_lower_limits_y)
#
#            if len(indices_upper_limits_x_lower_limits_y) != 0:
#                axes.errorbar(x[indices_upper_limits_x_lower_limits_y],
#                              y[indices_upper_limits_x_lower_limits_y], linestyle='',
#                              xerr=0.2*x[indices_upper_limits_x_lower_limits_y],
#                              xuplims=True,
#                              yerr=0.05,
#                              lolims=True, color=color)
#
#            # --- Plot x lower limits and y lower limits
#            indices_lower_limits_x_y = np.intersect1d(indices_lower_limits_x,
#                                                      indices_lower_limits_y)
#
#            if len(indices_lower_limits_x_y) != 0:
#                axes.errorbar(x[indices_lower_limits_x_y],
#                              y[indices_lower_limits_x_y], linestyle='',
#                              xerr=0.2*x[indices_lower_limits_x_y],
#                              xlolims=True,
#                              yerr=0.05,
#                              lolims=True, color=color)
#
#            # --- Plot x lower limits and y upper limits
#            indices_lower_limits_x_upper_limits_y = np.intersect1d(indices_lower_limits_x,
#                                                      indices_upper_limits_y)
#
#            if len(indices_lower_limits_x_upper_limits_y) != 0:
#                axes.errorbar(x[indices_lower_limits_x_upper_limits_y],
#                              y[indices_lower_limits_x_upper_limits_y], linestyle='',
#                              xerr=0.2*x[indices_lower_limits_x_upper_limits_y],
#                              xlolims=True,
#                              yerr=0.05,
#                              uplims=True, color=color)
#
#
#            # --- Plot x and y detections
#            if len(indices_detections_x_y) != 0:
#                axes.errorbar(x[indices_detections_x_y],
#                              y[indices_detections_x_y], linestyle='',
#                              xerr=ex[indices_detections_x_y],
#                              yerr=ey[indices_detections_x_y], color=color,
#                              marker=marker, markersize=markersize,
#                              markerfacecolor=markerfacecolor,
#                              markeredgecolor=markeredgecolor, label=label)
#        else:
#            # --- Plot x and y detections
#            if len(indices_detections_x_y) != 0:
#                axes.errorbar(x[indices_detections_x_y],
#                              y[indices_detections_x_y], linestyle='',
#                              color=color, marker=marker, markersize=markersize,
#                              markerfacecolor=markerfacecolor,
#                              markeredgecolor=markeredgecolor, label=label)
## =============================================================================
