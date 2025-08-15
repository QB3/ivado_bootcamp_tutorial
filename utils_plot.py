import os
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

import shutil


def configure_plt(fontsize=10, poster=True):
    """Configure matplotlib with TeX and seaborn."""
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    # usetex = matplotlib.checkdep_usetex(True)
    usetex = True if shutil.which('latex') else False
    # import ipdb; ipdb.set_trace()
    params = {'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")


def _plot_legend_apart(ax, figname, ncol=None):
    """Plot legend apart from figure."""
    # Do all your plots with fig, ax = plt.subplots(),
    # don't call plt.legend() at the end but this instead
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
               loc="upper center")
    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig


def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
