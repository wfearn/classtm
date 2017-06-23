import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import numpy as np


def get_values(filename):
    """Get values from file"""
    with open(filename) as ifh:
        return [float(a) for a in ifh]


def plot_hists(ax, data):
    """Plot histogram"""
    lines = []
    for (label, values) in data:
        lines.append(ax.hist(values, label=label, alpha=0.75))
    return lines


def _works():
    """This one works"""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    lines = plot_hists(ax1, data)
    plot_hists(ax2, data)
    ax1.set_ylabel('number of rows')
    ax2.set_xlim([0, 200])
    ax1.set_title('Full')
    ax2.set_title('Zoomed')
    fig.savefig('iters.pdf', bbox_inches='tight')


def _main():
    """Plot histograms of iteration data"""
    data = [
        ('supervised', get_values('sup.iters')),
        ('overwatched', get_values('over.iters')),
        ('free', get_values('projected.iters')),
    ]

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    plot_hists(ax, data)
    ax.legend(loc='lower right')
    ax.set_xlabel('iterations')
    ax.set_ylabel('rows')
    axins = inset_axes(ax, 3, 2, loc=1)
    plot_hists(axins, data)
    axins.axis([0, 200, 0, 1650])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls='dashdot')
    fig.savefig('iters_inset.pdf', bbox_inches='tight')


if __name__ == '__main__':
    _main()
