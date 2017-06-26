import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import numpy as np


def get_values(filename):
    """Get row normalized cooccurrences matrix"""
    with open(filename, 'rb') as ifh:
        result = np.load(ifh)
    return (result.T / result.sum(axis=1).T).T


def plot_heat(ax, data):
    """Plot heatmap"""
    return ax.matshow(data, cmap='viridis')


def plot_inset(ax, data, loc, loc1, loc2, limits):
    """Plot inset"""
    axins = inset_axes(ax, '25%', '25%', loc=loc)
    plot_heat(axins, data)
    axins.axis(limits)
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none")
    axins.xaxis.set_major_locator(FixedLocator([limits[0], limits[1]]))
    axins.yaxis.set_major_locator(FixedLocator([limits[2], limits[3]]))


def _main():
    """Plot histograms of iteration data"""
    data = [
        ('supervised', get_values('sup.Q')),
        ('overwatched', get_values('supnormed.Q')),
        ('free', get_values('projected.Q')),
    ]

    for datum in data:
        fig, ax = plt.subplots()
        # fig.set_size_inches(4, 4)
        caxins = plot_heat(ax, datum[1])
        fig.colorbar(caxins)
        height, width = datum[1].shape
        # far right (since we're using matshow, the y axes limits need to be
        # flipped for the inset axes to match the diretion of the rest of the
        # plot)
        plot_inset(ax, datum[1], 7, 2, 4, [width-21, width-1, 20, 0])
        # bottom corner
        plot_inset(ax, datum[1], 8, 1, 3, [width-21, width-1, height-1, height-21])
        fig.savefig(datum[0]+'_cooccurrences.pdf', bbox_inches='tight')


if __name__ == '__main__':
    _main()
