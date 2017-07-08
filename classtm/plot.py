import colorsys
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def get_separate_colors(count):
    num_div = np.linspace(0.0,1.0,num=count,endpoint=False)
    return [[a,b,c] for (a,b,c) in [colorsys.hsv_to_rgb(d,1.0,1.0) \
            for d in num_div]]

class Plotter(object):
    def __init__(self):
        self.fig, self.axis = plt.subplots(1,1)
        self.linesplotted_count = 0

    def plot(self, xmeans, ymeans, label, ymedians, yerr=None,
            xmedians=None, xerr=None):
        line, _, _ = self.axis.errorbar(xmeans, ymeans, xerr=xerr, yerr=yerr,
                label=label,
                linewidth=3)
        # plot dot at median
        if xmedians is not None:
            # if error bars extend on both axes
            self.axis.errorbar(xmedians, ymedians, fmt='o',
                    color=line.get_color())
        else:
            self.axis.errorbar(xmeans, ymedians, fmt='o',
                    color=line.get_color())

    def savefig(self, name):
        # put legend centered under plot
        lgd = self.axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        self.fig.savefig(name, bbox_extra_artist=(lgd,), bbox_inches='tight')

    def set_title(self, title):
        self.axis.set_title(title)

    def set_xlabel(self, label):
        self.axis.set_xlabel(label)

    def set_ylabel(self, label):
        self.axis.set_ylabel(label)

    def set_ylim(self, lims):
        self.axis.set_ylim(lims)

if __name__ == '__main__':
    # example of how to use Plotter
    line_count = 4
    plotter = Plotter()
    xmeans = np.linspace(0.0,1.0,10)
    for i in range(line_count):
        # generate some data
        ymeans = (xmeans * xmeans / 2) - i
        # calculate errorbars
        yerr = np.vstack((np.array([0.1]*len(xmeans)),
            np.array([0.1]*len(xmeans))))
        plotter.plot(xmeans, ymeans, 'line {:d}'.format(i), ymeans,
                yerr=yerr)
    plotter.set_title('Example')
    plotter.set_xlabel('x label')
    plotter.set_ylabel('y label')
    plotter.savefig('example.pdf')

