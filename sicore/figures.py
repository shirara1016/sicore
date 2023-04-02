from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import numpy as np
from scipy.stats import uniform, norm, chi
from statsmodels.distributions.empirical_distribution import ECDF

from .evaluation import false_positive_rate, power
from .intervals import intersection, not_

rcParams.update({"figure.autolayout": True})


def pvalues_hist(pvalues, bins=20, title=None, fname=None, figsize=(6, 4)):
    """
    Plot histogram of p-values.

    Args:
        pvalues (array-like): List of p-values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str, optional): Title of the figure. Defaults to None.
        fname (str, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (6, 4).
    """
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel("p-value")
    plt.ylabel("frequency")
    plt.hist(pvalues, bins=bins, range=(0, 1))
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()
    plt.close()


def pvalues_qqplot(pvalues, plot_pos=None, title=None, fname=None, figsize=(4, 4)):
    """
    Plot uniform Q-Q plot of p-values.

    Args:
        pvalues (array-like): List of p-values.
        plot_pos (array-like, optional): Plotting positions. If None, default plotting
            positions will be used. Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.
        fname (str, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (4, 4).
    """
    n = len(pvalues)
    plot_pos = plot_pos or [k / (n + 1) for k in range(1, n + 1)]
    t_quantiles = list(map(uniform.ppf, plot_pos))  # theoretical
    e_quantiles = list(map(ECDF(pvalues), plot_pos))  # empirical
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel("theoretical quantiles of Unif(0, 1)")
    plt.ylabel("empirical quantiles of p-values")
    plt.plot([0, 1], [0, 1])
    plt.plot(t_quantiles, e_quantiles, marker=".", linestyle="None")
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()
    plt.close()


class SummaryFigure:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data = defaultdict(list)

    def add_value(self, value, label, xloc):
        self.data[label].append((xloc, value))

    def plot(self, fname=None, sort_xlocs=True):
        """
        Plot the figure.

        Args:
            fname (str, optional): File name. If `fname` is given, the plotted figure
                will be saved as a file. Defaults to None.
            sort_xlocs (bool, optional): If True, xlocs will be sorted in ascending
                order.
        """
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        plt.ylim(0, 1)
        for label, list_ in self.data.items():
            if sort_xlocs:
                list_.sort(key=lambda x: x[0])
            xlocs = [v[0] for v in list_]
            values = [v[1] for v in list_]
            plt.plot(xlocs, values, label=label)
        plt.legend()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, transparent=True)
        plt.clf()
        plt.close()


class FprFigure(SummaryFigure):
    """
    Plot a fpr summary figure

    Usage1:
        fig = FprFigure(title='Figure1', xlabel='parameter')
        fig.add_fpr(0.1, 'naive', 1)
        fig.add_fpr(0.3, 'naive', 2)
        fig.add_fpr(0.8, 'naive', 3)
        fig.add_fpr(0.05, 'selective', 1)
        fig.add_fpr(0.05, 'selective', 2)
        fig.add_fpr(0.05, 'selective', 3)
        fig.plot()

    Usage2:
        fig = FprFigure(title='Figure2', xlabel='setting')
        fig.add_pvalues([0.8, 0.01, 0.06], 'naive', 'hoge')
        fig.add_pvalues([0.03, 0.05, 0.2], 'naive', 'foo')
        fig.add_pvalues([0.01, 0.0, 0.02], 'naive', 'bar')
        fig.plot(file_name='figure2.pdf', sort_xlocs=False)
    """

    def __init__(self, title=None, xlabel=None, ylabel="false positive rate"):
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)

    def add_pvalues(self, pvalues, label, xloc, alpha=0.05):
        """
        Add p-values to the figure.

        Args:
            pvalues (array-like): List of p-values.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
            alpha (float, optional): Significance level.
        """
        fpr = false_positive_rate(pvalues, alpha=alpha)
        self.add_fpr(fpr, label, xloc)

    def add_fpr(self, fpr, label, xloc):
        """
        Add a fpr value to the figure.

        Args:
            fpr (float): FPR value.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
        """
        self.add_value(fpr, label, xloc)


class PowerFigure(SummaryFigure):
    """
    Plot a power summary figure

    Usage1:
        fig = PowerFigure(title='Figure1', xlabel='parameter')
        fig.add_power(0.95, 'naive', 1)
        fig.add_power(0.95, 'naive', 2)
        fig.add_power(0.95, 'naive', 3)
        fig.add_power(0.1, 'selective', 1)
        fig.add_power(0.3, 'selective', 2)
        fig.add_power(0.8, 'selective', 3)
        fig.plot()

    Usage2:
        fig = PowerFigure(title='Figure2', xlabel='setting')
        fig.add_pvalues([0.8, 0.01, 0.2], 'naive', 'hoge')
        fig.add_pvalues([0.8, 0.01, 0.03], 'naive', 'foo')
        fig.add_pvalues([0.02, 0.01, 0.03], 'naive', 'bar')
        fig.plot(file_name='figure2.pdf', sort_xlocs=False)
    """

    def __init__(self, title=None, xlabel=None, ylabel="power"):
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)

    def add_pvalues(self, pvalues, label, xloc, alpha=0.05):
        """
        Add p-values to the figure.

        Args:
            pvalues (array-like): List of p-values.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
            alpha (float, optional): Significance level.
        """
        power_ = power(pvalues, alpha=alpha)
        self.add_power(power_, label, xloc)

    def add_power(self, power, label, xloc):
        """
        Add a power value to the figure.

        Args:
            power (float): Power value.
            label (str): Label plotted in the legend.
            xloc (str, float): xloc value.
        """
        self.add_value(power, label, xloc)


def search_history_visualizer(history, fname='search_history.pdf', figsize=(8, 6)):
    """
    Visualize history of searched intervals and truncated intervals.

    Args:
        history (List[Type[SearchProgress]]):
            A list consisting of data classes that summarizes the
            intermediate stages of each search and stores the history of
            the parametric search.
        fname (str, optional):
            File name. Defaults to 'search_history.pdf'.
        figsize (tuple, optional):
            Size of the figure. Defaults to (8, 6).
    """
    distribution = history[0].null_distribution
    if distribution == 'norm':
        dist = norm
        start = -6
        end = 6
    if distribution[:3] == 'chi':
        df = int(distribution[3:])
        dist = chi(df=df)
        start = min(0, np.sqrt(df - 1) - 6)
        end = start + 12
    x = np.linspace(start, end, 1000)
    y = dist.pdf(x)

    stat = history[0].stat
    startegy = history[0].choose_method
    prev_R, prev_S = [], []

    pdf_pages = PdfPages(fname)
    for progress in history:
        plt.figure(figsize=figsize)

        inf_p = progress.inf_p
        sup_p = progress.sup_p
        title = f'Search strategy is {startegy}, p-value in [{inf_p:.3f}, {sup_p:.3f}]'
        plt.title(title)

        plt.plot(x, y, color='black', lw=0.8)
        plt.plot(x, np.zeros_like(x), color='black', lw=0.8)

        plt.plot([stat, stat], [-0.008, 0.008], color='black', lw=0.8)
        plt.text(stat, -0.02, 'stat', ha='center', color='black')

        plt.scatter([progress.search_point], [0], color='black', s=8)

        current_R = progress.truncated_intervals
        current_S = progress.searched_intervals
        current_S = intersection(current_S, not_(current_R))

        newly_R = intersection(current_R, not_(prev_R))
        newly_S = intersection(current_S, not_(prev_S))

        for R in [current_R, newly_R]:
            for interval in R:
                mask = (x > interval[0]) & (x < interval[1])
                plt.fill_between(x, y, 0, where=mask, color='blue', alpha=0.3)

        for S in [current_S, newly_S]:
            for interval in S:
                mask = (x > interval[0]) & (x < interval[1])
                plt.fill_between(x, y, 0, where=mask, color='red', alpha=0.3)

        prev_R = current_R
        prev_S = current_S

        pdf_pages.savefig()
        plt.clf()
        plt.close()
    pdf_pages.close()


def pvalue_bounds_plot(history, title=None, fname=None, figsize=(8, 6)):
    """
    Plot the upper and lower bounds of the p-value obtained for each search.

    Args:
        history (List[Type[SearchProgress]]):
            A list consisting of data classes that summarizes the
            intermediate stages of each search and stores the history of
            the parametric search.
        title (str, optional):
            Title of the figure. Defaults to None.
        fname (str, optional):
            File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple, optional):
            Size of the figure. Defaults to (8, 6).
    """

    inf_ps = [0.0] + [prog.inf_p for prog in history]
    sup_ps = [1.0] + [prog.sup_p for prog in history]
    ps = [None] + [prog.p_value for prog in history]
    alpha = history[0].alpha
    alphas = alpha * np.ones(len(history) + 1)
    count = list(range(len(history) + 1))

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel('search count')
    plt.ylabel('p-value')
    plt.xlim([0, len(history)])
    plt.ylim([0.0, 1.0])
    yticks = [0, alpha, 0.2, 0.4, 0.6, 0.8, 1.0]
    ylabels = [f'{value:.2f}' for value in yticks]
    plt.yticks(yticks, ylabels)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.plot(count, alphas, color='tab:red', linestyle='--', lw=0.5)
    plt.plot(count, sup_ps, color='tab:orange',
             marker='o', ms=3, label='upper bound')
    plt.plot(count, inf_ps, color='tab:blue',
             marker='o', ms=3, label='lower bound')
    plt.plot(count, ps, color='black', marker='o',
             ms=2, label='p-value', lw=0.7)

    plt.legend()
    if fname is None:
        plt.show()
    else:
        plt.savefig('proceed.pdf', transparent=True)
    plt.clf()
    plt.close()
