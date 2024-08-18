import numpy as np
from scipy.stats import uniform, ecdf  # type: ignore
import matplotlib.pyplot as plt

from .evaluation import false_positive_rate, true_positive_rate

plt.rcParams.update({"figure.autolayout": True})


def p_values_hist(
    p_values: list[float] | np.ndarray,
    bins: int = 20,
    title: str | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = (6, 4),
):
    """
    Plot histogram of p-values.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str | None, optional): Title of the figure. Defaults to None.
        fname (str | None, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple[float, float], optional): Size of the figure. Defaults to (6, 4).
    """
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.xlabel("p-value")
    plt.ylabel("frequency")
    plt.hist(p_values, bins=bins, range=(0, 1), density=True)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, transparent=True)
    plt.clf()
    plt.close()


def p_values_qqplot(
    p_values: list[float] | np.ndarray,
    plot_pos: list[float] | np.ndarray | None = None,
    title: str | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = (4, 4),
):
    """
    Plot uniform Q-Q plot of p-values.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        plot_pos (list[float] | np.ndarray | None, optional): Plotting positions.
            If None, default plotting positions will be used. Defaults to None.
        title (str | None, optional): Title of the figure. Defaults to None.
        fname (str | None, optional): File name. If `fname` is given, the plotted figure will
            be saved as a file. Defaults to None.
        figsize (tuple[float, float], optional): Size of the figure. Defaults to (4, 4).
    """
    n = len(p_values)
    plot_pos = plot_pos or [k / (n + 1) for k in range(1, n + 1)]
    t_quantiles = list(map(uniform.ppf, plot_pos))  # theoretical
    e_quantiles = list(map(ecdf(p_values).cdf.evaluate, plot_pos))  # empirical
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
    """A class plotting a summary figure of experiments.

    Args:
        title (str | None, optional): Title of the figure. Defaults to None.
        xlabel (str | None, optional): Label of x-axis. Defaults to None.
        ylabel (str | None, optional): Label of y-axis. Defaults to None.
    """

    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """Initialize a summary figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional): Label of y-axis. Defaults to None.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data: dict[str, list] = dict()
        self.red_line = False

    def add_value(self, value: float, label: str, xloc: str | int | float):
        """Add a value to the figure.

        Args:
            value (float): Value to be plotted.
            label (str): Label corresponding to the value.
                To note that the label well be shown in the given order.
            xloc (str | int | float): Location of the value.
                If str, it will be equally spaced in the given order.
                If int or float, it will be the exact location.
        """
        self.data.setdefault(label, [])
        self.data[label].append((xloc, value))

    def plot(
        self,
        fname: str | None = None,
        ylim: tuple[float, float] = (0.0, 1.0),
        yticks: list[float] | None = None,
        loc: str | None = None,
        fontsize: int = 10,
    ):
        """
        Plot the figure.

        Args:
            fname (str | None, optional): File name. If `fname` is given, the plotted figure
                will be saved as a file. Defaults to None.
            ylim (tuple[float, float], optional): Range of y-axis. Defaults to (0.0, 1.0).
            yticks (list[float] | None, optional): List of y-ticks. Defaults to None.
                If None, y-ticks will be automatically determined.
            loc (str | None, optional): Location of the legend. Defaults to None.
                If None, the legend will be placed at the best location.
            fontsize (int, optional): Font size of the legend. Defaults to 10.
        """
        plt.rcParams.update({"font.size": fontsize})
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

        for label, xloc_value_list in self.data.items():
            xlocs_, values_ = zip(*xloc_value_list)
            xlocs, values = np.array(xlocs_), np.array(values_)
            if not all(isinstance(xloc, (str)) for xloc in xlocs):
                values = values[np.argsort(xlocs)]
                xlocs = np.sort(xlocs)
            plt.plot(xlocs, values, label=label, marker="x")

        if self.red_line:
            plt.plot(xlocs, [0.05] * len(xlocs), color="red", linestyle="--", lw=0.5)

        plt.xticks(xlocs)
        plt.ylim(ylim)
        if yticks is not None:
            plt.yticks(yticks)

        plt.legend(frameon=False, loc=loc)
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.close()


class FprFigure(SummaryFigure):
    """A class plotting a false positive rate (type I error rate) summary figure.

    Args:
        title (str | None, optional): Title of the figure. Defaults to None.
        xlabel (str | None, optional): Label of x-axis. Defaults to None.
        ylabel (str | None, optional): Label of y-axis. Defaults to Type I Error Rate.

    Examples:
        >>> fig = FprFigure(xlabel='Image Size')
        >>> fig.add_fpr(0.053, "proposed", "64")
        >>> fig.add_fpr(0.048, "proposed", "256")
        >>> fig.add_fpr(0.046, "proposed", "1024")
        >>> fig.add_fpr(0.052, "proposed", "4096")
        >>> fig.add_fpr(0.413, "naive", "64")
        >>> fig.add_fpr(0.821, "naive", "256")
        >>> fig.add_fpr(0.483, "naive", "1024")
        >>> fig.add_fpr(0.418, "naive", "4096")
        >>> fig.plot()
    """

    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str = "Type I Error Rate",
    ):
        """Initialize a false positive rate figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional): Label of y-axis. Defaults to Type I Error Rate.
        """
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)
        self.red_line = True

    def add_p_values(
        self,
        p_values: list[float] | np.ndarray,
        label: str,
        xloc: str | int | float,
        alpha: float = 0.05,
    ):
        """Add p-values to the figure.

        Args:
            p_values (list[float] | np.ndarray): List of p-values.
            label (str): Label corresponding to the p-values.
            xloc (str | int | float): Location of the p-values.
            alpha (float, optional): Significance level.
        """
        fpr = false_positive_rate(p_values, alpha=alpha)
        self.add_fpr(fpr, label, xloc)

    def add_fpr(self, fpr: float, label: str, xloc: str | int | float):
        """Add a fpr value to the figure.

        Args:
            fpr (float): FPR value.
            label (str): Label corresponding to the FPR value.
            xloc (str | int | float): Location of the FPR value.
        """
        self.add_value(fpr, label, xloc)


class TprFigure(SummaryFigure):
    """A class plotting a true positive rate (power) summary figure.

    Args:
        title (str | None, optional): Title of the figure. Defaults to None.
        xlabel (str | None, optional): Label of x-axis. Defaults to None.
        ylabel (str | None, optional): Label of y-axis. Defaults to Power.

    Examples:
        >>> fig = TprFigure(xlabel='signal')
        >>> fig.add_tpr(0.183, "proposed", 1.0)
        >>> fig.add_tpr(0.579, "proposed", 2.0)
        >>> fig.add_tpr(0.846, "proposed", 3.0)
        >>> fig.add_tpr(0.948, "proposed", 4.0)
        >>> fig.plot()
    """

    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str = "Power",
    ):
        """Initialize a true positive rate figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional): Label of y-axis. Defaults to Power.
        """
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)
        self.red_line = False

    def add_p_values(
        self,
        p_values: list[float] | np.ndarray,
        label: str,
        xloc: str | int | float,
        alpha: float = 0.05,
    ):
        """Add p-values to the figure.

        Args:
            p_values (list[float] | np.ndarray): List of p-values.
            label (str): Label corresponding to the p-values.
            xloc (str | int | float): Location of the p-values.
            alpha (float, optional): Significance level.
        """
        tpr = true_positive_rate(p_values, alpha=alpha)
        self.add_tpr(tpr, label, xloc)

    def add_tpr(self, tpr: float, label: str, xloc: str | int | float):
        """Add a tpr value to the figure.

        Args:
            tpr (float): TPR value.
            label (str): Label corresponding to the TPR value.
            xloc (str | int | float): Location of the TPR value.
        """
        self.add_value(tpr, label, xloc)
