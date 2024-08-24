"""Module containing the classes for plotting figures."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ecdf, uniform  # type: ignore[import]

from sicore.core.base import SelectiveInferenceResult
from sicore.utils.evaluation import rejection_rate

plt.rcParams.update({"figure.autolayout": True})


def pvalues_hist(
    p_values: list[float] | np.ndarray,
    bins: int = 20,
    title: str | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """Plot histogram of p-values.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str | None, optional): Title of the figure. Defaults to None.
        fname (str | None, optional):
            File name. If `fname` is given, the plotted figure will
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


def pvalues_qqplot(
    p_values: list[float] | np.ndarray,
    plot_pos: list[float] | np.ndarray | None = None,
    title: str | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = (4, 4),
) -> None:
    """Plot uniform Q-Q plot of p-values.

    Args:
        p_values (list[float] | np.ndarray): List of p-values.
        plot_pos (list[float] | np.ndarray | None, optional):
            Plotting positions. If None, default plotting positions will be used.
            Defaults to None.
        title (str | None, optional): Title of the figure. Defaults to None.
        fname (str | None, optional):
            File name. If `fname` is given, the plotted figure will
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
    ) -> None:
        """Initialize a summary figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional): Label of y-axis. Defaults to None.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data: dict[str, list] = {}
        self.red_line = False

    def _add_value(self, value: float, label: str, xloc: str | float) -> None:
        """Add a value to the figure.

        Args:
            value (float): Value to be plotted.
            label (str): Label corresponding to the value.
                To note that the label well be shown in the given order.
            xloc (str | float): Location of the value.
                If str, it will be equally spaced in the given order.
                If float, it will be the exact location.
        """
        self.data.setdefault(label, [])
        self.data[label].append((xloc, value))

    def add_results(
        self,
        results: list[SelectiveInferenceResult] | list[float] | np.ndarray,
        label: str,
        xloc: str | float,
        alpha: float = 0.05,
        *,
        naive: bool = False,
        bonferroni: bool = False,
        log_num_comparisons: float = 0.0,
    ) -> None:
        """Add rejection rate computed from the given results to the figure.

        Args:
            results (list[SelectiveInferenceResult] | list[float] | np.ndarray):
                List of SelectiveInferenceResult objects or p-values.
            label (str):
                Label corresponding to the results.
            xloc (str | float):
                Location of the results.
            alpha (float, optional): Significance level. Defaults to 0.05.
            naive (bool, optional):
                Whether to compute rejection rate of naive inference.
                This option is available only when results are
                SelectiveInferenceResult objects. Defaults to False.
            bonferroni (bool, optional):
                Whether to compute rejection rate with Bonferroni correction.
                This option is available only when results are
                SelectiveInferenceResult objects. Defaults to False.
            log_num_comparisons (float, optional):
                Logarithm of the number of comparisons for the Bonferroni correction.
                This option is ignored when bonferroni is False.
                Defaults to 0.0, which means no correction.
        """
        value = rejection_rate(
            results,
            alpha=alpha,
            naive=naive,
            bonferroni=bonferroni,
            log_num_comparisons=log_num_comparisons,
        )
        self._add_value(value, label, xloc)

    def plot(
        self,
        fname: str | None = None,
        ylim: tuple[float, float] = (0.0, 1.0),
        yticks: list[float] | None = None,
        legend_loc: str | None = None,
        fontsize: int = 10,
    ) -> None:
        """Plot the figure.

        Args:
            fname (str | None, optional):
                File name. If `fname` is given, the plotted figure
                will be saved as a file. Defaults to None.
            ylim (tuple[float, float], optional):
                Range of y-axis. Defaults to (0.0, 1.0).
            yticks (list[float] | None, optional):
                List of y-ticks. Defaults to None.
                If None, y-ticks will be automatically determined.
            legend_loc (str | None, optional):
                Location of the legend. Defaults to None.
                If None, the legend will be placed at the best location.
            fontsize (int, optional):
                Font size of the legend. Defaults to 10.
        """
        plt.rcParams.update({"font.size": fontsize})
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

        for label, xloc_value_list in self.data.items():
            xlocs_, values_ = zip(*xloc_value_list, strict=False)
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

        plt.legend(frameon=False, loc=legend_loc)
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
    ) -> None:
        """Initialize a false positive rate figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional):
                Label of y-axis. Defaults to 'Type I Error Rate'.
        """
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)
        self.red_line = True

    def add_fpr(self, fpr: float, label: str, xloc: str | float) -> None:
        """Add a fpr value to the figure.

        Args:
            fpr (float): FPR value.
            label (str): Label corresponding to the FPR value.
            xloc (str | float): Location of the FPR value.
        """
        self._add_value(fpr, label, xloc)


class TprFigure(SummaryFigure):
    """A class plotting a true positive rate (power) summary figure.

    Args:
        title (str | None, optional): Title of the figure. Defaults to None.
        xlabel (str | None, optional): Label of x-axis. Defaults to None.
        ylabel (str | None, optional): Label of y-axis. Defaults to 'Power'.

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
    ) -> None:
        """Initialize a true positive rate figure."""
        super().__init__(title=title, xlabel=xlabel, ylabel=ylabel)
        self.red_line = False

    def add_tpr(self, tpr: float, label: str, xloc: str | float) -> None:
        """Add a tpr value to the figure.

        Args:
            tpr (float): TPR value.
            label (str): Label corresponding to the TPR value.
            xloc (str | float): Location of the TPR value.
        """
        self._add_value(tpr, label, xloc)
