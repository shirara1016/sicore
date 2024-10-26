"""Module containing the classes for plotting figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ecdf, norm, uniform  # type: ignore[import]

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

    Parameters
    ----------
    p_values : list[float] | np.ndarray
        List of p-values.
    bins : int, optional
        The number of bins. Defaults to 20.
    title : str | None, optional
        Title of the figure. Defaults to None.
    fname : str | None, optional
        File name. If `fname` is given, the plotted figure will be saved as a file.
        Defaults to None.
    figsize : tuple[float, float], optional
        Size of the figure. Defaults to (6, 4).
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

    Parameters
    ----------
    p_values : list[float] | np.ndarray
        List of p-values.
    plot_pos : list[float] | np.ndarray | None, optional
        Plotting positions. If None, default plotting positions will be used.
        Defaults to None.
    title : str | None, optional
        Title of the figure. Defaults to None.
    fname : str | None, optional
        File name. If `fname` is given, the plotted figure will be saved as a file.
        Defaults to None.
    figsize : tuple[float, float], optional
        Size of the figure. Defaults to (4, 4).
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

    Examples
    --------
    >>> fig = SummaryFigure(xlabel='Image Size', ylabel="Type I Error Rate")
    >>> fig.add_value(0.053, "proposed", "64")
    >>> fig.add_value(0.048, "proposed", "256")
    >>> fig.add_value(0.046, "proposed", "1024")
    >>> fig.add_value(0.052, "proposed", "4096")
    >>> fig.add_value(0.413, "naive", "64")
    >>> fig.add_value(0.821, "naive", "256")
    >>> fig.add_value(0.483, "naive", "1024")
    >>> fig.add_value(0.418, "naive", "4096")
    >>> fig.add_red_line(0.05, "significance level")
    >>> fig.plot(filepath="fpr.pdf", fontsize=16)
    """

    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        """Initialize a summary figure.

        Parameters
        ----------
        title : str | None, optional
            Title of the figure. Defaults to None.
        xlabel : str | None, optional
            Label of x-axis. Defaults to None.
        ylabel : str | None, optional
            Label of y-axis. Defaults to None.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data: dict[str, list[tuple[str | float, float, float | None]]] = {}
        self.red_lines: list[tuple[float, str | None]] = []

    def add_value(
        self,
        value: float,
        label: str,
        xloc: str | float,
    ) -> None:
        """Add a value to the figure.

        Parameters
        ----------
        value : float
            Value to be plotted.
        label : str
            Label corresponding to the value.
            To note that the label well be shown in the given order.
        xloc : str | float
            Location of the value.
            If str, it will be equally spaced in the given order.
            If float, it will be the exact location.
        """
        self.data.setdefault(label, [])
        self.data[label].append((xloc, value, None))

    def add_results(
        self,
        results: list[SelectiveInferenceResult] | list[float] | np.ndarray,
        label: str,
        xloc: str | float,
        alpha: float = 0.05,
        confidence_level: float | None = None,
        *,
        naive: bool = False,
        bonferroni: bool = False,
        log_num_comparisons: float = 0.0,
    ) -> None:
        """Add rejection rate computed from the given results to the figure.

        Parameters
        ----------
        results : list[SelectiveInferenceResult] | np.ndarray | list[float]
            List of SelectiveInferenceResult objects or p-values.
        label : str
            Label corresponding to the results.
        xloc : str | float
            Location of the results.
        alpha : float, optional
            Significance level. Defaults to 0.05.
        naive (bool, optional):
            Whether to compute rejection rate of naive inference.
            This option is available only when results are
            SelectiveInferenceResult objects. Defaults to False.
        bonferroni : bool, optional
            Whether to compute rejection rate with Bonferroni correction.
            This option is available only when results are
            SelectiveInferenceResult objects. Defaults to False.
        log_num_comparisons : float, optional
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
        if confidence_level is None:
            self.add_value(value, label, xloc)
        else:
            scale = norm.ppf(1 - (1 - confidence_level) / 2)
            error = scale * np.sqrt(value * (1 - value) / len(results))
            self.data.setdefault(label, [])
            self.data[label].append((xloc, value, error))

    def add_red_line(self, value: float = 0.05, label: str | None = None) -> None:
        """Add a red line at the specified value.

        Parameters
        ----------
        value : float, optional
            Value to be plotted as a red line. Defaults to 0.05.
        label : str | None, optional
            Label of the red line. Defaults to None.
        """
        self.red_lines.append((value, label))

    def plot(
        self,
        filepath: Path | str | None = None,
        ylim: tuple[float, float] | None = (0.0, 1.0),
        yticks: list[float] | None = None,
        legend_loc: str | None = None,
        fontsize: int = 10,
    ) -> None:
        """Plot the figure.

        Parameters
        ----------
        filepath : Path | str | None, optional
            File path. If `filepath` is given, the plotted figure will be saved as a file.
            Defaults to None.
        ylim : tuple[float, float] | None, optional
            Range of y-axis. If None, range of y-axis will be automatically determined.
            Defaults to None.
        yticks : list[float] | None, optional
            List of y-ticks. If None, y-ticks will be automatically determined.
            Defaults to None.
        legend_loc : str | None, optional
            Location of the legend. If None, the legend will be placed at the best location.
            Defaults to None.
        fontsize : int, optional
            Font size of the legend. Defaults to 10.
        """
        plt.rcParams.update({"font.size": fontsize})

        plt.title(self.title if self.title is not None else "")
        plt.xlabel(self.xlabel if self.xlabel is not None else "")
        plt.ylabel(self.ylabel if self.ylabel is not None else "")

        for label, items in self.data.items():
            xlocs, values, errors = map(np.array, zip(*items, strict=True))
            if not all(isinstance(xloc, (str)) for xloc in xlocs):
                values = values[np.argsort(xlocs)]
                errors = errors[np.argsort(xlocs)]
                xlocs = np.sort(xlocs)
            if any(error is None for error in errors):
                plt.plot(xlocs, values, label=label, marker="x")
            else:
                plt.errorbar(
                    xlocs,
                    values,
                    errors,
                    fmt="o-",
                    capsize=3,
                    markersize=4,
                    label=label,
                    elinewidth=1.2,
                )

        for value, label_ in self.red_lines:
            plt.plot(
                xlocs,
                [value] * len(xlocs),
                color="red",
                linestyle="--",
                lw=0.5,
                label=label_,
            )

        plt.xticks(xlocs)
        plt.ylim(ylim)
        plt.yticks(yticks)

        handles_, labels_ = plt.gca().get_legend_handles_labels()
        handles, labels = zip(
            *(
                [
                    (handle, label)
                    for handle, label in zip(handles_, labels_, strict=True)
                    if not isinstance(handle, plt.Line2D)
                ]
                + [
                    (handle, label)
                    for handle, label in zip(handles_, labels_, strict=True)
                    if isinstance(handle, plt.Line2D)
                ]
            ),
            strict=False,
        )
        plt.legend(handles, labels, loc=legend_loc, frameon=False)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(
                str(filepath) if isinstance(filepath, Path) else filepath,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.clf()
        plt.close()
