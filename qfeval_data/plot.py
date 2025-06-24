import typing

import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as _plt

    # Dot-per-inch of figures (default: 100dpi)
    _plt.rcParams["figure.dpi"] = 200
    # Font size (default: 10pt)
    _plt.rcParams["font.size"] = 14

    # Show grids.
    _plt.rcParams["axes.grid"] = True
    _plt.rcParams["grid.color"] = "#aaaaaa"
    _plt.rcParams["grid.linewidth"] = 0.3

    # Set the graph frame's linewidth.
    _plt.rcParams["axes.linewidth"] = 1.0

    # Set the legend settings.
    _plt.rcParams["legend.facecolor"] = "white"
    _plt.rcParams["legend.edgecolor"] = "black"
    _plt.rcParams["legend.framealpha"] = 1.0
    _plt.rcParams["legend.fontsize"] = 10
except ImportError:
    matplotlib = None  # type:ignore[assignment]
    _plt = None  # type:ignore[assignment]


class Layer(object):
    """Layer of an additional plot."""

    def __init__(self) -> None:
        pass


def plot_dataframe(
    df: pd.DataFrame, **kwargs: typing.Any
) -> typing.List[matplotlib.axes.Axes]:  # type:ignore[return]
    ax = df.plot(**kwargs)
    # If a legend exists.
    if ax.get_legend() is not None:
        # Calculate the bounding box of the main graph.
        # NOTE: To determine the final bounding box (incl. texts) in the axes
        # coordinate (which is required to specify a legend's anchor point),
        # the bounding box in the absolute coordinate (pixels) is required.
        bbox_px = ax.get_tightbbox(ax.get_figure().canvas.get_renderer())
        bbox = ax.transAxes.inverted().transform(bbox_px)
        # Show a legend below the main graph.
        ax.legend(
            # Suppress the legend title because it is rarely useful.
            title=None,
            # Locate the legend below the main graph.
            loc="upper center",
            bbox_to_anchor=((bbox[0][0] + bbox[1][0]) / 2, bbox[0][1]),
            ncol=4,
            # Configure horizontal spaces between columns.
            columnspacing=0.5,
            # Configure horizontal spaces, each of which exists between a line
            # and a text.
            handletextpad=0.3,
        )
    return [ax]


class Figure(object):
    def __init__(
        self,
        figure: typing.Union[
            None,
            matplotlib.axes.Axes,  # type:ignore[valid-type,valid-type]
            matplotlib.figure.Figure,  # type:ignore[valid-type,valid-type]
        ] = None,
    ):
        if matplotlib is None:
            raise ImportError(
                "matplotlib is not installed. Please install it with `pip install qfeval-data[plot]`"
            )
        if isinstance(figure, matplotlib.figure.Figure):
            self.__figure = figure
        elif isinstance(figure, matplotlib.axes.Axes):
            self.__figure = figure.get_figure()  # type:ignore[assignment]
        else:
            self.__figure = _plt.gcf()
        if len(self.axes) == 0:
            self.figure.set_constrained_layout_pads(  # type:ignore[attr-defined]
                hspace=0.06
            )

    @property
    def figure(self) -> matplotlib.figure.Figure:  # type:ignore[return]
        return self.__figure

    @property
    def axes(self) -> typing.List[matplotlib.axes.Axes]:  # type:ignore[return]
        return list(
            sorted(self.figure.axes, key=lambda x: float(-x.get_position().y1))
        )

    @property
    def primary_axes(self) -> matplotlib.axes.Axes:  # type:ignore[return]
        if len(self.figure.axes) == 0:
            self.figure.add_subplot()
        return self.axes[0]

    def show(self) -> None:
        self.figure.show()

    def append_axes(
        self, scale: float = 0.4
    ) -> matplotlib.axes.Axes:  # type:ignore[return]
        hspace = self.figure.get_constrained_layout_pads()[3]
        params = matplotlib.figure.SubplotParams()
        height = params.top - params.bottom
        axes = self.axes
        for ax in axes:
            ax.set_xticklabels([])
        top = (
            params.top
            if len(axes) == 0
            else axes[-1].get_position().y0 - hspace
        )
        axes = self.figure.add_axes(  # type:ignore[call-overload]
            [
                params.left,
                top - height * scale,
                params.right - params.left,
                height * scale,
            ]
        )
        return axes  # type:ignore[return-value]
