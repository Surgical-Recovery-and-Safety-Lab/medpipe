"""
Themes for displaying graphs.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from matplotlib.typing import RcKeyType


@dataclass
class MedpipeTheme:
    """Central aesthetic configuration for Medpipe pipeline visualizations.

    This class manages color palettes, line weights, font sizes, axes spines, and
    Matplotlib runtime parameter updates across all graphics generated within the
    pipeline.

    Attributes
    ----------
    primary_color : str, default="#2D90D8"
        Hex string or Matplotlib color identifier for primary plot elements.
    palette : list of str
        Ordered list of color hex codes used for multi-class or multi-outcome figures.
    ci_alpha : float, default=0.3
        Opacity level for confidence interval shaded fills, bounded in [0.0, 1.0].
    linewidth : float, default=2.0
        Line width in points for main plotted curves.
        Base Matplotlib style sheet name applied to figure contexts.
    dpi : int, default=300
        Dots per inch (resolution) for saved rasterized images.
    font_family : str, default="sans-serif"
        Font family specification for titles, labels, and tick annotations.
    title_fontsize : int, default=12
        Font size in points for axes and figure titles.
    label_fontsize : int, default=10
        Font size in points for x and y axis labels.
    show_spines : bool, default=False
        Whether to draw top and right axes border lines.

    Methods
    -------
    to_rc_params()
        Convert theme options into a dictionary compatible with Matplotlib rcParams.
    get_color(index)
        Retrieve a color from the palette by index, wrapping cyclically if exceeded.
    from_dict(data)
        Construct a MedpipeTheme instance from a dictionary configuration.
    """

    primary_color: str = "#2D90D8"
    palette: List[str] = field(
        default_factory=lambda: [
            "#2D90D8",
            "#33367A",
            "#96690E",
            "#CDB4DB",
            "#F2CC8F",
        ]
    )
    ci_alpha: float = 0.3
    linewidth: float = 2.0
    dpi: int = 300
    font_family: str = "sans-serif"
    title_fontsize: int = 12
    label_fontsize: int = 10
    show_spines: bool = False
    show_grid: bool = False

    def to_rc_params(self) -> Dict[RcKeyType, Any]:
        """Convert theme attributes into Matplotlib runtime configurations.

        Returns
        -------
        Dict[RcKeytype, Any]
            Dictionary mapping standard Matplotlib `rcParams` keys to theme values.

        """
        return {
            "font.family": self.font_family,
            "axes.titlesize": self.title_fontsize,
            "axes.labelsize": self.label_fontsize,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "axes.spines.top": self.show_spines,
            "axes.spines.right": self.show_spines,
            "axes.grid": self.show_grid,
        }

    def get_color(self, index: int) -> str:
        """Retrieve a color from the palette by index.

        If the specified index exceeds the number of colors in the palette, the
        selection wraps around cyclically.

        Parameters
        ----------
        index : int
            The zero-based index of the requested color.

        Returns
        -------
        str
            Hex color code from the palette corresponding to the index.

        """
        if not self.palette:
            return self.primary_color
        return self.palette[index % len(self.palette)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedpipeTheme":
        """Instantiate a MedpipeTheme configuration from a dictionary.

        Filters out unrecognized keys to prevent initialization errors when reading
        from TOML or external configuration dictionaries.

        Parameters
        ----------
        data : dict of {str : Any}
            Dictionary containing theme attribute key-value pairs.

        Returns
        -------
        MedpipeTheme
            A newly constructed theme instance populated with provided settings.

        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
