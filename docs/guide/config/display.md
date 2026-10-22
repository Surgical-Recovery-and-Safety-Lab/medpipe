# `[display]`

The figures produced for each outcome. The table is required when `run_mode` is `"audit"` or `"eval"`, the modes that produce figures, and it can be left out otherwise.

```toml
[display.defaults]
n_bootstraps = 1000
save = true
show = false
n_bins = 10
dist_n_bins = 25
dist_yscale = "log"
strategy = "uniform"

[display.overrides.calibration]
n_bootstraps = 10

[display.outcome_overrides.ANY_COMP.reliability]
strategy = "quantile"

[display.theme]
dpi = 300
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| [`defaults`](#displaydefaults) | table | all defaults | Settings used by every figure. |
| [`overrides`](#displayoverrides-and-displayoutcome_overrides) | table | none | Settings for one type of figure, for all outcomes. |
| [`outcome_overrides`](#displayoverrides-and-displayoutcome_overrides) | table | none | Settings for one type of figure, for one outcome. |
| [`theme`](#displaytheme) | table | none | Fonts, colours and resolution of the figures. |

The figures of each outcome are saved in the `plots/<outcome>/` folder of the run, and the subgroup heatmaps in `plots/`. For a classifier these are:

| Figure | File | Plot type |
| :--- | :--- | :--- |
| ROC curve | `<outcome>_roc_curve.png` | `roc` |
| Precision-recall curve | `<outcome>_pr_curve.png` | `precision_recall` |
| Reliability diagram (calibration curve) | `<outcome>_reliability_diagram.png` | `reliability` |
| Histogram of the predicted probabilities | `<outcome>_data_distribution.png` | `data_distribution` |
| Decision curve analysis | `<outcome>_dca_curve.png` | `dca` |
| Subgroup heatmap, one per metric | `<metric>_strata_heatmap.png` | `strata_heatmap` |

The heatmaps are only drawn when [`[workflow.evaluation.fairness]`](workflow.md#fairness) is defined. Each has one row for the whole test set and one for every subgroup, and one column for every outcome. The text is the value of the metric, and the colour is how far the value is from the one of the whole test set.

## `[display.defaults]`

Settings that apply to every figure.

```toml
[display.defaults]
n_bootstraps = 1000
save = true
show = false
n_bins = 10
dist_n_bins = 25
dist_yscale = "log"
strategy = "uniform"
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_bootstraps` | integer | `1000` | Number of bootstrap resamples for the 95% confidence bands of the curves. With `0`, the bands are not drawn. |
| `save` | boolean | `true` | Whether the figure is saved in the run folder. |
| `show` | boolean | `false` | Whether the figure is displayed in a window. |
| `n_bins` | integer | `10` | Number of bins of the reliability diagram. |
| `strategy` | string | `"uniform"` | How the reliability diagram is computed: `"uniform"` (bins of equal width), `"quantile"` (bins with the same number of samples) or `"spline"` (a smooth curve fitted with splines). |
| `cv_splines` | integer | `3` | Number of internal folds of the spline fit, 2 or more. Only used when `strategy` is `"spline"`. |
| `dist_n_bins` | integer | `10` | Number of bins of the histogram of the predicted probabilities. It is also used for the histogram drawn under the reliability diagram. |
| `dist_yscale` | string | `"linear"` | Scale of the y-axis of these histograms: `"linear"`, `"log"`, `"symlog"` or `"logit"`. |
| `n_jobs` | integer | `1` | Number of parallel jobs for the bootstrap of the figures. It is independent of `workflow.n_jobs`. |

!!! note
    The histograms have their own `dist_n_bins` and `dist_yscale`. `n_bins` only applies to the reliability diagram.

## `[display.overrides]` and `[display.outcome_overrides]`

The defaults can be changed for one type of figure with `overrides`, and for one type of figure of one outcome with `outcome_overrides`. Both use the same keys as `defaults`, and only the keys that are written are changed.

```toml
# Changes the calibration figures of all outcomes
[display.overrides.calibration]
n_bootstraps = 10

[display.overrides.roc]
n_bootstraps = 500

# Changes the reliability diagram of ANY_COMP only
[display.outcome_overrides.ANY_COMP.reliability]
n_bootstraps = 50
strategy = "quantile"
```

For a given figure, the settings are combined in this order, and the last one wins:

1. `[display.defaults]`
2. `[display.overrides.<plot type>]`
3. `[display.outcome_overrides.<outcome>.<plot type>]`
4. The arguments given when a plot method is called in Python.

The plot type is the name of the figure in the table below. Each type also has aliases, which can be used in any of the tables, and the different names of a type are combined.

| Plot type | Aliases | Keys that have an effect |
| :--- | :--- | :--- |
| `roc` | `roc_curve` | `n_bootstraps`, `n_jobs`, `save`, `show` |
| `precision_recall` | `pr`, `pr_curve` | `n_bootstraps`, `n_jobs`, `save`, `show` |
| `reliability` | `calibration`, `reliability_diagram` | `n_bins`, `strategy`, `cv_splines`, `dist_n_bins`, `dist_yscale`, `n_bootstraps`, `n_jobs`, `save`, `show` |
| `data_distribution` | `distribution`, `dist` | `dist_n_bins`, `dist_yscale`, `save`, `show` |
| `dca` | `dca_curve` | `save`, `show` |
| `strata_heatmap` | `heatmap` | `save`, `show` |

The name of the outcome in `outcome_overrides` must be in `data.outcomes`. A plot type that is not in the table is rejected. A key that has no effect for a plot type, such as `n_bins` for the ROC curve, is accepted and ignored.

## `[display.theme]`

The look of all the figures.

```toml
[display.theme]
primary_color = "#2D90D8"
ci_alpha = 0.3
dpi = 300
font_family = "sans-serif"
show_grid = false
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `primary_color` | string | `"#2D90D8"` | Colour of the curves and bars. The heatmaps have their own colour scale. |
| `palette` | list of strings | five colours | Reserved for figures that draw several series. **Currently not used.** |
| `ci_alpha` | float | `0.3` | Opacity of the confidence bands, between 0 and 1. |
| `linewidth` | float | `2.0` | Width of the curves. |
| `dpi` | integer | `300` | Resolution of the saved images. |
| `font_family` | string | `"sans-serif"` | Font of the titles, labels and ticks. |
| `title_fontsize` | integer | `12` | Size of the titles. |
| `label_fontsize` | integer | `10` | Size of the axis labels. |
| `show_spines` | boolean | `false` | Whether the top and right borders of the axes are drawn. |
| `show_grid` | boolean | `false` | Whether a grid is drawn. |

A key that is not in the table is rejected. Colours can be written as hexadecimal codes or as any colour name that [Matplotlib](https://matplotlib.org/stable/gallery/color/named_colors.html) accepts.
