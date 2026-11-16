"""Gain a clue by clustering: interactive charts for exploring text embeddings."""

import re
import warnings

import altair as alt
import numpy as np
import pandas as pd

__all__ = ["plot_text", "plot_text_comparison", "ClueWidget"]

# Number of example texts shown next to the scatter plot.
_MAX_TEXT_ROWS = 17
# Texts are cut to this many characters in the text panel (the tooltip shows the full text).
_MAX_TEXT_CHARS = 120

# A compact English stop word list, so the "top words" panel shows content words.
STOP_WORDS = frozenset("""
    a about above after again against all also am an and any are aren't as at be because been before
    being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't
    down during each few for from further get got had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
    it's its itself just let's like me more most mustn't my myself no nor not now of off on once only or
    other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some
    such than that that's the their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we we'd we'll we're we've
    were weren't what what's when when's where where's which while who who's whom why why's will with
    won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split())

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9']+")
# URLs, @mentions and HTML entities (like &amp;) are dropped before counting words.
_NOISE_PATTERN = re.compile(r"https?://\S+|www\.\S+|@\w+|&\w+;")


def _tokens(text):
    """Return the unique, lowercased, non-stop-word tokens of a text as one space separated string."""
    words = (
        tok.strip("'").removesuffix("'s")
        for tok in _TOKEN_PATTERN.findall(_NOISE_PATTERN.sub(" ", text.lower()))
    )
    return " ".join(
        dict.fromkeys(w for w in words if len(w) >= 3 and w not in STOP_WORDS)
    )


def _as_coords(X, name):
    """Turn X into a numpy array and check it has (at least) two columns."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(
            f"`{name}` should be a 2D array with two columns, got shape {X.shape}."
        )
    return X


def _build_frame(coords, texts, color_array=None, color_words=None):
    """
    Collect everything the chart needs into one DataFrame.

    Returns the DataFrame, the color type ("nominal", "quantitative" or None)
    and the sort order for the color legend.
    """
    texts = pd.Series(list(texts), dtype=object).fillna("").astype(str)
    n = len(texts)
    for name, values in coords.items():
        if len(values) != n:
            raise ValueError(
                f"The number of text examples ({n}) should match the number of rows in the array ({len(values)})."
            )

    if color_array is not None and color_words:
        raise ValueError("Pass either `color_array` or `color_words`, not both.")

    df = pd.DataFrame({**coords, "text": texts})
    df["trunc_text"] = df["text"].str[:_MAX_TEXT_CHARS]
    df["r"] = 0
    df["tokens"] = df["text"].map(_tokens)

    color_type, color_sort = None, None
    if color_array is not None:
        colors = pd.Series(list(color_array))
        if len(colors) != n:
            raise ValueError(
                f"The length of `color_array` ({len(colors)}) should match the number of text examples ({n})."
            )
        if pd.api.types.is_float_dtype(colors):
            # Continuous values, like predicted probabilities.
            color_type = "quantitative"
            df["color"] = colors.to_numpy()
        else:
            # Labels, cluster ids, booleans: treat them as categories.
            color_type = "nominal"
            df["color"] = colors.astype(str).to_numpy()
            color_sort = sorted(df["color"].unique().tolist())
    elif color_words:
        color_words = [str(w) for w in color_words]
        lowered = df["text"].str.lower()
        df["color"] = "none"
        # When a text contains several of the words, the last matching word wins.
        for w in color_words:
            df.loc[lowered.str.contains(w.lower(), regex=False), "color"] = w
        color_type = "nominal"
        color_sort = ["none"] + color_words

    return df, color_type, color_sort


def _make_params(color_type):
    """Create the selections: a rectangle brush and, for categories, a clickable legend."""
    brush = alt.selection_interval(name="brush")
    legend = (
        alt.selection_point(name="legend", fields=["color"], bind="legend")
        if color_type == "nominal"
        else None
    )
    return brush, legend


def _apply_filters(chart, brush, legend):
    """Keep only rows inside the brush (and in the clicked legend category, if any)."""
    chart = chart.transform_filter(brush)
    if legend is not None:
        chart = chart.transform_filter(legend)
    return chart


def _scatter(df, x, y, brush, legend, color_type, color_sort, title):
    """Scatter plot of the embedding; points outside the selection turn grey."""
    active = brush if legend is None else brush & legend
    if color_type is None:
        color = alt.condition(active, alt.value("steelblue"), alt.value("lightgray"))
    else:
        legend_style = alt.Legend(orient="bottom", direction="horizontal", title=None)
        color = alt.condition(
            active,
            alt.Color("color", type=color_type, sort=color_sort, legend=legend_style),
            alt.value("lightgray"),
        )

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X(f"{x}:Q", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["text"],
            color=color,
        )
        .properties(width=350, height=350, title=title)
        .add_params(brush)
    )
    if legend is not None:
        chart = chart.add_params(legend)
    return chart


def _selection_test(params):
    """
    A Vega expression that is true when a row is inside all of the given selections.

    This mirrors what `transform_filter(param)` does, but as a 0/1 value we can sum, so the
    count still shows "0 of N selected" when nothing matches (a filter would leave no rows).
    """
    tests = [
        f"(!length(data('{p.name}_store')) || vlSelectionTest('{p.name}_store', datum))"
        for p in params
        if p is not None
    ]
    return " && ".join(tests)


def _count_panel(df, brush, legend):
    """A single line of text: how many points are currently selected."""
    return (
        alt.Chart(df)
        .transform_calculate(
            is_selected=f"({_selection_test([brush, legend])}) ? 1 : 0"
        )
        .transform_aggregate(n="sum(is_selected)", total="count()")
        .transform_calculate(
            label="format(datum.n, ',') + ' of ' + format(datum.total, ',') + ' selected'"
        )
        .mark_text(align="left", baseline="top", fontSize=13, fontWeight="bold")
        .encode(x=alt.value(0), y=alt.value(0), text="label:N")
        .properties(width=330, height=16)
    )


def _text_panel(df, brush, legend):
    """The first few texts inside the selection."""
    chart = alt.Chart(df).transform_window(row_number="row_number()")
    chart = _apply_filters(chart, brush, legend)
    return (
        chart.transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank <= _MAX_TEXT_ROWS)
        .mark_text(align="left")
        .encode(
            x=alt.X("r:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("row_number:O", axis=None),
            text="trunc_text:N",
        )
        .properties(width=330, height=310)
    )


def _words_panel(df, brush, legend, top_words):
    """Bar chart of the words that appear in the most selected texts."""
    chart = _apply_filters(alt.Chart(df), brush, legend)
    return (
        chart.transform_calculate(word="split(datum.tokens, ' ')")
        .transform_flatten(["word"])
        .transform_filter("datum.word != null && datum.word != ''")
        .transform_aggregate(texts="count()", groupby=["word"])
        .transform_window(
            rank="row_number()",
            sort=[
                alt.SortField("texts", order="descending"),
                alt.SortField("word", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= top_words)
        .mark_bar(color="steelblue")
        .encode(
            y=alt.Y("word:N", sort="-x", title=None),
            x=alt.X(
                "texts:Q",
                title="selected texts containing word",
                axis=alt.Axis(tickMinStep=1, format="d"),
            ),
            tooltip=["word:N", "texts:Q"],
        )
        .properties(width=180, height=alt.Step(18), title="top words")
    )


def _layout(scatters, df, brush, legend, top_words):
    """Put the scatter plot(s), the text panel and the words panel side by side."""
    text_column = alt.vconcat(
        _count_panel(df, brush, legend), _text_panel(df, brush, legend), spacing=4
    )
    panels = [*scatters, text_column]
    if top_words:
        panels.append(_words_panel(df, brush, legend, top_words))
    with warnings.catch_warnings():
        # The comparison chart shares one brush between both scatter plots on purpose
        # (that is what links them); newer Altair versions warn about this.
        warnings.filterwarnings(
            "ignore", message="Automatically deduplicated selection"
        )
        chart = alt.hconcat(*panels)
    return chart.configure_axis(grid=False).configure_view(strokeWidth=0)


def _finish(chart, df, coord_cols, color_type, widget):
    """Return a plain Altair chart, or a ClueWidget that can report its selection back to Python."""
    if not widget:
        return chart
    try:
        import anywidget  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "`widget=True` needs the anywidget package. Install it with: python -m pip install 'cluestar[widget]'"
        ) from e
    columns = ["text", *coord_cols] + (["color"] if color_type else [])
    return ClueWidget(chart, data=df[columns])


def plot_text(
    X,
    texts,
    color_array=None,
    color_words=None,
    disable_warning=True,
    top_words=10,
    widget=False,
):
    """
    Make a visualisation to help find clues in text data.

    Drag a rectangle over the scatter plot to select points. The panels on the right show
    how many points you selected, the first few selected texts and the most common words
    in the selection.

    Arguments:
        - `X`: the numeric features, a 2D array with two columns (one row per text)
        - `texts`: list of text data
        - `color_array`: values to color the points by. Floats (e.g. predicted probabilities)
          get a color scale; anything else (labels, cluster ids, booleans) is treated as
          categories, and you can click a category in the legend to filter on it.
        - `color_words`: list of words to highlight; a text gets the color of the (last) word it contains.
          Click a word in the legend to filter on it.
        - `disable_warning`: disable the standard altair max rows warning
        - `top_words`: how many words to show in the "top words" panel; use 0 to hide it
        - `widget`: if True, return a `ClueWidget`. Display it in a notebook, select points,
          then call `.selected()` to get the selected rows back as a DataFrame.
          Needs `pip install 'cluestar[widget]'`.
    """
    if disable_warning:
        alt.data_transformers.disable_max_rows()

    X = _as_coords(X, "X")
    df, color_type, color_sort = _build_frame(
        {"x1": X[:, 0], "x2": X[:, 1]}, texts, color_array, color_words
    )
    brush, legend = _make_params(color_type)

    scatter = _scatter(
        df, "x1", "x2", brush, legend, color_type, color_sort, title="embedding space"
    )
    chart = _layout([scatter], df, brush, legend, top_words)
    return _finish(chart, df, ["x1", "x2"], color_type, widget)


def plot_text_comparison(
    X1,
    X2,
    texts,
    disable_warning=True,
    color_array=None,
    color_words=None,
    top_words=10,
    widget=False,
):
    """
    Make a visualisation that compares two embeddings of the same texts.

    Selecting points in one scatter plot highlights the same texts in the other.

    Arguments:
        - `X1`: the numeric features of the first embedding, a 2D array with two columns
        - `X2`: the numeric features of the second embedding, a 2D array with two columns
        - `texts`: list of text data
        - `disable_warning`: disable the standard altair max rows warning
        - `color_array`: values to color the points by; see `plot_text`
        - `color_words`: list of words to highlight; see `plot_text`
        - `top_words`: how many words to show in the "top words" panel; use 0 to hide it
        - `widget`: if True, return a `ClueWidget` whose `.selected()` method returns the selected rows
    """
    if disable_warning:
        alt.data_transformers.disable_max_rows()

    X1 = _as_coords(X1, "X1")
    X2 = _as_coords(X2, "X2")
    coords = {"x1": X1[:, 0], "y1": X1[:, 1], "x2": X2[:, 0], "y2": X2[:, 1]}
    df, color_type, color_sort = _build_frame(coords, texts, color_array, color_words)
    brush, legend = _make_params(color_type)

    scatters = [
        _scatter(
            df,
            "x1",
            "y1",
            brush,
            legend,
            color_type,
            color_sort,
            title="embedding space X1",
        ),
        _scatter(
            df,
            "x2",
            "y2",
            brush,
            legend,
            color_type,
            color_sort,
            title="embedding space X2",
        ),
    ]
    chart = _layout(scatters, df, brush, legend, top_words)
    return _finish(chart, df, list(coords), color_type, widget)


try:
    _JupyterChart = alt.JupyterChart
except AttributeError:  # pragma: no cover - very old altair
    _JupyterChart = object


class ClueWidget(_JupyterChart):
    """
    A cluestar chart as a Jupyter widget, so the selection you make can be used in Python.

    Created by `plot_text(..., widget=True)` or `plot_text_comparison(..., widget=True)`.
    Display it in a notebook cell, select points with the mouse, then call `.selected()`.
    """

    def __init__(self, chart, data, **kwargs):
        super().__init__(chart, **kwargs)
        self._data = data

    def selected(self):
        """
        Return the rows that are currently selected as a DataFrame.

        The index matches the position of each text in the `texts` you passed in,
        so `my_df.iloc[widget.selected().index]` gets you back your original rows.
        With nothing selected, all rows are returned.
        """
        df = self._data
        mask = pd.Series(True, index=df.index)

        brush = getattr(self.selections, "brush", None)
        for field, bounds in (
            (brush.value if brush is not None else None) or {}
        ).items():
            if field in df.columns and len(bounds):
                mask &= df[field].between(min(bounds), max(bounds))

        legend = getattr(self.selections, "legend", None)
        clicked = [
            item["color"]
            for item in ((legend.value if legend is not None else None) or [])
            if "color" in item
        ]
        if clicked:
            mask &= df["color"].isin(clicked)

        return df[mask]
