import numpy as np
import pandas as pd
import altair as alt


def plot_text(X, texts, color_array=None, color_words=None, disable_warning=True):
    """
    Make a visualisation to help find clues in text data.

    Arguments:
        - `X`: the numeric features, should be a 2D numpy array
        - `texts`: list of text data
        - `color_words`: list of words to highlight
        - `color_array`: an array that represents color for the plot
        - `disable_warning`: disable the standard altair max rows warning
    """
    if disable_warning:
        alt.data_transformers.disable_max_rows()

    if len(texts) != X.shape[0]:
        raise ValueError(
            f"The number of text examples ({len(texts)}) should match X array ({X.shape[0]})."
        )

    df_ = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "text": texts}).assign(
        trunc_text=lambda d: d["text"].str[:120], r=0
    )

    if color_array is not None:
        if len(color_array) != X.shape[0]:
            raise ValueError(
                f"The number of color array ({len(color_array)}) should match X array ({X.shape[0]})."
            )
        df_ = df_.assign(color=color_array)

    if color_words:
        df_ = df_.assign(color="none")

        for w in color_words:
            predicate = df_["text"].str.lower().str.contains(w)
            df_ = df_.assign(color=lambda d: np.where(predicate, w, d["color"]))

    brush = alt.selection_interval()

    p1 = (
        alt.Chart(df_)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["text"],
        )
        .properties(width=350, height=350, title="embedding space")
        .add_params(brush)
    )

    if color_words:
        p1 = (
            alt.Chart(df_)
            .mark_circle(opacity=0.6, size=20)
            .encode(
                x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
                y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
                tooltip=["text"],
                color=alt.Color("color", sort=["none"] + color_words),
            )
            .properties(width=350, height=350, title="embedding space")
            .add_params(brush)
        )

    if color_array is not None:
        p1 = (
            alt.Chart(df_)
            .mark_circle(opacity=0.6, size=20)
            .encode(
                x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
                y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
                tooltip=["text"],
                color=alt.Color("color"),
            )
            .properties(width=350, height=350, title="embedding space")
            .add_params(brush)
        )

    p2 = (
        alt.Chart(df_)
        .mark_text()
        .encode(
            x=alt.X("r", axis=None),
            y=alt.Y("row_number:O", axis=None),
            text="trunc_text:N",
        )
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 18)
        .properties(title="text")
    )

    return (p1 | p2).configure_axis(grid=False).configure_view(strokeWidth=0)

def _single_scatter_chart(df_, idx, brush, title="embedding space", color_words=None, color_array=None):
    cols = ("x1:Q", "y1:Q") if idx == 1 else ("x2:Q", "y2:Q")
    if color_words:
        color=alt.Color("color", sort=["none"] + color_words)
    elif color_array:
        color=alt.Color("color")
    else:
        color=alt.condition(brush, 'id:O', alt.value('lightgray'), legend=None)
    return (
        alt.Chart(df_)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X(cols[0], axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y(cols[1], axis=None, scale=alt.Scale(zero=False)),
            tooltip=["text"],
            color=color,
        )
        .properties(width=350, height=350, title=title)
        .add_params(brush)
    )

def plot_text_comparison(X1, X2, texts, disable_warning=True, color_array=None, color_words=None):
    """
    Make a visualisation to help find clues in text data.

    Arguments:
        - `X1`: the numeric features, should be a 2D numpy array
        - `X2`: the numeric features, should be a 2D numpy array
        - `texts`: list of text data
        - `disable_warning`: disable the standard altair max rows warning
        - `color_words`: list of words to highlight
        - `color_array`: an array that represents color for the plot
    """
    if disable_warning:
        alt.data_transformers.disable_max_rows()

    if (len(texts) != X1.shape[0]) or (len(texts) != X2.shape[0]):
        raise ValueError(
            f"The number of text examples ({len(texts)}) should match X1/x2 array X1=({X1.shape[0]}) X2=({X2.shape[0]})."
        )
    
    df_ = pd.DataFrame({"x1": X1[:, 0], "y1": X1[:, 1], "x2": X2[:, 0], "y2": X2[:, 1], "text": texts}).assign(
        trunc_text=lambda d: d["text"].str[:120], r=0
    )
    
    if color_array is not None:
        if len(color_array) != X1.shape[0]:
            raise ValueError(
                f"The number of color array ({len(color_array)}) should match X array ({X.shape[0]})."
            )
        df_ = df_.assign(color=color_array)

    if color_words is not None:
        df_ = df_.assign(color="none")

        for w in color_words:
            predicate = df_["text"].str.lower().str.contains(w)
            df_ = df_.assign(color=lambda d: np.where(predicate, w, d["color"]))

    brush = alt.selection_interval()
    p1 = _single_scatter_chart(df_, 1, brush, title="embedding space X1", color_words=color_words, color_array=color_array)
    p2 = _single_scatter_chart(df_, 2, brush, title="embedding space X2", color_words=color_words, color_array=color_array)

    p3 = (
        alt.Chart(df_)
        .mark_text()
        .encode(
            x=alt.X("r", axis=None),
            y=alt.Y("row_number:O", axis=None),
            text="trunc_text:N",
        )
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 18)
        .properties(title="text")
    )

    return (p1 | p2 | p3).configure_axis(grid=False).configure_view(strokeWidth=0)