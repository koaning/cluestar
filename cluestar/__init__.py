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

    brush = alt.selection(type="interval")

    p1 = (
        alt.Chart(df_)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=alt.X("x1", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y("x2", axis=None, scale=alt.Scale(zero=False)),
            tooltip=["text"],
        )
        .properties(width=350, height=350, title="embedding space")
        .add_selection(brush)
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
            .add_selection(brush)
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
            .add_selection(brush)
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
