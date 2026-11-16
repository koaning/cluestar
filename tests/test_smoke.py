import numpy as np
import pandas as pd
import pytest

from cluestar import ClueWidget, plot_text, plot_text_comparison


@pytest.mark.parametrize("s", [500, 1000, 2000])
def test_smoke_plot_text(s):
    texts = [f"random {i}" for i in range(s)]
    X = np.random.normal(0, 1, (s, 2))
    assert texts[0] in plot_text(X, texts).to_json()


@pytest.mark.parametrize("s", [500, 1000, 2000])
def test_smoke_plot_text_comparison(s):
    texts = [f"random {i}" for i in range(s)]
    X1 = np.random.normal(0, 1, (s, 2))
    X2 = np.random.normal(0, 1, (s, 2))
    assert texts[0] in plot_text_comparison(X1, X2, texts).to_json()


TEXTS = [
    "Delivery was late again",
    "late delivery, no driver",
    "driver never came (late)",
    "voucher code broken",
    "my voucher expired",
    None,
]
X = np.array([[5, 5], [5.5, 5.2], [4.5, 5.4], [1, 1], [-1, -1], [-0.5, 0]])

COLOR_OPTIONS = [
    {},
    {"color_array": [0, 0, 0, 1, 1, 1]},  # cluster ids -> categories
    {"color_array": np.array([0, 0, 0, 1, 1, 1])},
    {"color_array": np.linspace(0, 1, 6)},  # probabilities -> color scale
    {
        "color_array": pd.Series(
            ["a", "b", "a", "b", "a", "b"], index=[10, 11, 12, 13, 14, 15]
        )
    },
    {"color_words": ["late", "voucher"]},
    {"color_words": ["(late)", "c++", "?"]},  # regex characters are matched literally
]


@pytest.mark.parametrize("kwargs", COLOR_OPTIONS)
@pytest.mark.parametrize("comparison", [False, True])
def test_color_options(kwargs, comparison):
    chart = (
        plot_text_comparison(X, X * 2, TEXTS, **kwargs)
        if comparison
        else plot_text(X, TEXTS, **kwargs)
    )
    spec = chart.to_dict()
    assert {p["name"] for p in spec["params"]} >= {"brush"}


def test_categories_get_clickable_legend_but_scales_do_not():
    names = lambda c: {p["name"] for p in c.to_dict()["params"]}  # noqa: E731
    assert "legend" in names(plot_text(X, TEXTS, color_array=[0, 0, 0, 1, 1, 1]))
    assert "legend" in names(plot_text(X, TEXTS, color_words=["late"]))
    assert "legend" not in names(plot_text(X, TEXTS, color_array=np.linspace(0, 1, 6)))
    assert "legend" not in names(plot_text(X, TEXTS))


def test_top_words_panel_can_be_hidden():
    with_words = plot_text(X, TEXTS).to_json()
    without_words = plot_text(X, TEXTS, top_words=0).to_json()
    assert "top words" in with_words
    assert "top words" not in without_words


def test_accepts_dataframe_and_series_inputs():
    df = pd.DataFrame(X, columns=["a", "b"], index=range(100, 106))
    texts = pd.Series(TEXTS, index=range(200, 206))
    assert "voucher" in plot_text(df, texts).to_json()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"X": X[:5]}, "should match"),
        ({"X": X[:, 0]}, "2D array"),
        ({"color_array": [1] * 5}, "color_array"),
        ({"color_array": [1] * 6, "color_words": ["a"]}, "not both"),
    ],
)
def test_helpful_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plot_text(**{"X": X, "texts": TEXTS, **kwargs})


def test_comparison_color_array_length_error():
    with pytest.raises(ValueError, match="color_array"):
        plot_text_comparison(X, X, TEXTS, color_array=[1] * 5)


def _select(widget, brush=None, legend=None):
    """Simulate what the browser sends back to Python when a user makes a selection."""
    state = {"brush": {"value": brush, "store": []}}
    if legend is not None:
        state["legend"] = {
            "value": {"vlPoint": {"or": [{"color": c} for c in legend]}},
            "store": [],
        }
    widget._vl_selections = state


def test_widget_selection_round_trip():
    pytest.importorskip("anywidget")
    w = plot_text(X, TEXTS, color_words=["late", "voucher"], widget=True)
    assert isinstance(w, ClueWidget)
    assert len(w.selected()) == 6  # nothing selected -> everything

    _select(w, brush={"x1": [4, 6], "x2": [4, 6]})
    assert w.selected().index.tolist() == [0, 1, 2]
    assert list(w.selected().columns) == ["text", "x1", "x2", "color"]

    _select(w, legend=["voucher"])
    assert w.selected().index.tolist() == [3, 4]

    _select(w, brush={"x1": [4, 6], "x2": [4, 6]}, legend=["voucher"])
    assert w.selected().empty


def test_widget_comparison_selection_in_second_chart():
    pytest.importorskip("anywidget")
    w = plot_text_comparison(X, X * 2, TEXTS, widget=True)
    _select(w, brush={"x2": [6, 20], "y2": [6, 20]})
    assert w.selected().index.tolist() == [0, 1, 2]
