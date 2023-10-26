import pytest
import numpy as np 
from cluestar import plot_text, plot_text_comparison


@pytest.mark.parametrize("s", [500, 1000, 2000])
def test_smoke_plot_text(s):
    texts = ["random {i}" for i in range(s)]
    X = np.random.normal(0, 1, (s, 2))
    assert texts[0] in plot_text(X, texts).to_json()


@pytest.mark.parametrize("s", [500, 1000, 2000])
def test_smoke_plot_text_comparison(s):
    texts = ["random {i}" for i in range(s)]
    X1 = np.random.normal(0, 1, (s, 2))
    X2 = np.random.normal(0, 1, (s, 2))
    assert texts[0] in plot_text_comparison(X1, X2, texts).to_json()
