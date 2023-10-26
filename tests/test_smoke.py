import pytest
import numpy as np 
from cluestar import plot_text, plot_text_comparison


@pytest.mark.parametrize("s", [500, 1000, 2000])
def smoke_test_plot_text(s):
    texts = ["random {i}" for i in range(s)]
    X = np.random.normal(0, 1, (s, 2))
    plot_text(X, texts)


@pytest.mark.parametrize("s", [500, 1000, 2000])
def smoke_test_plot_text_comparison(s):
    texts = ["random {i}" for i in range(s)]
    X1 = np.random.normal(0, 1, (s, 2))
    X2 = np.random.normal(0, 1, (s, 2))
    plot_text_comparison(X1, X2, texts)
