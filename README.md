<img src="cluestar.png" width=175 align="right">

# cluestar

> Gain a clue by clustering!

This library contains visualisation tools that might help you
get started with classification tasks. The idea is that if you
can inspect clusters easily, you might gain a clue on what
good labels for your dataset might be!

## Install

```
python -m pip install "cluestar @ git+https://github.com/koaning/cluestar.git"
```
## Interactive Demo

You can see an interactive demo of the generated widgets [here]().

You can also toy around with the demo notebook found [here]().
## Usage

The first step is to encode textdata in two dimensions, like below.

```python
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD())

X = pipe.fit_transform(texts)
```

From here you can make an interactive chart via;

```python
from cluestar import plot_text

plot_text(X, texts)
```

The best results are likely found when you use
[umap](https://umap-learn.readthedocs.io/en/latest/)
together with something like
[universal sentence encoder](https://koaning.github.io/whatlies/api/language/universal_sentence/).

