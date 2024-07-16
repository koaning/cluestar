import pathlib
from setuptools import setup, find_packages


base_packages = ["altair>=5.0.0", "numpy>=1.19.2"]

dev_packages = base_packages + [
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pre-commit>=2.2.0",
    "jupyterlab",
]


setup(
    name="cluestar",
    version="0.2.1",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Gain a clue by clustering!",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/koaning/cluestar/",
    project_urls={
        "Documentation": "https://github.com/koaning/cluestar/",
        "Source Code": "https://github.com/koaning/cluestar/",
        "Issue Tracker": "https://github.com/koaning/cluestar/issues",
    },
    install_requires=base_packages,
    extras_require={"base": base_packages, "dev": dev_packages},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
