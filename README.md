# RAN - Relational Asset Network

[![Paper](https://img.shields.io/badge/Paper-PDF-8A2BE2?style=for-the-badge&logo=readdotcv&logoColor=white)](https://github.com/ak495867/RAN/PAPER/RAN.pdf)
[![GitHub last commit](https://img.shields.io/github/last-commit/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/ak495867/HPT-DTE-CAFP?style=for-the-badge)](https://github.com/ak495867/HPT-DTE-CAFP)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![yfinance](https://img.shields.io/badge/yfinance-0B0B0B?style=for-the-badge)](https://github.com/ranaroussi/yfinance)


This repository contains a reproducible empirical study of the **Relational Asset Network (RAN)**, a temporal convolutional graph model for cross-sectional asset ranking. RAN combines causal temporal convolutions with a dynamic correlation graph to learn asset representations and generate multi-horizon rankings.

## Current evidence

The development study uses **56 instruments across 7 asset classes**, 57 causal features, a 60-session lookback, and walk-forward evaluation across 10D, 20D, and 60D horizons.

| Horizon | Rank IC | Net Sharpe |
| ------- | ------: | ---------: |
| 10D     |  0.0923 |      0.556 |
| 20D     |  0.1159 |      0.521 |
| 60D     |  0.0807 |      0.156 |

The archived results are development results; the configured post-2023 holdout is not included.


## Experimental design

The model uses a causal temporal CNN followed by dynamic graph convolution. Portfolio construction ranks assets cross-sectionally, taking long positions in the top 10% and short positions in the bottom 10%, with one-day execution lag and transaction costs included.

## Limitations

This release reports development-period evidence only. Further validation on the frozen post-2023 holdout, alternative universes, costs, turnover, and robustness tests is required before drawing broader conclusions.
