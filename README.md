# RAN - Relational Asset Network

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
