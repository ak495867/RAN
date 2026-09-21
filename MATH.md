# RAN: Mathematical Specification and Evaluation Protocol

This document specifies, in mathematical terms, exactly what `ran_research.py` computes. The model, RAN, is a cross-sectional stock-ranking network: a causal dilated temporal convolution encodes each asset's recent feature history, a graph convolution over a correlation $k$-nearest-neighbour graph lets assets exchange information, and a linear head emits one score per asset per day. Scores are evaluated as rank predictors and traded as a dollar-neutral long-short book under a leak-free walk-forward protocol.

> **Scope note.** The implementation contains no attention layer; "RAN" is the name used in the code. Everything below describes the code as written, including its limitations (Section 15).

---

## 1. Purpose and pipeline

At the close of day $t$, given $A$ assets, the model produces a score vector $s_t\in\mathbb{R}^{A}$. Only the **cross-sectional ordering** of $s_t$ matters. The task is to rank assets by their forward $H$-day return, where the return is earned after a one-day execution delay.

$$
\text{prices}
\rightarrow \text{causal features}
\rightarrow \text{train-only normalization}
\rightarrow \text{60-day windows}
\rightarrow \text{temporal encoder}
\rightarrow \text{graph convolution}
\rightarrow \text{score}
\rightarrow \text{long-short book}
\rightarrow \text{statistics}.
$$

Notation used throughout:

| Symbol | Meaning | Code |
| --- | --- | --- |
| $A$ | number of assets | `len(tickers)` |
| $T$ | number of trading days (rows) | `len(dates)` |
| $F$ | number of features per asset (57) | `len(fnames)` |
| $L$ | window length (60) | `seq_len` |
| $H$ | forecast/holding horizon in days | `horizons` |
| $\ell_x$ | execution lag (1) | `exec_lag` |
| $K$ | neighbours per node in the graph (10) | `knn_k` |
| $d$ | hidden width (32) | `hidden` |
| $q$ | fraction of assets on each side of the book (0.10) | `top_frac` |

---

## 2. Data, universe and label

### 2.1 Panel

For asset $i$ and day $t$ the data are adjusted open, high, low, close and volume, $(O_{i,t},H_{i,t},L_{i,t},C_{i,t},V_{i,t})$, downloaded with `yfinance` (`auto_adjust=True`). All assets share a common calendar; a ticker is kept only if it covers at least a fraction `min_history_frac` (0.995) of that calendar, starts within its first 10 rows, and has strictly positive prices. Short gaps are forward-filled (at most 3 days). The daily return is

$$
r_{i,t}=\frac{C_{i,t}}{C_{i,t-1}}-1 .
$$

### 2.2 Label with execution lag

A signal formed at the close of day $t$ is executed at the close of day $t+\ell_x$ and held for $H$ days. The realised target is

$$
y_{i,t}^{(H)}=\frac{C_{i,t+\ell_x+H}}{C_{i,t+\ell_x}}-1 ,
$$

and is undefined (`NaN`) when $t+\ell_x+H>T-1$. The **label span** is $\sigma=\ell_x+H$: the label of sample $t$ is fully determined by prices up to index $t+\sigma$.

---

## 3. Feature construction

All features at row $t$ use data at or before $t$. There are $F=57$ features in four groups.

### 3.1 Per-asset features (27)

Drop the asset index $i$. For lags $\ell\in\{1,3,5,10,20\}$:

$$
\mathrm{ret}_\ell=\frac{C_t}{C_{t-\ell}}-1,\qquad
\mathrm{high\_ret}_\ell=\frac{H_t}{H_{t-\ell}}-1,\qquad
\mathrm{low\_ret}_\ell=\frac{L_t}{L_{t-\ell}}-1 \quad(15\text{ features}).
$$

Realised volatility over $w\in\{5,20,60\}$ days (3 features):

$$
\mathrm{vol}_w=\operatorname{sd}\big(r_{t-w+1},\ldots,r_t\big).
$$

RSI (1 feature), with $\Delta C_t=C_t-C_{t-1}$, $U_t=\max(\Delta C_t,0)$, $D_t=\max(-\Delta C_t,0)$ and $\mathrm{MA}_{14}$ a 14-day rolling mean:

$$
\mathrm{rsi}_t=100-\frac{100}{1+\mathrm{MA}_{14}(U)_t\,/\,\big(\mathrm{MA}_{14}(D)_t+10^{-6}\big)} .
$$

Order-flow proxy (1 feature):

$$
\mathrm{ofi}_t=\frac{\mathrm{MA}_5\big(\Delta C\cdot\Delta V\big)_t}{\mathrm{MA}_5(V)_t+10^{-6}} .
$$

Momentum-type features (2 features), sums of daily returns:

$$
\mathrm{mom\_60\_5}_t=\sum_{j=0}^{59}r_{t-j}-\sum_{j=0}^{4}r_{t-j},\qquad
\mathrm{mom\_120\_20}_t=\sum_{j=0}^{119}r_{t-j}-\sum_{j=0}^{19}r_{t-j}.
$$

Intraday structure (4 features), with $\mathrm{ov}_t=O_t/C_{t-1}-1$ and $\mathrm{id}_t=C_t/O_t-1$:

$$
\mathrm{overnight}_t=\mathrm{ov}_t,\quad
\mathrm{intraday}_t=\mathrm{id}_t,\quad
\mathrm{ov\_20}_t=\mathrm{MA}_{20}(\mathrm{ov})_t,\quad
\mathrm{id\_20}_t=\mathrm{MA}_{20}(\mathrm{id})_t .
$$

Amihud illiquidity (1 feature):

$$
\mathrm{amihud}_t=\mathrm{MA}_{20}\!\left(\frac{\lvert r_t\rvert}{C_tV_t}\right).
$$

### 3.2 Cross-sectional features (30)

Let $\mathcal{J}$ be the 22 per-asset features whose names do not start with `ret_`. For $j\in\mathcal{J}$, the **cross-sectional $z$-score** on day $t$ is

$$
\mathrm{cs\_z}_{i,t,j}=\frac{x_{i,t,j}-\bar x_{t,j}}{\operatorname{sd}_i(x_{\cdot,t,j})+10^{-9}},
$$

where mean and standard deviation are taken across assets on that day (22 features). For $j\in\{\mathrm{ret}_5,\mathrm{ret}_{20},\mathrm{mom\_60\_5},\mathrm{mom\_120\_20}\}$ the **cross-sectional percentile rank** $\mathrm{cs\_rank}_{i,t,j}\in(0,1]$ is added (4 features).

With the equal-weight market return $m_t=\frac1A\sum_i r_{i,t}$, the **market-relative momentum** for $\ell\in\{5,20,60\}$ is

$$
\mathrm{excess\_mom}_\ell=\sum_{j=0}^{\ell-1}\big(r_{t-j}-m_{t-j}\big)\quad(3\text{ features}),
$$

and the 60-day rolling **beta** to the equal-weight market is

$$
\beta_{60,t}=\frac{\widehat{\operatorname{Cov}}_{60}(r_t,m_t)}{\widehat{\operatorname{Var}}_{60}(m_t)+10^{-9}}\quad(1\text{ feature}).
$$

The feature count is $27+22+4+3+1=57$. Rows before all features are defined (about the first 120 days) are discarded (warm-up cut).

### 3.3 Normalization

Let $\mathcal{R}_w$ be the block of feature rows used in walk-forward window $w$ (Section 8). Per-feature mean $\mu_j$ and standard deviation $s_j$ are estimated on $\mathcal{R}_w$ **only** (pooled over rows and assets, ignoring `NaN`), and applied to every row, including test rows:

$$
\tilde x_{i,t,j}=\operatorname{clip}\!\Big(\frac{x_{i,t,j}-\mu_j}{s_j+10^{-6}},\,-5,\,5\Big),\qquad
\text{remaining }\mathrm{NaN}\mapsto 0 .
$$

### 3.4 Target standardization

The network is trained on the per-day cross-sectionally standardised return

$$
\tilde y_{i,t}=\frac{y_{i,t}-\bar y_t}{\operatorname{sd}_i(y_{\cdot,t})+10^{-6}} ,
$$

which removes the market component of the target. Evaluation always uses raw $y$.

---

## 4. Tensors

For the window ending at day $t$ (rows $t-L+1,\ldots,t$, inclusive of $t$):

$$
X_t\in\mathbb{R}^{A\times L\times F},\qquad X_{t}[i,\tau,:]=\tilde x_{i,\,t-L+1+\tau},\quad \tau=0,\ldots,L-1 .
$$

For a batch of $B$ dates the input is $\mathbb{R}^{B\times A\times L\times F}$. Windows are gathered on demand from a single $(T,A,F)$ panel, so no $(N,A,L,F)$ tensor is ever materialised.

---

## 5. Correlation kNN graph

At day $t$, standardise each asset's last $L$ returns, $z_{i}=\big(r_{i,t-L+1:t}-\bar r_i\big)/(\operatorname{sd}(r_{i,t-L+1:t})+10^{-8})$, and form the sample correlation matrix

$$
C_t=\frac1L\,Z_t^\top Z_t\in\mathbb{R}^{A\times A}.
$$

For each node $i$, let $\mathcal{N}_K(i)$ be the $K$ assets with the largest $C_t[i,j]$, $j\ne i$. The raw adjacency keeps only non-negative correlations:

$$
\breve A_t[i,j]=\begin{cases}\max\big(C_t[i,j],0\big),& j\in\mathcal{N}_K(i),\\ 0,&\text{otherwise.}\end{cases}
$$

It is then symmetrised and row-normalised (rows summing to zero are left as zero):

$$
\bar A_t=\tfrac12\big(\breve A_t+\breve A_t^\top\big),\qquad
\tilde A_t=D_t^{-1}\bar A_t,\quad D_t=\operatorname{diag}\big(\textstyle\sum_j \bar A_t[i,j]\big).
$$

Only the indices and weights of the $K$ neighbours are stored, as `int16`/`float32` arrays of shape $(T,A,K)$; the dense $A\times A$ matrix is rebuilt per batch. The graph at $t$ uses returns at or before $t$ and is therefore causal.

---

## 6. Temporal encoder

Each asset's window is encoded independently (weights shared across assets) by three causal dilated 1-D convolutions with kernel size 3 and dilations $1,2,4$. Let $u^{(0)}_\tau=X_t[i,\tau,:]\in\mathbb{R}^F$. For $m=1,2,3$ with dilation $d_m$:

$$
u^{(m)}_\tau=\operatorname{ReLU}\!\Big(b_m+\sum_{k=0}^{2}W_m^{(k)}\,u^{(m-1)}_{\tau-(2-k)d_m}\Big),\qquad u_{\tau}=0\text{ for }\tau<0 ,
$$

with $W_1^{(k)}\in\mathbb{R}^{d\times F}$ and $W_{2,3}^{(k)}\in\mathbb{R}^{d\times d}$. In the code, the convolution is padded symmetrically and the last $2d_m$ outputs are removed, which makes output $\tau$ depend only on inputs $\le\tau$. The receptive field is $1+2(1+2+4)=15$ timesteps. The window is summarised by mean pooling:

$$
h_i^{(0)}=\frac1L\sum_{\tau=0}^{L-1}u^{(3)}_{i,\tau}\in\mathbb{R}^{d}.
$$

---

## 7. Graph convolution, score head and loss

### 7.1 Graph convolution

Stack the node embeddings as $H^{(0)}\in\mathbb{R}^{A\times d}$. Add self-loops and renormalise the rows:

$$
\hat A_t=\Big(\operatorname{diag}\big((\tilde A_t+I)\mathbf 1\big)\Big)^{-1}(\tilde A_t+I).
$$

The graph layer is a residual one-hop aggregation (cf. Kipf and Welling):

$$
H^{(1)}=\operatorname{ReLU}\!\big(\hat A_t\,H^{(0)}W_g+\mathbf 1 b_g^\top\big)+H^{(0)},\qquad W_g\in\mathbb{R}^{d\times d}.
$$

The **ablation `ran_nograph`** sets $H^{(1)}=H^{(0)}$.

### 7.2 Score

After dropout with rate $p=0.5$ (training only), a shared linear head gives each asset a scalar score:

$$
s_{i,t}=w^\top\operatorname{Dropout}\!\big(H^{(1)}_{i}\big)+b .
$$

With $F=57$ and $d=32$ the model has $5504+3104+3104+1056+33=12{,}801$ trainable parameters.

### 7.3 Loss

For a batch of dates, with $s_t,\tilde y_t\in\mathbb{R}^A$,

$$
\mathcal{L}=\underbrace{\frac{1}{BA}\sum_{t,i}\big(s_{i,t}-\tilde y_{i,t}\big)^2}_{\text{MSE}}
+\lambda\Big(1-\frac1B\sum_{t}\rho\big(s_t,\tilde y_t\big)\Big),\qquad \lambda=2,
$$

where $\rho$ is the Pearson correlation across assets on day $t$, computed after subtracting each day's cross-sectional mean:

$$
\rho(s,y)=\frac{\sum_i(s_i-\bar s)(y_i-\bar y)}{\lVert s-\bar s\rVert_2\,\lVert y-\bar y\rVert_2+10^{-8}} .
$$

The second term is a differentiable surrogate for the information coefficient (IC).

### 7.4 Optimization

Adam ($\eta=10^{-3}$, coupled $L_2$ weight decay $10^{-3}$), cosine-annealed learning rate over 15 epochs, batches of 32 dates in random order, gradient-norm clipping at 1, and Gaussian input noise $\tilde x\leftarrow\tilde x+0.1\,\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$. After every epoch the mean daily **validation rank IC** is computed; the parameters with the best validation IC are kept, and training stops after 3 epochs without improvement. The final score of a window is the average over $S$ independently seeded networks:

$$
\bar s_{i,t}=\frac1S\sum_{k=1}^{S}s^{(k)}_{i,t}.
$$

---

## 8. Walk-forward protocol

Let $s_0=L$ be the first usable row, $W_{\mathrm{tr}}$ the training window (1008 rows) and $\Delta$ the step (126 rows, about 6 months). For test start $t_s=s_0+W_{\mathrm{tr}}+k\Delta$, $k=0,1,\ldots$ the sets are:

| Set | Definition |
| --- | --- |
| candidates | $\mathcal{C}=\{\,i:\ \max(s_0,t_s-W_{\mathrm{tr}})\le i\le t_s-\sigma,\ y_i\text{ defined}\,\}$ |
| validation | the last 20% of $\mathcal{C}$, with first index $v_0$ |
| fit | $\{\,i\in\mathcal{C}:\ i\le v_0-\sigma\,\}$ (purged so no fit label reaches into validation) |
| test | $\{\,t\in[t_s,\,t_s+\Delta):\ y_t\text{ defined}\,\}$ |

The window is skipped if the fit set has fewer than `min_train` (500) rows. Normalization statistics (Section 3.3) come from feature rows in $[\,\text{lo}-L+1,\ \max\mathcal{C}\,]$. The network is trained on the fit set, early-stopped on validation, and applied without change to the whole test block. All models (RAN, ablation, ridge, factor baselines) are evaluated on **identical** windows, which makes paired comparisons valid.

**Frozen holdout.** In the `final` stage one additional model per learned method is trained on the $W_{\mathrm{tr}}$ rows before `holdout_start` and applied, without retraining, to every later row.

**Stage separation.** With `--stage dev` the panel is truncated at `holdout_start`, so holdout data are never loaded.

### Proposition 1 (no look-ahead)

For every window, every quantity used to train, validate or normalise is computable from prices available at the close of day $t_s$.

**Proof.** A fit or validation sample $i\in\mathcal{C}$ has $i\le t_s-\sigma$, so its label uses prices at indices $\le i+\sigma\le t_s$. Its features and graph use rows $\le i$. Normalization statistics use rows $\le\max\mathcal{C}\le t_s-\sigma$. Test features at $t\ge t_s$ use rows $\le t$, and the position is opened at $t+\ell_x>t$. The purge $i\le v_0-\sigma$ ensures no fit label depends on prices after $v_0$, the first validation date. $\square$

### Proposition 2 (causal encoder)

Output $\tau$ of the temporal encoder depends only on inputs $\tau'\le\tau$.

**Proof.** In each layer, tap $k$ reads position $\tau-(2-k)d_m\le\tau$, and positions $<0$ are zero. Composition of causal maps is causal. $\square$

---

## 9. Baselines

All baselines score the same rows as RAN.

**Ridge.** On the normalised last-day feature vector $x_{i,t}\in\mathbb{R}^F$, with pooled centred data,

$$
\hat\beta_\alpha=\big(X_c^\top X_c+\alpha\,n\,I\big)^{-1}X_c^\top\tilde y_c,\qquad s_{i,t}=\hat\beta^\top x_{i,t}.
$$

$\alpha$ is chosen from $\{10^{-4},\ldots,1\}$ by validation rank IC and the model is refit on fit plus validation.

**Momentum:** $s_{i,t}=\mathrm{mom\_120\_20}_{i,t}$. **Reversal:** $s_{i,t}=-\mathrm{ret}_{5,i,t}$. **Random:** $s_{i,t}\sim\mathcal N(0,1)$ i.i.d. (a null control).

---

## 10. Portfolio construction and costs

Let $k=\max\big(1,\operatorname{round}(qA)\big)$. At each day $t$ with valid scores, a **tranche** is opened with weights

$$
w_{i,t}=\begin{cases}+\dfrac{1}{2k},& i\text{ among the }k\text{ highest scores},\\[4pt] -\dfrac{1}{2k},& i\text{ among the }k\text{ lowest scores},\\[4pt] 0,&\text{otherwise,}\end{cases}
\qquad\sum_i w_{i,t}=0,\quad\sum_i\lvert w_{i,t}\rvert=1 .
$$

Each tranche is held $H$ days, so at any time $H$ tranches are alive (a **staggered book**). This removes the dependence of results on a single rebalance start day and uses every row.

Gross return, turnover, cost and borrow of tranche $t$ (turnover is measured against the tranche opened $H$ days earlier, which is the position it replaces):

$$
g_t=\sum_i w_{i,t}\,y_{i,t},\qquad
\tau_t=\sum_i\big\lvert w_{i,t}-w_{i,t-H}\big\rvert,
$$

$$
r_t=g_t-c\,\tau_t-b_H,\qquad c=\text{cost\_bps}\times10^{-4},\qquad
b_H=\tfrac12\cdot\text{borrow\_bps}\times10^{-4}\cdot\frac{H}{252}.
$$

Here $c$ is per unit of traded notional (a complete replacement has $\tau_t=2$) and $b_H$ charges the short half of the book. The first tranche of a period is charged full entry cost.

Annualised statistics use $252/H$ periods per year:

$$
\mathrm{SR}=\frac{\bar r}{\operatorname{sd}(r)}\sqrt{\frac{252}{H}},\qquad
\text{ann. return}=\bar r\cdot\frac{252}{H},\qquad
\text{ann. vol}=\operatorname{sd}(r)\sqrt{\frac{252}{H}} .
$$

Reported turnover is $\overline{\tau_t}/2$ (fraction of the book replaced per rebalance). For the equity curve and maximum drawdown, the return of tranche $t$ is spread evenly over its holding period, $E_t=\prod_{u\le t}\big(1+r_u/H\big)$; this is an approximation to a daily-marked P&L series.

### Proposition 3 (dollar neutrality)

Each tranche has zero net exposure and unit gross exposure, and its gross return is invariant to any strictly increasing transformation of the scores.

**Proof.** $\sum_i w_{i,t}=k\cdot\frac{1}{2k}-k\cdot\frac{1}{2k}=0$ and $\sum_i\lvert w_{i,t}\rvert=2k\cdot\frac{1}{2k}=1$. The weights depend on $s_t$ only through which assets are among its $k$ largest and $k$ smallest entries, which is unchanged by strictly increasing maps. $\square$

**Consequence.** Score scale and calibration are irrelevant for performance; only the ordering matters.

---

## 11. Predictive-accuracy statistics

### 11.1 Rank IC

On each day the Spearman correlation between score and realised return is

$$
\mathrm{IC}_t=\rho\big(\operatorname{rank}(s_t),\operatorname{rank}(y_t)\big),\qquad
\overline{\mathrm{IC}}=\frac1n\sum_t\mathrm{IC}_t,\qquad
\mathrm{ICIR}=\frac{\overline{\mathrm{IC}}}{\operatorname{sd}(\mathrm{IC}_t)} .
$$

Because $\mathrm{IC}_t$ depends only on ranks, it is invariant to increasing transformations of $s_t$ and to the cross-sectional standardisation of $y_t$.

### 11.2 Newey-West test for the mean IC

Labels overlap for $H>1$, so consecutive $\mathrm{IC}_t$ are serially correlated. With $\hat\gamma_\ell$ the lag-$\ell$ sample autocovariance of $\mathrm{IC}_t$ and bandwidth $H$,

$$
\hat\sigma^2_{\mathrm{NW}}=\hat\gamma_0+2\sum_{\ell=1}^{H}\Big(1-\frac{\ell}{H+1}\Big)\hat\gamma_\ell,\qquad
t_{\mathrm{NW}}=\frac{\overline{\mathrm{IC}}}{\sqrt{\hat\sigma^2_{\mathrm{NW}}/n}} .
$$

The reported $p$-value is one-sided, $1-\Phi(t_{\mathrm{NW}})$, against $H_1:\ \mathbb{E}[\mathrm{IC}]>0$ (normal approximation).

### 11.3 Sector neutralisation (variant)

For every sector with at least 3 assets, the daily sector mean of the score is subtracted, $s_{i,t}\leftarrow s_{i,t}-\bar s_{\mathrm{sec}(i),t}$; smaller sectors are left unchanged. Both the IC and the book are recomputed on the neutralised scores and reported as the `sector-neutral` variant. For the multi-asset universe the asset class plays the role of the sector.

### 11.4 Decile profile

With $\operatorname{ord}_{i,t}\in\{0,\ldots,A-1\}$ the ordinal rank of $s_{i,t}$, the decile is $\lfloor10\operatorname{ord}_{i,t}/A\rfloor$. The reported quantity is the mean forward return per decile; a monotone profile indicates a usable ranking beyond the extreme deciles.

---

## 12. Inference on the strategy

### 12.1 Stationary bootstrap for the Sharpe ratio

Let $r_1,\ldots,r_n$ be the tranche net returns. Following Politis and Romano, a bootstrap sample draws indices $I_1\sim\mathrm{Unif}\{1,\ldots,n\}$ and, for $u\ge2$,

$$
I_u=\begin{cases}\text{new }\mathrm{Unif}\{1,\ldots,n\}&\text{with probability }p=1/L_b,\\ (I_{u-1}\bmod n)+1&\text{otherwise,}\end{cases}\qquad L_b=\max(2H,5).
$$

The Sharpe ratio is recomputed on $(r_{I_1},\ldots,r_{I_n})$ for $B$ draws. The reported interval is the 2.5%--97.5% percentile interval, and "$P(\mathrm{SR}\le0)$" is the fraction of draws with $\mathrm{SR}\le0$. The mean block length $2H$ covers the serial dependence induced by overlapping holding periods.

### 12.2 Paired bootstrap for differences

To compare RAN with a comparator, the **same** index draws are applied to both return (or IC) series, and the statistic difference $\mathrm{SR}_{\mathrm{RAN}}-\mathrm{SR}_{\mathrm{cmp}}$ (or $\overline{\mathrm{IC}}_{\mathrm{RAN}}-\overline{\mathrm{IC}}_{\mathrm{cmp}}$) is computed per draw. This preserves the cross-model dependence, which is large because all models trade the same assets on the same days.

### 12.3 Probabilistic Sharpe Ratio

On the non-overlapping subsample $r_1,r_{1+H},r_{1+2H},\ldots$ of size $n$, with per-period Sharpe $\widehat{\mathrm{SR}}$, skewness $\hat\gamma_3$ and (non-excess) kurtosis $\hat\gamma_4$, the probability that the true Sharpe exceeds $\mathrm{SR}^*=0$ is (Bailey and Lopez de Prado)

$$
\mathrm{PSR}=\Phi\!\left(\frac{\big(\widehat{\mathrm{SR}}-\mathrm{SR}^*\big)\sqrt{n-1}}{\sqrt{1-\hat\gamma_3\widehat{\mathrm{SR}}+\frac{\hat\gamma_4-1}{4}\widehat{\mathrm{SR}}^2}}\right).
$$

### 12.4 Permutation test

Restrict to non-overlapping rows $t\in\{o,o+H,o+2H,\ldots\}$ for three start offsets $o\in\{0,\lfloor(H-1)/2\rfloor,H-1\}$. Under the null that scores carry no information, returns are exchangeable across assets within a day. The test statistic is the gross annualised Sharpe of the long-short book. The null distribution is generated by drawing, for every row, a uniformly random permutation of assets and using its first $k$ entries as the long side and the next $k$ as the short side; with $B$ draws,

$$
p=\frac{1+\#\{b:\ \mathrm{SR}^{(b)}_{\mathrm{null}}\ge\mathrm{SR}_{\mathrm{obs}}\}}{1+B},
$$

and the median over offsets is reported. Non-overlapping rows are used because independent within-day shuffles would understate the null variance if labels overlapped.

### Proposition 4 (null expectation)

If returns are exchangeable across assets within each day, then a random long-short book has $\mathbb{E}[g_t]=0$ and random scores have $\mathbb{E}[\mathrm{IC}_t]=0$.

**Proof.** Under exchangeability every asset is equally likely to fall in the long set or in the short set, so $\mathbb{E}\big[\frac1k\sum_{i\in\text{long}}y_i\big]=\mathbb{E}\big[\frac1k\sum_{i\in\text{short}}y_i\big]=\mathbb{E}[\bar y_t]$, and their difference vanishes. Likewise the ranks of random scores are uniformly distributed permutations independent of $\operatorname{rank}(y_t)$, so their correlation has mean zero. $\square$

### 12.5 Multiple testing

For each evaluation period, the one-sided Newey-West $p$-values of all (model, horizon) pairs (excluding the random control) are adjusted with Holm's step-down procedure: with ordered $p_{(1)}\le\cdots\le p_{(m)}$,

$$
\tilde p_{(j)}=\max_{i\le j}\min\big\{1,\ (m-i+1)\,p_{(i)}\big\}.
$$

Every model and horizon that is run is reported, so the family size is not selectively reduced.

---

## 13. Reported periods

| Period | Rows | Available in |
| --- | --- | --- |
| `full` | all out-of-sample rows | both stages |
| `dev` | dates before `holdout_start` | `final` only |
| `holdout` | dates from `holdout_start` on | `final` only |

The design decisions (horizon, `top_frac`, universe, hyperparameters, primary variant) are to be fixed using the `dev` stage; the `final` stage is run once.

---

## 14. Summary of components

| Component | Mathematical role | Output |
| --- | --- | --- |
| Feature engineering | Causal per-asset and cross-sectional covariates | $x_{i,t}\in\mathbb{R}^{57}$ |
| Normalization | Train-window affine map, clip at $\pm5$ | $\tilde x_{i,t}$ |
| Target standardization | Per-day demean and scale | $\tilde y_{i,t}$ |
| Window builder | Trailing 60-day history | $X_t\in\mathbb{R}^{A\times60\times57}$ |
| Correlation kNN graph | Data-driven asset relations | $\tilde A_t\in\mathbb{R}^{A\times A}$ |
| Dilated causal CNN | Temporal feature extraction | $H^{(0)}\in\mathbb{R}^{A\times32}$ |
| Graph convolution | Neighbour aggregation with residual | $H^{(1)}\in\mathbb{R}^{A\times32}$ |
| Linear head | Score per asset | $s_t\in\mathbb{R}^A$ |
| MSE + IC loss | Fit and rank-alignment objective | scalar |
| Validation early stopping | Model selection without test data | best epoch |
| Seed ensemble | Variance reduction | $\bar s_t$ |
| Long-short tranche book | Dollar-neutral trading rule | $r_t$ |
| Newey-West, bootstrap, PSR | Inference under overlap | $p$, CI |
| Permutation test | Distribution-free null | $p$ |
| Holm | Familywise error control | $\tilde p$ |

The conceptual chain is **representation** (features, encoder, graph), **scoring** (head), **ranking-based decision** (long-short book), and **inference** (statistics that respect overlap and multiplicity). None of these stages, by itself, proves the existence of an economic edge.

---

## 15. Known limitations

| Issue | Why it matters | Mitigation |
| --- | --- | --- |
| Universe is current index membership | Survivorship bias inflates returns of the full-history filter | State it; supply a point-in-time universe with `--universe csv` |
| Free `yfinance` data | Adjustments, delistings and vendor errors are not audited | Report the data hash from `manifest.json`; cross-check with a second source |
| Close-to-close execution with a one-day delay | No intraday fills, market impact or capacity limits | Cost sensitivity table; treat results as gross-of-impact |
| Flat cost and borrow assumptions | Real costs vary by name, size and regime | Use `cost_grid` sensitivity |
| Coupled $L_2$ weight decay in Adam | Not equivalent to decoupled weight decay | Note in the paper, or switch to AdamW as an ablation |
| Mean pooling over zero-padded convolution outputs | Early timesteps in a window see padding artefacts | Ablate window length and pooling |
| Correlation graph is contemporaneous and linear | Captures co-movement, not lead-lag or fundamental links | Ablate `knn_k`; compare with sector graph |
| Graph edges from a 60-day sample correlation | Noisy for $L\ll A$ | Ablate `seq_len` |
| Equity curve spreads tranche returns evenly over $H$ days | Maximum drawdown is approximate | Prefer Sharpe, IC and their intervals |
| Family of tests includes only what was run | Untracked earlier experiments inflate significance | Log every configuration tried; Holm covers only this run |
| Percentile bootstrap and normal Newey-West $p$-values | Small-sample coverage is approximate | Read alongside `N indep.` (rows$/H$); distrust cells with few independent trades |
| Model selection horizon and `top_frac` | Choosing them on `final` would contaminate the holdout | Use `--stage dev` for all choices |

---

## References

1. Kipf and Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
2. Bai, Kolter and Koltun, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271).
3. Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
4. Newey and West (1987), A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix, *Econometrica* 55(3).
5. Politis and Romano (1994), The stationary bootstrap, *Journal of the American Statistical Association* 89(428).
6. Bailey and Lopez de Prado (2012), The Sharpe ratio efficient frontier, *Journal of Risk* 15(2).
7. Holm (1979), A simple sequentially rejective multiple test procedure, *Scandinavian Journal of Statistics* 6(2).
8. Amihud (2002), Illiquidity and stock returns: cross-section and time-series effects, *Journal of Financial Markets* 5(1).