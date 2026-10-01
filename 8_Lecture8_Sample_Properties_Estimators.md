# Sample Properties & Estimators

> **Lecture theme:** How do we say anything about a *population* when all we ever have is a *sample* drawn from it? This lecture builds up the machinery of **estimators** — the sample mean, sample variance, sample standard deviation, sample covariance, and finally the **sample covariance matrix** — and asks, for each one, the key question: *is it biased or unbiased?*

---

## 1. The core problem: we don't know the underlying distribution

In an ideal world we would know the exact probability density function (pdf) $f(x)$ of whatever quantity we are measuring. If we knew $f(x)$, everything about the population would follow by direct calculation. For example, the population mean would be

$$
E(X) = \mu = \int_{-\infty}^{\infty} x\, f(x)\, dx.
$$

**But in practice we almost never know $f(x)$.** We are handed a dataset — a finite collection of measurements — with no label telling us which distribution produced it.

The consequence is unavoidable:

> Because $f(x)$ is unknown, we **cannot compute** the true population quantities $\mu$ (mean), $\sigma^2$ (variance), $\sigma$ (standard deviation), etc. directly.

So instead of *computing* these population properties, we must **estimate** them from the sample we do have. This is the entire motivation for the rest of the lecture.

---

## 2. Estimators, estimands, and bias

### What is an estimator?

> An **estimator** $\hat{\theta}$ is an *expression* (a formula, a recipe) used to estimate a statistical quantity $\theta$ — for example the population mean or the population variance.

A little vocabulary that keeps everything straight:

| Symbol | Name | Meaning |
|--------|------|---------|
| $\theta$ | **estimand** | the *true* population quantity we wish we knew (e.g. $\mu$, $\sigma^2$) |
| $\hat{\theta}$ | **estimator** | the formula we compute from the sample to approximate $\theta$ |
| $E(\hat{\theta})$ | expected value of the estimator | the average value the estimator would take over infinitely many samples |

The hat ("$\hat{\;}$") is the universal notation for "an estimate of." So $\hat{\theta}$ is our best guess at $\theta$, built out of data.

### The bias of an estimator

Because an estimator is computed from *random* data, it is itself a random quantity — a different sample would give a slightly different value. A fair way to judge whether an estimator is "aimed correctly" is to ask: *on average, does it land on the true value?* This is captured by the **bias**:

$$
\boxed{\,B(\hat{\theta}) = E(\hat{\theta}) - \theta\,}
$$

Reading the formula piece by piece:

- $B(\hat{\theta})$ — the **bias of the estimator**
- $E(\hat{\theta})$ — the **expected value of the estimator** (its long-run average)
- $\theta$ — the **true value of the estimand**

**Physical intuition:** think of an archer aiming at a target. Each arrow is one sample. The bias is the gap between the *center of the cluster of arrows* and the *bullseye*. If the cluster is centered on the bullseye, the archer has no systematic error — even though individual arrows scatter.

### Two cases

This single definition splits estimators into two families:

**Unbiased estimator** — $B(\hat{\theta}) = 0$, i.e. $E(\hat{\theta}) = \theta$.
> The estimator, on average, hits the true value exactly. There is no systematic error.

**Biased estimator** — $B(\hat{\theta}) \neq 0$, i.e. $E(\hat{\theta}) \neq \theta$.
> The estimator systematically over- or under-shoots the true value, no matter how the noise averages out.

Note that "unbiased" does **not** mean "correct for any single sample" — it only means correct *on average*. Individual estimates still scatter around the truth.

---

## 3. Sample mean → an *unbiased* estimator of the population mean

### Definition

Given $n$ measurements that together make up a sample, the **sample mean** is just the arithmetic average of the values obtained for the random variable.

For $n$ values $x_1, x_2, \ldots, x_n$:

$$
\bar{x} = \frac{1}{n}\left(\sum_{i=1}^{n} x_i\right)
$$

This is our estimator $\hat{\theta}$ for the estimand $\theta = \mu$, where the true population mean is

$$
\bar{X} = \mu = \int_{-\infty}^{\infty} x\, f(x)\, dx \quad \text{(the true value of the population mean — the estimand).}
$$

### Proof that the sample mean is unbiased

We compute the expected value of the estimator and check whether it equals $\mu$.

$$
E(\bar{x}) = E\!\left(\frac{1}{n}\sum_{i=1}^{n} x_i\right)
$$

Pull the constant $\tfrac{1}{n}$ out and use linearity of expectation (the expectation of a sum is the sum of expectations):

$$
= \frac{1}{n}\sum_{i=1}^{n} E(x_i)
$$

Here is the crucial modelling assumption:

> **Each measurement $x_i$ is a random variable drawn from the *same* underlying distribution.**

Because every $x_i$ comes from the same distribution, each has the same expected value, namely the true population mean: $E(x_i) = \mu$. Substituting:

$$
= \frac{1}{n}\sum_{i=1}^{n} \mu = \frac{1}{n}\times n\,\mu = \boxed{\mu}
$$

### Conclusion

$$
B(\bar{x}) = E(\bar{x}) - \mu = \mu - \mu = \boxed{0}
$$

The bias is exactly zero, so **the sample mean is an unbiased estimator of the population mean.** This is why averaging your measurements is the "honest" thing to do — averaging introduces no systematic error.

---

## 4. Sample variance → an *unbiased* estimator of the population variance

The sample variance estimates the population variance $\sigma^2$. Its formula carries a famous and slightly surprising feature — the denominator is $n-1$, **not** $n$:

$$
s^2 = \frac{1}{\,n-1\,}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2
$$

where $\bar{x}$ is the sample mean computed above.

> With the $\tfrac{1}{n-1}$ factor, $s^2$ **is an unbiased estimator** of $\sigma^2$ (the population variance).

### Why $n-1$? — Degrees of freedom

> The factor $(n-1)$ appears because the **number of degrees of freedom, *after* defining the sample mean, is $(n-1)$.**

**Intuition:** the deviations $(x_i - \bar{x})$ are not all free to vary independently. By construction they must sum to zero:

$$
\sum_{i=1}^{n}(x_i - \bar{x}) = 0.
$$

That single constraint means that once you know $n-1$ of the deviations, the last one is completely determined. So there are only $n-1$ *independent* pieces of information left about the spread — we "used up" one degree of freedom to estimate the mean. Dividing by $n-1$ rather than $n$ exactly compensates for this and removes the bias.

### The biased alternative: the "uncorrected" sample variance

If we naively divide by $n$ instead, we get the **uncorrected sample variance**:

$$
\tilde{s}^2 = \frac{1}{n}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2 \qquad \text{: uncorrected sample variance (biased estimator).}
$$

Dividing by $n$ *systematically under-estimates* the true variance, because the data points are, on average, closer to their own sample mean $\bar{x}$ than to the true mean $\mu$. This is a **biased estimator**. Using $n-1$ inflates the estimate by just the right amount to fix that under-estimation.

---

## 5. Sample standard deviation → the square root of the sample variance

The **sample standard deviation** is simply the square root of the sample variance. It brings the measure of spread back into the same units as the data itself.

**Unbiased (corrected) version — divide by $n-1$:**

$$
s = \sqrt{\frac{1}{\,n-1\,}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2} \qquad \big(\texttt{STDEV.P}\big)
$$

**Uncorrected version — divide by $n$:**

$$
\tilde{s} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2} \qquad \big(\texttt{STDEV in Excel}\big)
$$

> **Practical/software note (as written on the board):** the two spreadsheet functions correspond to the two formulas. Be careful which one a tool gives you — the "$n$" and "$n-1$" versions are genuinely different, and confusing them is a common source of small numerical discrepancies.
>
> *(Heads-up: Excel's actual convention is the reverse of the labels shown — `STDEV`/`STDEV.S` uses $n-1$ and `STDEV.P` uses $n$. The board is illustrating "there are two versions, one per formula"; double-check the exact function name in your own software before using it.)*

**A subtlety worth knowing:** even though $s^2$ (with $n-1$) is an unbiased estimator of $\sigma^2$, taking a square root is a nonlinear operation, so $s$ is *not* a perfectly unbiased estimator of $\sigma$. For most practical purposes this small residual bias is ignored, but it is the reason people say "the $n-1$ version corrects the *variance*."

---

## 6. Sample covariance → measuring how two variables move together

So far we have described a *single* variable. Covariance extends the idea to **two** random variables at once, quantifying whether they tend to increase and decrease together.

Setup: for two random variables $X$ and $Y$, suppose we have $n$ data points (pairs of measurements):

$$
(x_1, y_1),\ (x_2, y_2),\ \ldots,\ (x_n, y_n).
$$

The **sample covariance** is:

$$
\boxed{\,q_{xy} = \frac{1}{\,n-1\,}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)\left(y_i - \bar{y}\right)\,}
$$

where $\bar{x}$ and $\bar{y}$ are the sample means of the two variables respectively.

**How to read it:**
- Each term multiplies the deviation of $x_i$ from its mean by the deviation of $y_i$ from its mean.
- If $X$ and $Y$ tend to be *above* their means together (and below together), the products are positive → **positive covariance**.
- If one tends to be above while the other is below, the products are negative → **negative covariance**.
- The same $(n-1)$ degrees-of-freedom correction appears, for the same reason as in the sample variance.
- Note that the covariance of a variable with *itself* is exactly the sample variance: $q_{xx} = s^2$.

---

## 7. Setting up for many variables: the data matrix

To generalise beyond two variables, arrange the whole dataset as a matrix. Each **row** is a data point (an observation); each **column** is a variable (a feature):

$$
\mathbf{X} =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
$$

Orientation of the matrix:

- **Rows** run over the $n$ data points — the 1st row is the *1st data point*, the $n$-th row is the *$n$-th data point*.
- **Columns** run over the $p$ variables — the 1st column is the *1st variable*, the $p$-th column is the *$p$-th variable*.
- So $x_{ij}$ = the value of the *$j$-th variable* for the *$i$-th data point*. The matrix is $n \times p$.

### The mean vector

Instead of a single mean, we now have one mean per variable (per column). Collect them into a **mean vector**:

$$
\bar{\mathbf{x}} = \begin{pmatrix} \bar{x}_1 & \bar{x}_2 & \cdots & \bar{x}_p \end{pmatrix}
= \begin{pmatrix} \dfrac{1}{n}\sum_{i=1}^{n} x_{i1} & \dfrac{1}{n}\sum_{i=1}^{n} x_{i2} & \cdots & \dfrac{1}{n}\sum_{i=1}^{n} x_{ip} \end{pmatrix}
$$

Compactly, this is just the average of the data-point (row) vectors:

$$
\boxed{\,\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i\,} \qquad \text{where } \mathbf{x}_i = (x_{i1}\ x_{i2}\ \cdots\ x_{ip}).
$$

The highlighted term $(x_{i1}\ x_{i2}\ \cdots\ x_{ip})$ is simply the $i$-th row — the full vector of measurements for data point $i$.

---

## 8. The sample covariance matrix $S$

When there are $p$ variables, every *pair* of variables has a covariance. Collecting all of these into a single $p \times p$ array gives the **sample covariance matrix** $S$.

With indices $j, k = 1, 2, 3, \ldots, p$ labelling the variables:

$$
S =
\begin{pmatrix}
s_{11} & s_{12} & \cdots & s_{1p} \\
s_{21} & s_{22} & \cdots & s_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
s_{p1} & s_{p2} & \cdots & s_{pp}
\end{pmatrix}
$$

where each entry is a sample covariance between variable $j$ and variable $k$:

$$
s_{jk} = \frac{1}{\,n-1\,}\sum_{i=1}^{n}\left(x_{ij} - \bar{x}_j\right)\left(x_{ik} - \bar{x}_k\right).
$$

### Two key properties

**(a) The diagonal entries are the sample variances.**
The diagonal entry $s_{jj}$ is the covariance of variable $j$ with *itself*, which is exactly its sample variance:

$$
s_{jj} = \frac{1}{\,n-1\,}\sum_{i=1}^{n}\left(x_{ij} - \bar{x}_j\right)^2 = s_j^2.
$$

**(b) $S$ is a symmetric matrix**, because covariance does not care about the order of the two variables:

$$
s_{jk} = s_{kj}.
$$

### Written out in full

Writing every entry explicitly makes the structure transparent — variances on the diagonal, covariances off-diagonal:

$$
S =
\begin{pmatrix}
\dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_{i1}-\bar{x}_1)^2
& \dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_{i1}-\bar{x}_1)(x_{i2}-\bar{x}_2)
& \cdots
& \dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_{i1}-\bar{x}_1)(x_{ip}-\bar{x}_p) \\[2ex]
\vdots & & & \vdots \\[1ex]
\dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_{ip}-\bar{x}_p)(x_{i1}-\bar{x}_1)
& \cdots & \cdots
& \dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_{ip}-\bar{x}_p)^2
\end{pmatrix}_{p\times p}
$$

### The compact outer-product form

All of this collapses into one elegant expression. For each data point form the **column** deviation vector (of size $p \times 1$):

$$
(\mathbf{x}_i - \bar{\mathbf{x}}) =
\begin{pmatrix}
x_{i1} - \bar{x}_1 \\
x_{i2} - \bar{x}_2 \\
\vdots \\
x_{ip} - \bar{x}_p
\end{pmatrix}_{p\times 1}
$$

Multiplying this column ($p\times 1$) by its transpose, the row ($1\times p$), gives a $p\times p$ matrix (an **outer product**). Summing over all data points and applying the $\tfrac{1}{n-1}$ factor:

$$
S = \frac{1}{\,n-1\,}\sum_{i=1}^{n}
\underbrace{(\mathbf{x}_i - \bar{\mathbf{x}})}_{p\times 1}\;
\underbrace{(\mathbf{x}_i - \bar{\mathbf{x}})^{\!\top}}_{1\times p}
$$

> **Note on convention:** the board writes this as $\frac{1}{n-1}\sum (\mathbf{x}_i-\bar{\mathbf{x}})^{\top}(\mathbf{x}_i-\bar{\mathbf{x}})$. The point is simply "(column deviation) times (row deviation)" so that a $p\times 1$ times a $1\times p$ produces the $p\times p$ covariance matrix. The dimensions must be arranged so the result is $p \times p$.

$$
\boxed{\,S = \frac{1}{\,n-1\,}\sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})^{\top}(\mathbf{x}_i - \bar{\mathbf{x}}) \quad \longrightarrow \quad \textbf{Sample covariance matrix}\,}
$$

This single matrix packages *all* the variances (diagonal) and *all* the pairwise covariances (off-diagonal) of a multi-variable dataset into one object — the foundation for methods like PCA, Mahalanobis distance, and multivariate Gaussians later in the course.

---

## Quick reference — all the estimators at a glance

| Quantity | Estimator | Formula | Bias |
|----------|-----------|---------|------|
| Population mean $\mu$ | Sample mean $\bar{x}$ | $\dfrac{1}{n}\sum_i x_i$ | **Unbiased** ($B=0$) |
| Population variance $\sigma^2$ | Sample variance $s^2$ | $\dfrac{1}{n-1}\sum_i (x_i-\bar{x})^2$ | **Unbiased** |
| Population variance $\sigma^2$ | Uncorrected variance $\tilde{s}^2$ | $\dfrac{1}{n}\sum_i (x_i-\bar{x})^2$ | **Biased** (under-estimates) |
| Population std. dev. $\sigma$ | Sample std. dev. $s$ | $\sqrt{\dfrac{1}{n-1}\sum_i (x_i-\bar{x})^2}$ | ~unbiased (small residual bias) |
| Covariance of $X,Y$ | Sample covariance $q_{xy}$ | $\dfrac{1}{n-1}\sum_i (x_i-\bar{x})(y_i-\bar{y})$ | **Unbiased** |
| All variances + covariances | Covariance matrix $S$ | $\dfrac{1}{n-1}\sum_i (\mathbf{x}_i-\bar{\mathbf{x}})^{\top}(\mathbf{x}_i-\bar{\mathbf{x}})$ | **Unbiased**, symmetric |

**The one idea to carry away:** we can never *compute* population properties (we don't know $f(x)$), so we *estimate* them from a sample — and the gold standard for an estimator is that it be **unbiased**, meaning $E(\hat{\theta}) = \theta$. The recurring $n-1$ factor is precisely what makes the variance and covariance estimators unbiased, by accounting for the one degree of freedom spent estimating the mean.
