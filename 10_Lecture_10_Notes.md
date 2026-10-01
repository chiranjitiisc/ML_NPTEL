# Lecture 10 — Confidence Intervals & Hypothesis Testing for Linear-Regression Parameters

> Source: handwritten lecture slides (NPTEL, IISc), folder `ML_LECNOTES/10`
>
> Continues from Lecture 9, where we derived $\hat{\beta} = (X^TX)^{-1}X^TY$. That gave us a *number* for each parameter. This lecture asks: **how much do we trust that number?**

---

## 1. Confidence intervals — definition

A **confidence interval (CI)** at the $c = 100(1-\alpha)\%$ level refers to an interval around a **stochastic quantity** in which it will lie $c\%$ of times.

A related quantity is the **significance level**, denoted $\alpha$.

| Confidence level $c$ | Significance level $\alpha$ |
|---|---|
| 99 % | 0.01 |
| 95 % | 0.05 |
| 90 % | 0.1 |

$$\boxed{c = 100(1-\alpha)\%}$$

> **Intuition.** The parameter itself is fixed but unknown; what is random is *our estimate of it*, because it came from a finite noisy sample. A 95 % CI is a recipe that, repeated over many samples, brackets the true value 95 % of the time.

---

## 2. Hypothesis testing — the framework

**Hypothesis testing** refers to the validation of a certain statement, called the **null hypothesis** $(H_0)$, as opposed to a contrary statement called the **alternate hypothesis** $(H_a)$, with respect to a given significance level.

**Variable of interest** $\longrightarrow$ the **population mean of the linear regression parameters**.

Two tests for the population mean:

| Test | When to use |
|---|---|
| **Z-test** | population $\sigma$ is **known** |
| **t-test** | population $\sigma$ is **unknown** |

---

## 3. The Z-test

Consider an independent sample $(y_1, y_2, \dots, y_n)$ collected from a population with a **known population standard deviation**.

**Sample mean:**

$$\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$$

**Sample variance:**

$$s^{2} = \frac{1}{n-1}\sum_{i=1}^{n}\left(y_i - \bar{y}\right)^{2}$$

*(The slide writes this as $\sigma^2$, but it is the **sample** variance — the same quantity that appears later as $s$ in the t-statistic. The $n-1$ denominator, rather than $n$, is what makes it an unbiased estimator, since $\bar{y}$ was itself estimated from the same data.)*

### The sample mean is unbiased

$$E[\bar{y}] = E\left[\frac{1}{n}\sum_{i=1}^{n} y_i\right] = \frac{1}{n}\sum_{i=1}^{n} E[y_i] = \frac{1}{\cancel{n}} \times \cancel{n}\,\mu = \boxed{\mu}$$

### and its variance shrinks like $1/n$

$$\text{Var}[\bar{y}] = \text{Var}\left[\frac{1}{n}\sum_{i=1}^{n} y_i\right] = \frac{1}{n^{2}}\sum_{i=1}^{n}\text{Var}[y_i] = \frac{1}{n^{2}} \times n\sigma^{2} = \boxed{\frac{\sigma^{2}}{n}}$$

> The step $\text{Var}\left[\sum y_i\right] = \sum \text{Var}[y_i]$ uses **independence** of the samples — cross-covariance terms vanish. The $1/n^2$ comes out because $\text{Var}[aX] = a^2\text{Var}[X]$.
>
> This is the whole reason more data helps: the *spread* of the estimate falls as $\sigma/\sqrt{n}$.

### The Z-statistic

If we assume each $y_i$ to be **normally distributed**, then:

$$y_i \sim \mathcal{N}(\mu, \sigma^{2}) \qquad \Longrightarrow \qquad \bar{y} \sim \mathcal{N}\!\left(\mu, \frac{\sigma^{2}}{n}\right)$$

Standardising:

$$Z = \frac{\bar{y} - \mu}{\sigma/\sqrt{n}} \qquad \Longrightarrow \qquad \boxed{Z \sim \mathcal{N}(0,1)}$$

The quantity $Z$ is the **Z-statistic**. Note the denominator $\sigma/\sqrt{n}$ is the **standard error** of the mean — the standard deviation of $\bar{y}$, not of a single data point.

---

## 4. One-sided vs two-sided tests

$$\text{Hypothesis test} \begin{cases} \text{One-sided test} \\ \text{Two-sided test}\end{cases}$$

### Two-sided test

```
              f(z)
               ^
               |      .-.
               |     /   \
               |    /     \
        alpha/2|  ,'  1-a  ',   alpha/2
            <--|_/           \_|-->
        _______|_:___________:_|________> z
                z_(a/2)    z_(1-a/2)
```

$H_0$ (null hypothesis) is **rejected** if

$$Z > z_{1-\frac{\alpha}{2}} \qquad \text{or} \qquad Z < z_{\frac{\alpha}{2}}$$

Otherwise the null hypothesis **cannot be rejected**.

The probability mass $(1-\alpha)$ sits in the middle; each tail carries $\alpha/2$.

### One-sided test

```
              f(z)
               ^
               |    .-.
               |   /   \
               |  /     \
               | /  1-a  \
               |/         \  alpha
        _______|___________:\____> z
                         z_(1-a)
```

$H_0$ is **rejected** if

$$Z > z_{1-\alpha}$$

Otherwise the null hypothesis cannot be rejected. The whole $\alpha$ sits in one tail.

> **Which to use:** two-sided when the alternate hypothesis is "$\ne$" (the parameter differs from the null value in *either* direction); one-sided when it is "$>$" or "$<$".
>
> Note "cannot be rejected" is deliberately *not* "is accepted" — failing to detect an effect is not evidence there is none.

---

## 5. Student's t-test

Test for the population mean when the population standard deviation $(\sigma)$ is **unknown**.

Random variable:

$$T = \frac{\bar{y} - \mu}{s/\sqrt{n}} \sim t_{n-1}$$

where

- $s$ = **sample** standard deviation (replacing the unknown $\sigma$),
- $t_{n-1}$ = **Student's t distribution with $(n-1)$ degrees of freedom.**

> Replacing $\sigma$ by the noisy estimate $s$ injects extra uncertainty, so the distribution is *fatter-tailed* than the normal. As $n \to \infty$, $s \to \sigma$ and $t_{n-1} \to \mathcal{N}(0,1)$ — the t-test collapses back onto the Z-test.

### Deriving the confidence interval for $\mu$

Start from the probability statement that $T$ lies between its two critical values:

$$P\!\left(t_{\frac{\alpha}{2}} \le T \le t_{1-\frac{\alpha}{2}}\right) = 1-\alpha$$

Substitute $T$:

$$P\!\left(t_{\frac{\alpha}{2}} \le \frac{\bar{y}-\mu}{s/\sqrt{n}} \le t_{1-\frac{\alpha}{2}}\right) = 1-\alpha$$

Multiply through by $s/\sqrt{n}$:

$$P\!\left(t_{\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}} \le \bar{y}-\mu \le t_{1-\frac{\alpha}{2}}\cdot\frac{s}{\sqrt{n}}\right) = 1-\alpha$$

Multiply by $-1$ — **which flips both inequality signs**, so the limits swap ends:

$$P\!\left(-t_{1-\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}} \le \mu - \bar{y} \le -t_{\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}}\right) = 1-\alpha$$

Add $\bar{y}$:

$$\boxed{P\!\left(\bar{y} - t_{1-\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}} \;\le\; \mu \;\le\; \bar{y} - t_{\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}}\right) = 1-\alpha}$$

This is the **$100(1-\alpha)\%$ confidence interval for the population mean**.

> Because the t distribution is symmetric, $t_{\frac{\alpha}{2}} = -\,t_{1-\frac{\alpha}{2}}$, so the boxed result is the familiar symmetric interval
> $$\bar{y} \pm t_{1-\frac{\alpha}{2}}\,\frac{s}{\sqrt{n}}$$
> The "minus" in the upper limit is not a typo — the critical value $t_{\alpha/2}$ is itself negative.

---

## 6. Confidence intervals for the linear-regression parameters

From Lecture 9:

$$\hat{\beta} = \left(X^TX\right)^{-1}X^TY$$

To assign a CI to $\hat{\beta}$, we need to know:

- $E(\hat{\beta})$ — where the estimate sits on average
- $\text{Var}(\hat{\beta})$ — how much it scatters

**Assumption:** the errors $(\epsilon)$ in prediction are **normally distributed**.

The true data-generating model is

$$Y = X\beta + \epsilon$$

| Symbol | Meaning |
|---|---|
| $\beta$ | the **actual** (true, unknown) parameters |
| $\epsilon$ | the **error** |
| $\hat{\beta}$ | our least-squares estimate of $\beta$ |

Standard assumptions on the noise: $E(\epsilon) = 0$, $\text{Var}(\epsilon) = \sigma^2$, errors uncorrelated — i.e. $E[\epsilon\epsilon^T] = \sigma^2 I$.

---

## 7. $\hat{\beta}$ is an unbiased estimator

$$
\begin{aligned}
E[\hat{\beta}] &= E\!\left[(X^TX)^{-1}X^TY\right] \\[4pt]
&= E\!\left[(X^TX)^{-1}X^T\left(X\beta + \epsilon\right)\right] \\[4pt]
&= E\Big[\underbrace{(X^TX)^{-1}X^TX}_{=\;I}\,\beta \;+\; (X^TX)^{-1}X^T\epsilon\Big] \\[4pt]
&= E[I\beta] \;+\; (X^TX)^{-1}X^T\underbrace{E(\epsilon)}_{=\;0} \\[4pt]
&= E[\beta] + 0
\end{aligned}
$$

$$\boxed{E[\hat{\beta}] = E[\beta] = \beta}$$

$\hat{\beta}$ is an **unbiased estimator** for $\beta$.

> $(X^TX)^{-1}X^TX = I$ exactly — that is what makes the least-squares estimator hit the right target on average. Nothing about the size of the noise enters here; only $E(\epsilon) = 0$ is needed.

---

## 8. Variance of $\hat{\beta}$

$$\text{Cov}(\hat{\beta}) = E\!\left[(\hat{\beta}-\beta)(\hat{\beta}-\beta)^{T}\right]$$

**First get $\hat{\beta}-\beta$ in terms of $\epsilon$.** From the normal equation,

$$\hat{\beta} = (X^TX)^{-1}X^TY \quad \Longrightarrow \quad X^TX\hat{\beta} = X^TY$$

and from the model $Y = X\beta + \epsilon$,

$$X^TY = X^TX\beta + X^T\epsilon$$

Equating the two expressions for $X^TY$:

$$X^TX\hat{\beta} = X^TX\beta + X^T\epsilon$$

$$\Longrightarrow \quad \boxed{\hat{\beta}-\beta = \left(X^TX\right)^{-1}X^T\epsilon}$$

**Substitute into the covariance:**

$$
\begin{aligned}
\text{Cov}(\hat{\beta}) &= E\!\left[(X^TX)^{-1}X^T\epsilon \cdot \left((X^TX)^{-1}X^T\epsilon\right)^{T}\right] \\[4pt]
&= \left(X^TX\right)^{-1} E\!\left[\epsilon\epsilon^{T}\right]
\end{aligned}
$$

> **Working the middle step in full.** $\left((X^TX)^{-1}X^T\epsilon\right)^T = \epsilon^T X (X^TX)^{-1}$, since $(X^TX)^{-1}$ is symmetric. So
> $$\text{Cov}(\hat{\beta}) = (X^TX)^{-1}X^T\,E[\epsilon\epsilon^T]\,X(X^TX)^{-1}$$
> With $E[\epsilon\epsilon^T] = \sigma^2 I$ this collapses:
> $$= \sigma^2 (X^TX)^{-1}X^TX(X^TX)^{-1} = \sigma^2\left(X^TX\right)^{-1}$$
> which is the compressed form written on the slide.

**Taking the diagonal** (the variance of each individual parameter, ignoring the parameter-to-parameter covariances):

$$\text{Var}(\hat{\beta}) = \text{diag}\!\left[(X^TX)^{-1}E[\epsilon\epsilon^T]\right] = \text{diag}\!\left((X^TX)^{-1}\right)\cdot \underbrace{\text{Var}(\epsilon)}_{=\;\sigma^{2}}$$

$$\boxed{\text{Var}(\hat{\beta}) = \sigma^{2}\,\text{diag}\!\left(\left(X^TX\right)^{-1}\right)}$$

Hence

$$\hat{\beta} \sim \mathcal{N}\!\left(\beta,\; \sigma^{2}\,\text{diag}\!\left((X^TX)^{-1}\right)\right)$$

> Read this off: the uncertainty in each fitted parameter is set by two things — how noisy the data are $(\sigma^2)$, and the geometry of the design matrix $\left((X^TX)^{-1}\right)$. Nearly-collinear features make $X^TX$ close to singular, its inverse blows up, and the parameter estimates become wildly uncertain even though the fit itself may look fine.

---

## 9. The t-statistic for a regression parameter

$$\frac{\hat{\beta}-\beta}{\sqrt{\sigma^{2}\,\text{diag}\!\left((X^TX)^{-1}\right)}} \;\sim\; t_{n-p}$$

where $p$ = **number of parameters in the linear regression model**, so $n-p$ is the number of degrees of freedom.

> Why $t$ and not $Z$: $\sigma^2$ is not actually known — it is estimated from the residuals, $\hat{\sigma}^2 = \text{SSE}/(n-p)$. Fitting $p$ parameters uses up $p$ degrees of freedom out of the $n$ data points, leaving $n-p$. (In the simple sample-mean case of §5, only one parameter — the mean — was fitted, giving $n-1$.)

---

## 10. Final result — CI for the regression parameters

$$\boxed{P\!\left(\hat{\beta} - t_{n-p,\,1-\frac{\alpha}{2}}\sqrt{\sigma^{2}\,\text{diag}\!\left((X^TX)^{-1}\right)} \;\le\; \beta \;\le\; \hat{\beta} - t_{n-p,\,\frac{\alpha}{2}}\sqrt{\sigma^{2}\,\text{diag}\!\left((X^TX)^{-1}\right)}\right) = 1-\alpha}$$

If we use $\alpha = 0.01$, we will obtain the **99 % CI**.

Exactly as in §5, symmetry of the t distribution means $t_{n-p,\,\alpha/2} = -\,t_{n-p,\,1-\alpha/2}$, so in practice this is

$$\hat{\beta}_j \;\pm\; t_{n-p,\,1-\frac{\alpha}{2}}\;\times\;\underbrace{\sqrt{\hat{\sigma}^{2}\left[(X^TX)^{-1}\right]_{jj}}}_{\text{standard error of }\hat{\beta}_j}$$

one interval per parameter $j$.

> **Why this matters practically.** If the CI for a parameter $\beta_j$ **straddles zero**, you cannot reject the null hypothesis $H_0: \beta_j = 0$ at that significance level — that feature is not demonstrably contributing to the model. This is precisely the "p-value" column that regression software prints next to each coefficient.

---

## Summary — the whole lecture in one chain

1. A CI at level $c = 100(1-\alpha)\%$ brackets a stochastic quantity $c\%$ of the time; $\alpha$ is the significance level.
2. Hypothesis testing pits $H_0$ against $H_a$ at a chosen $\alpha$; reject in one tail (one-sided) or two (two-sided).
3. For the mean of a sample: $E[\bar{y}] = \mu$, $\text{Var}[\bar{y}] = \sigma^2/n$.
4. $\sigma$ known $\Rightarrow$ $Z = \dfrac{\bar{y}-\mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)$. &nbsp; $\sigma$ unknown $\Rightarrow$ $T = \dfrac{\bar{y}-\mu}{s/\sqrt{n}} \sim t_{n-1}$.
5. Invert the probability statement on $T$ to get the CI for $\mu$.
6. For regression: $Y = X\beta+\epsilon$, and $\hat{\beta}-\beta = (X^TX)^{-1}X^T\epsilon$.
7. $E[\hat{\beta}] = \beta$ (**unbiased**) and $\text{Var}(\hat{\beta}) = \sigma^2\,\text{diag}\!\left((X^TX)^{-1}\right)$.
8. So $\dfrac{\hat{\beta}-\beta}{\sqrt{\sigma^2\text{diag}((X^TX)^{-1})}} \sim t_{n-p}$ and the CI follows by the same inversion as step 5.
