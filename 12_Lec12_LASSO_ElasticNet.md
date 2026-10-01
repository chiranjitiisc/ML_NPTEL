# Lecture 12 — LASSO, Ridge, and Elastic Net
### With fully worked numerical examples, and how to choose $\lambda$ (a.k.a. `alpha`)

---

## Table of contents

0. [Why we need regularization at all](#0-why-we-need-regularization-at-all)
1. [Notation and the one thing you must do first: standardization](#1-notation-and-standardization)
2. [Ridge regression — recap with a numerical example](#2-ridge-regression-recap)
3. [LASSO — what the name actually means](#3-lasso-least-absolute-shrinkage-and-selection-operator)
4. [The geometry: why LASSO gives *exact* zeros and ridge does not](#4-the-geometry-why-lasso-gives-exact-zeros)
5. [**Worked Example A** — orthogonal design → LASSO by hand (soft-thresholding)](#5-worked-example-a-orthogonal-design)
6. [Why there is no analytical solution for LASSO](#6-why-there-is-no-analytical-solution-for-lasso)
7. [The numerical method: coordinate descent, step by step](#7-numerical-method-coordinate-descent)
8. [**Worked Example B** — correlated predictors: ridge vs LASSO side by side](#8-worked-example-b-correlated-predictors)
9. [Elastic Net — combining both penalties](#9-elastic-net)
10. [**Choosing $\lambda$ / `alpha`** — from the very basics](#10-choosing-lambda-alpha-from-the-very-basics)
11. [**Worked Example C** — a complete 4-fold cross-validation, every number shown](#11-worked-example-c-complete-cross-validation)
12. [The $\lambda$ ↔ `alpha` conversion (why your hand answer ≠ scikit-learn's)](#12-the-lambda-alpha-conversion)
13. [Reproduce everything: Python code](#13-python-code-to-reproduce-every-number)
14. [Cheat sheet](#14-cheat-sheet)

---

## 0. Why we need regularization at all

Start from ordinary linear regression. We have $n$ observations, each with $p$ features:

$$
\hat{y}_i \;=\; \beta_0 \;+\; \sum_{j=1}^{p}\beta_j x_{ij}
$$

Ordinary Least Squares (OLS) picks the $\beta$'s that minimise the **sum of squared errors**:

$$
\text{SSE}(\beta)\;=\;\sum_{i=1}^{n}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_j x_{ij}\Big)^2
$$

This has a closed-form answer, $\hat\beta_{\text{OLS}}=(X^\top X)^{-1}X^\top y$. So why do we need anything else?

**Three failure modes of OLS:**

| Problem | What goes wrong | Symptom |
|---|---|---|
| $p$ close to (or larger than) $n$ | $X^\top X$ is singular or nearly so | $(X^\top X)^{-1}$ blows up; infinitely many solutions |
| Correlated predictors (multicollinearity) | The loss surface has a long flat valley | Coefficients become huge and of opposite sign; tiny data changes flip them |
| Too many features | The model fits noise | Great training error, terrible test error (**overfitting**) |

The common thread: **the coefficients get too large.** So the fix is to *penalise* large coefficients. We add a penalty term to the loss:

$$
L(\beta)\;=\;\underbrace{\text{SSE}(\beta)}_{\text{fit the data}}\;+\;\underbrace{\lambda\cdot P(\beta)}_{\text{keep }\beta\text{ small}}
$$

- $P(\beta)=\sum_j \beta_j^2$ → **Ridge regression** ($L_2$ penalty)
- $P(\beta)=\sum_j |\beta_j|$ → **LASSO** ($L_1$ penalty)
- Both together → **Elastic Net**

$\lambda \ge 0$ is the **regularization strength** (the "knob"). It is *not* learned from the training loss — if it were, the answer would always be $\lambda=0$. It is chosen by **cross-validation** (Section 10).

> **A note on the intercept $\beta_0$:** it is **never penalised.** $\beta_0$ just sets the overall level of $y$; shrinking it would bias every prediction toward zero. Standard practice: centre $y$ and $X$, fit the penalised model without an intercept, then recover $\hat\beta_0=\bar y-\sum_j \hat\beta_j\bar x_j$.

---

## 1. Notation and standardization

### 1.1 The convention used throughout these notes

$$
\boxed{\;L(\beta)\;=\;\sum_{i=1}^{n}\Big(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\Big)^{2}\;+\;\lambda\sum_{j=1}^{p}|\beta_j|\;}
\qquad\text{(LASSO)}
$$

I will call the knob $\lambda$. **scikit-learn calls it `alpha` and defines the loss differently** — see Section 12 for the exact conversion. Do not mix them up; it is the single most common source of confusion.

### 1.2 Why you MUST standardize first

The penalty $\sum_j|\beta_j|$ adds up coefficients from different features. But coefficients carry *units*.

Suppose $x_1$ = temperature in **kelvin** and $x_2$ = pressure in **pascal**. Now re-express pressure in **bar** ($1\,\text{bar}=10^5\,\text{Pa}$). The data has not changed. But $\beta_2$ must become $10^5$ times bigger to give the same prediction — so $|\beta_2|$ now dominates the penalty completely, and the LASSO will crush that feature to zero for reasons that have nothing to do with the physics.

**A penalty that treats $\beta_1$ and $\beta_2$ on equal footing is only meaningful if $x_1$ and $x_2$ are on equal footing.**

So before fitting, replace each column by its **z-score**:

$$
z_{ij}\;=\;\frac{x_{ij}-\bar x_j}{s_j},\qquad
\bar x_j=\frac1n\sum_i x_{ij},\qquad
s_j=\sqrt{\frac1n\sum_i (x_{ij}-\bar x_j)^2}
$$

After this, every column has mean 0 and standard deviation 1.

**Tiny numerical illustration.** $x=(2,4,6,8)$ → $\bar x=5$, $s=\sqrt{\tfrac14(9+1+1+9)}=\sqrt5=2.2361$.

$$
z=\left(\frac{2-5}{2.2361},\frac{4-5}{2.2361},\frac{6-5}{2.2361},\frac{8-5}{2.2361}\right)=(-1.3416,\,-0.4472,\,0.4472,\,1.3416)
$$

Check: mean $=0$ ✓, $\frac14\sum z_i^2 = \frac14(1.8+0.2+0.2+1.8)=1$ ✓.

### 1.3 Getting back to original units

If you fit on standardized $z$ and get $\hat\beta^{z}_j$, convert back with

$$
\hat\beta^{\text{orig}}_j=\frac{\hat\beta^{z}_j}{s_j},
\qquad
\hat\beta_0=\bar y-\sum_{j=1}^p \hat\beta^{\text{orig}}_j\,\bar x_j
$$

(You will see this applied with real numbers at the end of Worked Example C.)

> ⚠️ **The classic mistake:** standardizing using the mean/SD of the *whole* dataset, then doing cross-validation. That leaks test-fold information into training. Compute $\bar x_j, s_j$ **from the training fold only**, and apply those same numbers to the test fold. Example C does it correctly.

---

## 2. Ridge regression (recap)

$$
L(\beta)=\sum_{i}\Big(y_i-\sum_j x_{ij}\beta_j\Big)^2+\lambda\sum_j\beta_j^2
\qquad (y,X \text{ centred})
$$

Differentiate and set to zero:

$$
-2X^\top(y-X\beta)+2\lambda\beta=0
\;\;\Longrightarrow\;\;
\boxed{\;\hat\beta_{\text{ridge}}=(X^\top X+\lambda I)^{-1}X^\top y\;}
$$

Two things to notice:

1. **A closed-form solution exists.** Adding $\lambda I$ to the diagonal makes the matrix invertible even when $X^\top X$ is singular. That is why ridge is sometimes called "regularized least squares" or Tikhonov regularization.
2. **$\hat\beta_{\text{ridge}}$ is never exactly zero** for finite $\lambda$ (unless $X^\top y$ happens to be exactly zero). Each coefficient is divided by something like $(\text{scale}+\lambda)$, which shrinks it smoothly toward zero but *never reaches* it.

### 2.1 The cleanest possible ridge example

Take an **orthogonal** design, so $X^\top X$ is diagonal. Then the formula decouples completely:

$$
\hat\beta^{\text{ridge}}_j=\frac{x_j^\top y}{x_j^\top x_j+\lambda}
$$

With $x_j^\top x_j = 4$ and $x_j^\top y = 12$:

| $\lambda$ | 0 | 2 | 4 | 8 | 16 | 100 |
|---|---|---|---|---|---|---|
| $\hat\beta_j = 12/(4+\lambda)$ | **3.000** | 2.000 | 1.500 | 1.000 | 0.600 | 0.115 |

It keeps getting smaller, forever — but it is never 0. **That is the whole difference between ridge and LASSO in one table.**

---

## 3. LASSO: Least Absolute Shrinkage and Selection Operator

Unpack the name — every word is doing work:

- **Least Absolute** → the penalty is the sum of **absolute values**, $\sum_j|\beta_j|$ (the $L_1$ norm), not squares.
- **Shrinkage** → it pushes coefficients toward the smallest values consistent with a good fit.
- **Selection** → because some coefficients become **exactly zero**, LASSO *selects* a subset of features. The features with $\hat\beta_j=0$ are simply dropped from the model. Ridge can never do this.
- **Operator** → it is a map from data to a coefficient vector.

$$
\boxed{\;L(\beta)=\sum_{i=1}^{n}\Big(y_i-\big(\beta_0+\textstyle\sum_{j=1}^{p}x_{ij}\beta_j\big)\Big)^{2}+\lambda\sum_{j=1}^{p}\big|\beta_j\big|\;}
$$

So LASSO does **regularization and feature selection at the same time**, in one fit. If you have 500 candidate descriptors and suspect only 8 matter, LASSO is the natural tool.

### 3.1 The equivalent constrained form

Every penalised problem has a **constrained twin**. Minimising $\text{SSE}+\lambda\sum_j|\beta_j|$ is exactly equivalent to

$$
\min_\beta \;\text{SSE}(\beta)
\qquad\text{subject to}\qquad
\sum_{j=1}^{p}|\beta_j|\;\le\;t
$$

for some budget $t$ that depends on $\lambda$. Similarly for ridge, with $\sum_j\beta_j^2\le t$.

- Large $\lambda$ ⟺ small budget $t$ (heavy regularization)
- $\lambda=0$ ⟺ $t=\infty$ (no constraint → OLS)

This constrained view is what makes the geometry picture work.

---

## 4. The geometry: why LASSO gives exact zeros

Draw the $(\beta_1,\beta_2)$ plane.

- The SSE is a quadratic bowl. Its **level sets (iso-contours) are ellipses**, all centred on $\hat\beta_{\text{OLS}}$. Moving inward = lower loss.
- The **constraint region** is:
  - Ridge: $\beta_1^2+\beta_2^2\le t$ → a **circle** (a sphere/ellipsoid in higher $p$). Smooth everywhere.
  - LASSO: $|\beta_1|+|\beta_2|\le t$ → a **diamond** (a cross-polytope in higher $p$), with **sharp corners sitting exactly on the axes**.

```
        RIDGE                              LASSO
          β₂                                 β₂
          │      ╱ iso-contours              │      ╱ iso-contours
          │    ╱╱  of SSE                    │    ╱╱  of SSE
       ╭──┼──╮╱                            ╱ ◆ ╲╱
      ╱   │  ╳●  ← touch point            ╱  │ ●╳  ← touch point is
     │    │ ╱ ╲                          ╱   │╱ ╲     a CORNER: β₁ = 0
─────┼────┼╱───┼───── β₁          ───────◆───┼───◆──────── β₁
     │    │    │                          ╲  │  ╱
      ╲   │   ╱                            ╲ │ ╱
       ╰──┼──╯                              ╲◆╱
          │                                  │
  β₁ ≠ 0 and β₂ ≠ 0                  β₁ = 0 EXACTLY
  (smooth circle — a tangency        (sharp corner — the ellipse
   at a corner is impossible)         hits the vertex first)
```

The optimum is the point where the **shrinking ellipse first touches the constraint region.**

- On a **circle**, the touch point is a smooth tangency. The probability of that tangency landing precisely on an axis is essentially zero → **ridge coefficients are generically all non-zero.**
- On a **diamond**, the corners *stick out*. An expanding ellipse is very likely to hit a corner first — and a corner lies **on an axis**, meaning one or more coefficients are **exactly zero**.

In $p$ dimensions this gets stronger: the $L_1$ ball has corners, edges, and faces of every dimension, so LASSO can zero out any number of coefficients at once. This is the whole mechanism of "selection".

> **As opposed to ridge regression, in LASSO some parameters in the optimal $\hat\beta$ can be *exactly* zero.** Not "small". Not "negligible". Exactly zero.

---

## 5. Worked Example A: orthogonal design

This is the example to do by hand, because with an orthogonal design **LASSO has a closed form** and you can see the mechanism nakedly.

### 5.1 The data

| $i$ | $x_{i1}$ | $x_{i2}$ | $y_i$ |
|:-:|:-:|:-:|:-:|
| 1 | $-1$ | $-1$ | 2 |
| 2 | $-1$ | $+1$ | 2 |
| 3 | $+1$ | $-1$ | 7 |
| 4 | $+1$ | $+1$ | 9 |

$n=4$, $p=2$. Both columns already have mean 0. $\bar y=\frac{2+2+7+9}{4}=5$, so centred $y_c=(-3,\,-3,\,+2,\,+4)$.

**The key quantities:**

$$
X^\top X=\begin{pmatrix}4&0\\0&4\end{pmatrix}\quad\text{(orthogonal! off-diagonals are 0)},
\qquad
X^\top y_c=\begin{pmatrix}12\\2\end{pmatrix}
$$

Check by hand: $x_1^\top y_c=(-1)(-3)+(-1)(-3)+(1)(2)+(1)(4)=3+3+2+4=12$ ✓
and $x_2^\top y_c=(-1)(-3)+(1)(-3)+(-1)(2)+(1)(4)=3-3-2+4=2$ ✓

### 5.2 The OLS baseline

$$
\hat\beta_{\text{OLS}}=(X^\top X)^{-1}X^\top y_c=\begin{pmatrix}12/4\\2/4\end{pmatrix}=\begin{pmatrix}3.0\\0.5\end{pmatrix},\qquad \hat\beta_0=5
$$

Fitted centred values: $3x_1+0.5x_2=(-3.5,-2.5,2.5,3.5)$.
Residuals: $(0.5,-0.5,-0.5,0.5)$ → $\text{SSE}_{\min}=4\times0.25=\mathbf{1.0}$.

So feature 1 is strong ($\beta_1=3$) and feature 2 is weak ($\beta_2=0.5$). **A good selection method should kill feature 2 first.**

### 5.3 Deriving the LASSO solution for this design

Minimise $L(\beta)=\|y_c-X\beta\|^2+\lambda(|\beta_1|+|\beta_2|)$. Because $X^\top X$ is diagonal, the two coordinates decouple:

$$
L=\underbrace{\big(4\beta_1^2-2(12)\beta_1+\lambda|\beta_1|\big)}_{\text{depends only on }\beta_1}+\underbrace{\big(4\beta_2^2-2(2)\beta_2+\lambda|\beta_2|\big)}_{\text{depends only on }\beta_2}+\text{const}
$$

Take one coordinate. For $\beta_j>0$, $\frac{\partial L}{\partial\beta_j}=8\beta_j-2(x_j^\top y_c)+\lambda=0$, giving $\beta_j=\dfrac{x_j^\top y_c-\lambda/2}{4}$.
For $\beta_j<0$ the sign flips. And $\beta_j=0$ is optimal whenever $|x_j^\top y_c|\le\lambda/2$.

All three cases collapse into the **soft-thresholding operator**:

$$
\boxed{\;
S(z,\gamma)=\operatorname{sign}(z)\,\max\!\big(|z|-\gamma,\,0\big),
\qquad
\hat\beta^{\text{LASSO}}_j=\frac{S\!\big(x_j^\top y_c,\;\lambda/2\big)}{x_j^\top x_j}
\;}
$$

**Read it in words:** *move $x_j^\top y_c$ toward zero by a fixed amount $\lambda/2$; if it would cross zero, stop at zero.*

```
   S(z, γ)
      │        ╱
      │      ╱
      │    ╱
──────┼──┬─┬──────── z
    ╱ │ -γ  γ
  ╱   │   ↑
      │  dead zone: any z with |z| ≤ γ  →  output exactly 0
```

Compare with ridge on the same design: $\hat\beta^{\text{ridge}}_j = \dfrac{x_j^\top y_c}{x_j^\top x_j+\lambda}$ — a *multiplicative* shrink, no dead zone, never exactly zero.

### 5.4 The numbers — LASSO vs Ridge on the same data

Using $x_1^\top y_c=12$, $x_2^\top y_c=2$, $x_j^\top x_j=4$:

| $\lambda$ | $\lambda/2$ | $\hat\beta_1^{\text{LASSO}}$ | $\hat\beta_2^{\text{LASSO}}$ | $\hat\beta_1^{\text{ridge}}$ | $\hat\beta_2^{\text{ridge}}$ | # features kept |
|---:|---:|---:|---:|---:|---:|:-:|
| 0 | 0 | 3.000 | 0.500 | 3.000 | 0.500 | 2 |
| 2 | 1 | 2.750 | 0.250 | 2.000 | 0.3333 | 2 |
| **4** | **2** | **2.500** | **0.000** ← *dropped* | 1.500 | 0.2500 | **1** |
| 8 | 4 | 2.000 | 0.000 | 1.000 | 0.1667 | 1 |
| 16 | 8 | 1.000 | 0.000 | 0.600 | 0.1000 | 1 |
| 24 | 12 | 0.000 | 0.000 | 0.4286 | 0.0714 | 0 |
| 30 | 15 | 0.000 | 0.000 | 0.3529 | 0.0588 | 0 |

**Every number, verified by hand:**

- $\lambda=2$: $\hat\beta_1=\dfrac{12-1}{4}=\dfrac{11}{4}=2.75$;  $\hat\beta_2=\dfrac{2-1}{4}=\dfrac{1}{4}=0.25$
- $\lambda=4$: $\hat\beta_1=\dfrac{12-2}{4}=2.5$;  $|x_2^\top y_c|=2\le\lambda/2=2$ → $\hat\beta_2=\mathbf{0}$
- $\lambda=8$: $\hat\beta_1=\dfrac{12-4}{4}=2.0$; $\hat\beta_2=0$
- $\lambda=24$: $|x_1^\top y_c|=12\le 12$ → $\hat\beta_1=\mathbf{0}$ too. The model is now just $\hat y=\bar y=5$.

**Three lessons from this table:**

1. **LASSO drops feature 2 at $\lambda=4$ and it stays dropped.** Ridge shrinks it to 0.25, then 0.167, then 0.1 … and never lets go. The ridge column has no zeros anywhere.
2. There is a **$\lambda_{\max}$ beyond which everything is zero.** Here $\lambda_{\max}=2\max_j|x_j^\top y_c|=24$. This is important later: it tells you where to start your search grid.
3. LASSO applies the **same absolute shrinkage $\lambda/2=2$ to both coefficients** ($12\to10$ and $2\to0$). Ridge applies the **same relative shrinkage** ($\times\frac{4}{4+\lambda}$) to both. That is why LASSO kills small coefficients and ridge merely fades them.

---

## 6. Why there is no analytical solution for LASSO

For ridge we differentiated $\lambda\sum\beta_j^2$ and got $2\lambda\beta$ — smooth, no problem. For LASSO we must differentiate $\lambda\sum|\beta_j|$. And:

```
        |x|
         │╲          ╱
          ╲        ╱
           ╲     ╱
            ╲  ╱
─────────────●─────────────  x
             ↑
      the function |x| is NOT
      differentiable at x = 0
      (left slope −1, right slope +1)
```

$$
\frac{d}{dx}|x|=\begin{cases}+1 & x>0\\ -1 & x<0\\ \textbf{undefined} & x=0\end{cases}
$$

So in

$$
L=\text{SSE}+\lambda\sum_{j=1}^{p}|\beta_j|
$$

the loss function is **not analytically differentiable whenever any parameter $\beta_j$ is at (or close to) zero** — and those are exactly the points we care about, because that is where the selection happens.

$$
\Downarrow
$$

**We cannot set $\nabla L=0$ and solve. An analytical solution is not possible for LASSO, although we derived one for ridge regression.**

Two escape routes exist, and both matter:

**(a) Subgradients.** Replace the derivative at 0 with the *set* $[-1,+1]$. The optimality condition becomes: for each $j$,
$$
-2\,x_j^\top(y-X\hat\beta)+\lambda s_j=0,\qquad
s_j=\begin{cases}\operatorname{sign}(\hat\beta_j) & \hat\beta_j\neq0\\ \text{any value in }[-1,1] & \hat\beta_j=0\end{cases}
$$
This is what gave us the soft-thresholding formula in Example A. It works in closed form **only** when $X^\top X$ is diagonal (orthogonal design). In general it does not close.

**(b) Numerical optimization.** In the general case:

> **Numerical methods can be used to minimize the loss function and thus select the optimal $\hat\beta_{\text{LASSO}}$.**

The workhorse is **coordinate descent** (Section 7); alternatives include LARS (which traces the whole solution path) and proximal gradient / ISTA.

The good news: **$L(\beta)$ is convex** (SSE is convex, $|\cdot|$ is convex, sums of convex functions are convex). So there are no local minima to get trapped in — any numerical method that goes downhill reaches the global optimum.

---

## 7. Numerical method: coordinate descent

### 7.1 The idea, in one sentence

*You cannot solve for all $p$ coefficients at once — but you **can** solve for one coefficient exactly if you freeze all the others. So cycle through them one at a time until nothing changes.*

### 7.2 The update rule, derived

Fix $j$, hold all $\beta_k$ ($k\neq j$) at their current values. Define the **partial residual** — what's left of $y$ after every *other* feature has done its job:

$$
r^{(j)}\;=\;y_c-\sum_{k\neq j}\beta_k x_k
$$

Now the objective, as a function of $\beta_j$ alone, is
$$
\|r^{(j)}-\beta_j x_j\|^2+\lambda|\beta_j| \;+\;\text{const}
$$
which is *exactly* the one-variable problem we already solved in Section 5.3. So:

$$
\boxed{\;
\rho_j = x_j^\top r^{(j)},
\qquad
\beta_j \;\leftarrow\; \frac{S\!\left(\rho_j,\;\lambda/2\right)}{x_j^\top x_j}
\;}
$$

Convenient shortcut for computing $\rho_j$ without rebuilding the residual each time:

$$
\rho_j \;=\; x_j^\top y_c-\sum_{k\neq j}\beta_k\,(x_j^\top x_k)
$$

### 7.3 The algorithm

```
initialise β = 0                        (all coefficients zero)
repeat:
    for j = 1, 2, ..., p:
        ρⱼ ← xⱼᵀy_c − Σ_{k≠j} β_k (xⱼᵀx_k)
        βⱼ ← S(ρⱼ, λ/2) / (xⱼᵀxⱼ)
until max |Δβ| < tolerance
```

Why this is efficient: once a coefficient hits 0 it usually stays 0, and the algorithm can skip it ("active set" strategy). This is why coordinate descent handles $p$ in the tens of thousands.

---

## 8. Worked Example B: correlated predictors

Now a design where the columns are **not** orthogonal — the realistic case, and where the difference between ridge and LASSO becomes most interesting.

### 8.1 The data

| $i$ | $x_{i1}$ | $x_{i2}$ | $y_i$ | $y_{c,i}=y_i-10$ |
|:-:|:-:|:-:|:-:|:-:|
| 1 | $-2$ | $-1$ | 2  | $-8$ |
| 2 | $-1$ | $-2$ | 3  | $-7$ |
| 3 | $+1$ | $+2$ | 15 | $+5$ |
| 4 | $+2$ | $+1$ | 20 | $+10$ |

Both $x$ columns are already centred; $\bar y=\frac{2+3+15+20}{4}=10$.

$$
X^\top X=\begin{pmatrix}10 & 8\\ 8 & 10\end{pmatrix},
\qquad
X^\top y_c=\begin{pmatrix}48\\42\end{pmatrix}
$$

Hand-check: $x_1^\top x_1=4+1+1+4=10$; $x_1^\top x_2=2+2+2+2=8$; $x_1^\top y_c=16+7+5+20=48$; $x_2^\top y_c=8+14+10+10=42$.

**Correlation between the two features:** $\dfrac{8}{\sqrt{10\cdot10}}=\mathbf{0.80}$ — strongly correlated.

### 8.2 OLS baseline

Solve $\begin{pmatrix}10&8\\8&10\end{pmatrix}\beta=\begin{pmatrix}48\\42\end{pmatrix}$. Determinant $=100-64=36$.

$$
\hat\beta_1=\frac{10(48)-8(42)}{36}=\frac{480-336}{36}=\frac{144}{36}=\mathbf{4},
\qquad
\hat\beta_2=\frac{10(42)-8(48)}{36}=\frac{420-384}{36}=\frac{36}{36}=\mathbf{1}
$$

$\hat\beta_{\text{OLS}}=(4,\,1)$, $\hat\beta_0=10$, and $\text{SSE}_{\min}=4.0$.

### 8.3 LASSO by coordinate descent, iteration by iteration ($\lambda=20$, so $\lambda/2=10$)

Using $\rho_1=48-8\beta_2$ and $\rho_2=42-8\beta_1$, and dividing by $x_j^\top x_j=10$:

| pass | update | $\rho_j$ | $S(\rho_j,10)$ | new $\beta_j$ |
|:-:|:--|--:|--:|--:|
| 0 | start | — | — | $\beta=(0,\;0)$ |
| 1 | $\beta_1$ | $48-8(0)=48.000$ | 38.000 | **3.8000** |
| 1 | $\beta_2$ | $42-8(3.8)=11.600$ | 1.600 | **0.1600** |
| 2 | $\beta_1$ | $48-8(0.16)=46.720$ | 36.720 | **3.6720** |
| 2 | $\beta_2$ | $42-8(3.672)=12.624$ | 2.624 | **0.2624** |
| 3 | $\beta_1$ | $48-8(0.2624)=45.901$ | 35.901 | **3.5901** |
| 3 | $\beta_2$ | $42-8(3.5901)=13.279$ | 3.279 | **0.3279** |
| 4 | $\beta_1$ | $48-8(0.3279)=45.377$ | 35.377 | **3.5377** |
| 4 | $\beta_2$ | $42-8(3.5377)=13.698$ | 3.698 | **0.3698** |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| ∞ | converged | — | — | $\beta=(\mathbf{3.4444},\;\mathbf{0.4444})$ |

**Verifying the fixed point algebraically** (both coefficients positive, so both are in the "active" branch):

$$
\beta_1=\frac{48-10-8\beta_2}{10}=\frac{38-8\beta_2}{10},
\qquad
\beta_2=\frac{42-10-8\beta_1}{10}=\frac{32-8\beta_1}{10}
$$

Substitute the second into the first:
$$
10\beta_1=38-\frac{8(32-8\beta_1)}{10}=38-25.6+6.4\beta_1
\;\Longrightarrow\;
3.6\beta_1=12.4
\;\Longrightarrow\;
\beta_1=\frac{12.4}{3.6}=3.4\overline{4}
$$
$$
\beta_2=\frac{32-8(3.4444)}{10}=\frac{32-27.556}{10}=0.4\overline{4}\quad✓
$$

### 8.4 Finding the exact $\lambda$ at which $\beta_2$ dies

Write $\gamma=\lambda/2$. **Suppose** $\beta_2=0$. Then $\beta_1=\dfrac{48-\gamma}{10}$. This is consistent only if the "keep $\beta_2$ at zero" condition holds:

$$
\big|\rho_2\big|=\big|42-8\beta_1\big|\le\gamma
\;\Longrightarrow\;
\left|42-\frac{8(48-\gamma)}{10}\right|\le\gamma
\;\Longrightarrow\;
|3.6+0.8\gamma|\le\gamma
\;\Longrightarrow\;
3.6\le0.2\gamma
\;\Longrightarrow\;
\gamma\ge18
$$

$$
\boxed{\;\lambda\ge36 \;\Rightarrow\; \hat\beta_2=0\;}
$$

### 8.5 Ridge vs LASSO on the *same* correlated data

| $\lambda$ | $\hat\beta^{\text{LASSO}}$ | $\hat\beta^{\text{ridge}}$ | comment |
|---:|:--|:--|:--|
| 0 | $(4.000,\;1.000)$ | $(4.000,\;1.000)$ | both = OLS |
| 2 | $(3.944,\;0.944)$ | $(3.000,\;1.500)$ | ridge is already **redistributing** |
| 10 | $(3.722,\;0.722)$ | $(1.857,\;1.357)$ | ridge has nearly equalised them |
| 20 | $(3.444,\;0.444)$ | $(1.321,\;1.048)$ | |
| **36** | $(\mathbf{3.000},\;\mathbf{0})$ | $(0.912,\;0.754)$ | ← LASSO **selects** feature 1 |
| 40 | $(2.800,\;\mathbf{0})$ | $(0.847,\;0.704)$ | |
| 60 | $(1.800,\;\mathbf{0})$ | $(0.625,\;0.529)$ | ridge still has two non-zeros |

Check one ridge entry by hand ($\lambda=2$): $\begin{pmatrix}12&8\\8&12\end{pmatrix}\beta=\begin{pmatrix}48\\42\end{pmatrix}$, $\det=144-64=80$,
$\beta_1=\frac{12(48)-8(42)}{80}=\frac{240}{80}=3$, $\beta_2=\frac{12(42)-8(48)}{80}=\frac{120}{80}=1.5$ ✓

### 8.6 The single most important behavioural difference

Look at the $\lambda=2$ row: OLS said $(4,\,1)$, ridge says $(3,\,1.5)$. **Ridge moved $\beta_2$ *up*.** That is not a bug.

- **Ridge with correlated predictors: shares the credit.** The $L_2$ penalty is minimised, for a fixed total effect, by splitting it evenly ($3^2+1.5^2=11.25 < 4^2+1^2=17$). Ridge pulls correlated coefficients *toward each other*. It keeps everything, at reduced weight — a **grouping** effect.
- **LASSO with correlated predictors: picks a winner.** The $L_1$ penalty is indifferent to how the total is split ($|3|+|1.5|=|4|+|0.5|$), so nothing pulls the coefficients together; instead the data decides, and the weaker of two correlated features gets zeroed. LASSO tends to pick **one representative from a correlated group, arbitrarily**, and drop the rest.

That last property is both LASSO's strength (a sparse, readable model) and its weakness (if $x_1$ and $x_2$ are two measurements of the same physical quantity, LASSO keeps one at random and a tiny data perturbation can flip which one — *unstable selection*). **Elastic Net exists precisely to fix this.**

---

## 9. Elastic Net

> **Elastic net combines the loss function for ridge regression and LASSO.**

$$
\boxed{\;
L(\beta)=\sum_{i=1}^{n}\Big(y_i-\sum_{j=1}^{p}\beta_j x_{ij}\Big)^{2}
+\sum_{j=1}^{p}\Big(\lambda_r\,\beta_j^{2}+\lambda_\ell\,|\beta_j|\Big)
\;}
$$

Here you regularize the loss function using **both** the sum of squared parameters **and** the sum of absolute values of the parameters.

- $\lambda_r$ (the ridge / $L_2$ knob) → shrinks smoothly, handles correlated groups, stabilises the fit
- $\lambda_\ell$ (the LASSO / $L_1$ knob) → creates exact zeros, does the selection

You get sparsity **and** stability: correlated features are kept or dropped **together as a group**, at shared weight.

### 9.1 The coordinate-descent update

The $L_2$ term is smooth, so it just adds to the denominator; the $L_1$ term still does the thresholding:

$$
\boxed{\;\beta_j\;\leftarrow\;\frac{S\!\left(\rho_j,\;\lambda_\ell/2\right)}{x_j^\top x_j+\lambda_r}\;}
$$

Note how this contains all three methods as special cases:

| $\lambda_\ell$ | $\lambda_r$ | reduces to |
|:-:|:-:|:--|
| 0 | 0 | OLS: $\beta_j=\rho_j/(x_j^\top x_j)$ |
| 0 | $>0$ | Ridge: $\beta_j=\rho_j/(x_j^\top x_j+\lambda_r)$ — no threshold, no zeros |
| $>0$ | 0 | LASSO: $\beta_j=S(\rho_j,\lambda_\ell/2)/(x_j^\top x_j)$ |
| $>0$ | $>0$ | Elastic Net |

### 9.2 Numerical example — same data as Example B

Take $\lambda_\ell=20$ (so $\lambda_\ell/2=10$) and $\lambda_r=10$. Denominator $=x_j^\top x_j+\lambda_r=10+10=20$.

Fixed-point equations:
$$
\beta_1=\frac{48-10-8\beta_2}{20}=\frac{38-8\beta_2}{20},
\qquad
\beta_2=\frac{42-10-8\beta_1}{20}=\frac{32-8\beta_1}{20}
$$

Substitute:
$$
20\beta_1=38-\frac{8(32-8\beta_1)}{20}=38-12.8+3.2\beta_1
\;\Longrightarrow\;
16.8\beta_1=25.2
\;\Longrightarrow\;
\beta_1=\mathbf{1.5}
$$
$$
\beta_2=\frac{32-8(1.5)}{20}=\frac{20}{20}=\mathbf{1.0}
$$

Sanity check that $\beta_2$ should indeed be non-zero: $\rho_2=42-8(1.5)=30>10=\lambda_\ell/2$ ✓

### 9.3 All four methods, one table (Example B data)

| method | knobs | $\hat\beta_1$ | $\hat\beta_2$ | zeros? | $\hat\beta_1/\hat\beta_2$ |
|:--|:--|--:|--:|:-:|--:|
| OLS | — | 4.000 | 1.000 | no | 4.00 |
| Ridge | $\lambda=10$ | 1.857 | 1.357 | no | 1.37 |
| LASSO | $\lambda=20$ | 3.444 | 0.444 | no | 7.75 |
| LASSO | $\lambda=40$ | 2.800 | **0.000** | **yes** | ∞ |
| Elastic Net | $\lambda_r=10,\lambda_\ell=20$ | 1.500 | 1.000 | no | 1.50 |

Read the last column: **ridge pushes the ratio toward 1** (equalise), **LASSO pushes it toward ∞** (select), **elastic net sits in between and can do either**, depending on how you set the two knobs.

### 9.4 The scikit-learn parametrisation of elastic net

`sklearn` reparametrises the two knobs as one strength + one mixing fraction:

$$
\frac{1}{2n}\|y-Z\beta\|^2+\alpha\rho\|\beta\|_1+\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2
$$

with `alpha` $=\alpha$ and `l1_ratio` $=\rho\in[0,1]$.

- `l1_ratio = 1` → pure LASSO
- `l1_ratio = 0` → pure ridge
- `l1_ratio = 0.5` → equal mix (a common default starting point)

Mapping from my $(\lambda_\ell,\lambda_r)$:
$$
\alpha=\frac{\lambda_\ell+2\lambda_r}{2n},\qquad \rho=\frac{\lambda_\ell}{\lambda_\ell+2\lambda_r}
$$
For Example B ($n=4$, $\lambda_\ell=20$, $\lambda_r=10$): $\alpha=\frac{20+20}{8}=5.0$, $\rho=\frac{20}{40}=0.5$. Feeding those to `ElasticNet` returns $(1.5,\,1.0)$ — matching the hand calculation exactly.

---

## 10. Choosing $\lambda$ / `alpha` — from the very basics

> **The $\lambda$ value should be chosen by *cross-validation*, to obtain good performance on both the train and the test sets.**

This is the part that trips people up, so let us build it from nothing.

### 10.1 Why you cannot choose $\lambda$ by minimising the training loss

Obvious but worth stating: $\lambda=0$ *always* gives the lowest possible training SSE, because that is literally what OLS minimises. Any $\lambda>0$ makes training error worse — on purpose.

So if you tune $\lambda$ on the training set you will always get $\lambda=0$ and no regularization at all. **$\lambda$ must be judged on data the model has not seen.**

### 10.2 The bias–variance trade-off, in words and in numbers

Prediction error on new data decomposes as

$$
\mathbb{E}\big[(y-\hat y)^2\big]
= \underbrace{\text{Bias}^2}_{\text{systematically wrong}}
+ \underbrace{\text{Variance}}_{\text{jumpy across datasets}}
+ \underbrace{\sigma^2}_{\text{irreducible noise}}
$$

| $\lambda$ | Bias | Variance | Training error | Test error |
|:--|:--|:--|:--|:--|
| $\lambda=0$ (OLS) | lowest (unbiased) | **highest** | lowest | high — *overfitting* |
| moderate $\lambda$ | some | reduced | slightly worse | **lowest** ← the sweet spot |
| $\lambda\to\infty$ | **highest** ($\hat\beta\to0$) | lowest (≈0) | worst | high — *underfitting* |

```
error
  │╲                                    ╱
  │ ╲            test error           ╱
  │  ╲___                          ╱
  │      ╲___                  ╱
  │          ╲______      ___╱
  │                 ╲____╱   ← minimum: the λ we want
  │  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  │ ╱  training error (only ever goes up with λ)
  │╱
  └────────────────────────────────────── λ  (log scale)
   0                                    ∞
```

Cross-validation is just a way of estimating that U-shaped test-error curve **without touching your real test set.**

### 10.3 What $k$-fold cross-validation actually does

1. Shuffle the training data and split it into $k$ equal parts (**folds**). $k=5$ or $k=10$ is standard; $k=n$ is leave-one-out.
2. For a candidate $\lambda$:
   - For each fold $m=1\ldots k$: **train** on the other $k-1$ folds, **predict** the held-out fold $m$, record its MSE.
   - Average the $k$ MSEs → $\text{CV}(\lambda)$.
3. Repeat for every $\lambda$ on your grid.
4. Pick the $\lambda$ that minimises $\text{CV}(\lambda)$.
5. **Refit on all the training data** with that $\lambda$ — that is your final model.

```
data:  [ F1 ][ F2 ][ F3 ][ F4 ]

round 1:  TEST  train train train   → MSE₁
round 2:  train  TEST train train   → MSE₂
round 3:  train train  TEST train   → MSE₃
round 4:  train train train  TEST   → MSE₄
                                       ↓
                     CV(λ) = (MSE₁+MSE₂+MSE₃+MSE₄)/4
```

Every observation is used for training $k-1$ times and for validation exactly once. That is why CV is far more stable than a single train/validation split.

### 10.4 How to build the $\lambda$ grid (do not guess)

Three rules:

**(1) Search on a log scale.** $\lambda$ acts multiplicatively. A grid like $0.001,0.01,0.1,1,10,100$ is right; $1,2,3,\dots,100$ is wasteful and covers nothing.

**(2) Start from $\lambda_{\max}$ — the smallest $\lambda$ that kills *everything*.** There is a formula. Every coefficient is zero as soon as $\lambda/2$ exceeds every $|x_j^\top y_c|$:

$$
\boxed{\;\lambda_{\max}=2\max_j\big|x_j^\top y_c\big|\;}
\qquad\text{(in sklearn's units: }\alpha_{\max}=\tfrac{1}{n}\max_j|z_j^\top y_c|\text{)}
$$

Above $\lambda_{\max}$, nothing changes — the model is just the intercept. So there is no point searching there.

**(3) Sweep down about three decades from $\lambda_{\max}$**, in ~50–100 log-spaced steps:
$$
\lambda \in \big[\,10^{-3}\lambda_{\max},\;\lambda_{\max}\,\big]
$$
This is exactly what `LassoCV` does by default (`n_alphas=100`, `eps=1e-3`).

**Sanity check after fitting:** if the winning $\lambda$ sits at either *end* of your grid, the grid was too narrow. Extend it and refit.

### 10.5 `min-CV` vs the `1-SE` rule

The CV curve is itself estimated from only $k$ numbers, so it is noisy. Two standard choices:

- **`lambda.min`** — the $\lambda$ with the smallest $\text{CV}(\lambda)$. Best raw predictive accuracy.
- **`lambda.1se`** — the **largest** $\lambda$ whose CV error is still within one standard error of the minimum:
 $$
 \text{SE}(\lambda)=\frac{\text{std dev of the }k\text{ fold-MSEs}}{\sqrt k},
 \qquad
 \text{choose } \max\Big\{\lambda:\ \text{CV}(\lambda)\le \text{CV}_{\min}+\text{SE}(\lambda_{\min})\Big\}
 $$

The 1-SE rule says: *among all models that are statistically indistinguishable from the best one, take the simplest.* It gives a sparser, more interpretable, more reproducible model at negligible cost in accuracy. **For feature selection — which is usually why you reached for LASSO — prefer `1-SE`.** For pure prediction, `min` is fine.

### 10.6 The rules you must not break

| Rule | Why |
|:--|:--|
| Standardize **inside** each fold, using training-fold statistics only | Otherwise the test fold's mean and SD leak into training → optimistic CV |
| Never let the final held-out test set influence $\lambda$ | Otherwise your reported test error is not an honest estimate |
| Use the *same* folds for every $\lambda$ | Otherwise you are comparing $\lambda$'s across different noise, not different models |
| Shuffle before splitting if the data has any ordering (time, batch, temperature) | Contiguous folds on ordered data give wildly pessimistic CV |
| Never penalise the intercept | Shrinking $\beta_0$ biases all predictions toward zero |
| If observations are grouped (same molecule, same subject, same trajectory), split **by group** | Otherwise near-duplicates land in train and test → leakage |

### 10.7 Also tune `l1_ratio` for elastic net

Elastic net has **two** knobs, so cross-validate over a 2-D grid: a few values of `l1_ratio` (e.g. $0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0$) × the usual `alpha` path for each. `ElasticNetCV` does this for you. Note the grid is deliberately biased toward 1 — small differences near the pure-LASSO end matter most.

---

## 11. Worked Example C: complete cross-validation

Everything above, on one dataset, with every number shown.

### 11.1 The data

$n=12$, $p=3$. Features $x_1,x_2$ are real; **$x_3$ is a decoy** — I generated it to have nothing to do with $y$. The truth is $y=20+3x_1-2x_2+\varepsilon$ with tiny noise $\varepsilon=\pm1$. A good method must recover $\beta_3=0$.

| $i$ | $x_1$ | $x_2$ | $x_3$ | $y$ |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 2 | 1 | 7 | 25 |
| 2 | 5 | 4 | 2 | 26 |
| 3 | 1 | 3 | 9 | 17 |
| 4 | 8 | 2 | 4 | 41 |
| 5 | 4 | 6 | 11 | 19 |
| 6 | 7 | 5 | 1 | 31 |
| 7 | 3 | 8 | 6 | 14 |
| 8 | 9 | 7 | 12 | 32 |
| 9 | 6 | 10 | 3 | 18 |
| 10 | 10 | 9 | 8 | 33 |
| 11 | 2 | 12 | 5 | 1 |
| 12 | 8 | 11 | 10 | 22 |

Column statistics (whole dataset): $\bar x=(5.4167,\;6.5,\;6.5)$, $s=(2.9004,\;3.4521,\;3.4521)$, $\bar y=23.25$.

Correlations with $y$: $\text{corr}(x_1,y)=+0.719$, $\text{corr}(x_2,y)=-0.515$, $\text{corr}(x_3,y)=-0.044$ ← the decoy is visibly useless, as designed.

### 11.2 Setting up the search grid

On standardized columns, $\tfrac1n|z_j^\top y_c| = (7.263,\;5.202,\;0.447)$, so

$$
\alpha_{\max}=\max_j\tfrac1n|z_j^\top y_c|=\mathbf{7.263}
$$

Any $\alpha\ge7.263$ gives the all-zero model. So a grid running from about $10$ down to $0.01$ brackets everything of interest. I use $\alpha\in\{0.01,\,0.03,\,0.1,\,0.3,\,1,\,3,\,10,\,30\}$.

### 11.3 The folds

$k=4$, assigned by interleaving (**not** contiguous blocks — the rows are ordered):

| fold | held-out rows |
|:-:|:--|
| F1 | 1, 5, 9 |
| F2 | 2, 6, 10 |
| F3 | 3, 7, 11 |
| F4 | 4, 8, 12 |

For each fold: standardize with the 9 training rows' $\bar x_j, s_j$; centre $y$ with the training $\bar y$; fit LASSO; apply the **same** $\bar x_j, s_j, \bar y$ to the 3 held-out rows; compute MSE.

### 11.4 The full CV table

| `alpha` | F1 MSE | F2 MSE | F3 MSE | F4 MSE | **CV-MSE** | SE | coefficients on full data (standardized) |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 0.01 | 0.614 | 2.434 | 0.770 | 1.033 | 1.2127 | 0.416 | $(8.831,\,-7.129,\,-0.091)$ |
| 0.03 | 0.629 | 2.297 | 0.700 | 0.937 | 1.1406 | 0.391 | $(8.805,\,-7.106,\,-0.072)$ |
| **0.10** | 0.667 | 1.891 | 0.622 | 0.694 | **0.9685** ← min | 0.308 | $(8.713,\,-7.023,\,-0.006)$ |
| **0.30** | 0.799 | 1.370 | 1.921 | 0.981 | 1.2677 | 0.248 | $(8.456,\,-6.768,\,\mathbf{0})$ ← **1-SE choice** |
| 1.00 | 1.546 | 3.096 | 24.915 | 6.031 | 8.8970 | 5.420 | $(7.558,\,-5.869,\,0)$ |
| 3.00 | 6.100 | 17.652 | 248.234 | 53.760 | 81.4362 | 56.518 | $(4.992,\,-3.303,\,0)$ |
| 10.0 | 21.420 | 89.667 | 329.716 | 186.160 | 156.7407 | 66.831 | $(0,\,0,\,0)$ |
| 30.0 | 21.420 | 89.667 | 329.716 | 186.160 | 156.7407 | 66.831 | $(0,\,0,\,0)$ |

**Read the table:**

- **The U-shape is right there in the CV-MSE column:** $1.213 \to 1.141 \to \mathbf{0.969} \to 1.268 \to 8.90 \to 81.4 \to 156.7$. Too little regularization on the left, too much on the right.
- The last two rows are **identical** — both are past $\alpha_{\max}=7.263$, so both give the null model. That confirms the grid is wide enough (the winner is comfortably interior, not at an edge).
- Watch the **decoy $\beta_3$** collapse down the table: $-0.091 \to -0.072 \to -0.006 \to \mathbf{0}$. Ridge would never have given you that final zero.
- **`min-CV` choice:** $\alpha=0.10$, CV-MSE $=0.9685$, SE $=0.308$.
- **1-SE rule:** threshold $=0.9685+0.308=1.2765$. The rows with CV-MSE $\le1.2765$ are $\alpha=0.01,0.03,0.10,0.30$. The **largest** of these is $\alpha=\mathbf{0.30}$ → and *that* model has $\beta_3$ **exactly 0**. The 1-SE rule bought us a genuinely sparser model for a CV-MSE cost of $1.268$ vs $0.969$ — statistically indistinguishable.

### 11.5 Back to physical units

Take the 1-SE model, $\alpha=0.30$. Standardized coefficients $\hat\beta^{z}=(8.456,\,-6.768,\,0)$. Divide by $s_j=(2.9004,\,3.4521,\,3.4521)$:

$$
\hat\beta^{\text{orig}}=\left(\frac{8.456}{2.9004},\;\frac{-6.768}{3.4521},\;0\right)=(\,\mathbf{2.916},\;\mathbf{-1.960},\;\mathbf{0}\,)
$$

$$
\hat\beta_0=\bar y-\sum_j\hat\beta^{\text{orig}}_j\bar x_j
=23.25-\big[2.916(5.4167)+(-1.960)(6.5)+0(6.5)\big]=\mathbf{20.198}
$$

**Final model:** $\;\hat y = 20.198+2.916\,x_1-1.960\,x_2$

**The truth was:** $\;y = 20 + 3x_1 - 2x_2$

LASSO recovered the right *structure* (it found and eliminated the decoy) and coefficients within ~3% of the truth. The small downward bias — $2.916$ instead of $3$, $-1.960$ instead of $-2$ — is the **shrinkage**, and it is the price you pay for the variance reduction. It is not an error.

> If you need unbiased coefficient values, a common two-step is the **relaxed LASSO / debiasing**: use LASSO only to *select* the non-zero set, then refit plain OLS on just those features. Here that would give back $\approx(3,-2)$ exactly.

---

## 12. The $\lambda$ ↔ `alpha` conversion

This is where hand calculations and library output disagree, and it is purely a matter of convention.

| | loss function | knob |
|:--|:--|:--|
| **These notes / most textbooks** | $\displaystyle \lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_1$ | $\lambda$ |
| **`sklearn.linear_model.Lasso`** | $\displaystyle \frac{1}{2n}\lVert y-X\beta\rVert^2+\alpha\lVert\beta\rVert_1$ | `alpha` |
| **`sklearn.linear_model.Ridge`** | $\displaystyle \lVert y-X\beta\rVert^2+\alpha\lVert\beta\rVert_2^2$ | `alpha` |
| **`glmnet` (R)** | $\displaystyle \frac{1}{2n}\lVert y-X\beta\rVert^2+\lambda\big[\tfrac{1-\rho}{2}\lVert\beta\rVert_2^2+\rho\lVert\beta\rVert_1\big]$ | `lambda`, `alpha`$=\rho$ |

Multiply the sklearn Lasso objective by $2n$: $\|y-X\beta\|^2+2n\alpha\|\beta\|_1$. So

$$
\boxed{\;\lambda_{\text{ours}}=2n\,\alpha_{\text{sklearn}}
\qquad\Longleftrightarrow\qquad
\alpha_{\text{sklearn}}=\frac{\lambda_{\text{ours}}}{2n}\;}
$$

**Careful:** for `Ridge`, sklearn does **not** divide by $n$, so there $\alpha_{\text{sklearn}}=\lambda_{\text{ours}}$ directly. Same word, two different conventions inside the same library.

### Verification on Example A ($n=4$)

| $\lambda$ (ours) | $\alpha=\lambda/8$ | our hand answer | `Lasso(alpha=...)` output |
|---:|---:|:--|:--|
| 2 | 0.25 | $(2.75,\;0.25)$ | $(2.75,\;0.25)$ ✓ |
| 4 | 0.50 | $(2.50,\;0)$ | $(2.50,\;0)$ ✓ |
| 8 | 1.00 | $(2.00,\;0)$ | $(2.00,\;0)$ ✓ |

And on Example B ($n=4$): $\lambda=40\Rightarrow\alpha=5.0$; `Lasso(alpha=5.0)` returns $(2.8,\;0)$, matching Section 8.5 exactly.

---

## 13. Python code to reproduce every number

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def soft(z, g):
    """Soft-thresholding operator S(z, gamma)."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)

def lasso_cd(X, y, lam_l, lam_r=0.0, iters=1000, tol=1e-12):
    """Coordinate descent for  ||y - Xb||^2 + lam_r*||b||^2 + lam_l*||b||_1
       (X and y must already be centred)."""
    n, p = X.shape
    b   = np.zeros(p)
    nrm = (X**2).sum(axis=0)
    G   = X.T @ X            # precompute Gram matrix
    c   = X.T @ y
    for _ in range(iters):
        b_old = b.copy()
        for j in range(p):
            rho  = c[j] - G[j] @ b + G[j, j] * b[j]   # = xj . partial residual
            b[j] = soft(rho, lam_l/2) / (nrm[j] + lam_r)
        if np.max(np.abs(b - b_old)) < tol:
            break
    return b

# ---------- Example A : orthogonal design ----------
XA = np.array([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]])
yA = np.array([2., 2., 7., 9.])
ycA = yA - yA.mean()                       # (-3, -3, 2, 4)

print("OLS      :", np.linalg.solve(XA.T@XA, XA.T@ycA))     # [3.0, 0.5]
for lam in [0, 2, 4, 8, 16, 24]:
    bl = soft(XA.T@ycA, lam/2) / 4                          # closed form (orthogonal)
    br = np.linalg.solve(XA.T@XA + lam*np.eye(2), XA.T@ycA) # ridge closed form
    print(f"lam={lam:3d}  lasso={bl}  ridge={br}")

# cross-check against sklearn  (alpha = lam / (2n),  n = 4)
for lam in [2, 4, 8]:
    m = Lasso(alpha=lam/8).fit(XA, yA)
    print(f"sklearn lam={lam}: coef={m.coef_}  intercept={m.intercept_}")

# ---------- Example B : correlated design ----------
XB = np.array([[-2.,-1.], [-1.,-2.], [1.,2.], [2.,1.]])
yB = np.array([2., 3., 15., 20.])
ycB = yB - yB.mean()                       # (-8, -7, 5, 10)

print("\nX'X =", XB.T@XB, "  X'y =", XB.T@ycB)             # [[10,8],[8,10]], [48,42]
print("OLS  :", np.linalg.solve(XB.T@XB, XB.T@ycB))        # [4.0, 1.0]
for lam in [0, 2, 10, 20, 36, 40, 60]:
    print(f"lam={lam:3d}  lasso={lasso_cd(XB,ycB,lam)}"
          f"  ridge={np.linalg.solve(XB.T@XB+lam*np.eye(2), XB.T@ycB)}")

# elastic net:  lam_l = 20, lam_r = 10  ->  (1.5, 1.0)
print("enet :", lasso_cd(XB, ycB, lam_l=20, lam_r=10))
print("enet sklearn:", ElasticNet(alpha=(20+2*10)/(2*4),
                                  l1_ratio=20/(20+2*10)).fit(XB, yB).coef_)

# ---------- Example C : cross-validation ----------
x1 = np.array([2,5,1,8,4,7,3,9,6,10,2,8], float)
x2 = np.array([1,4,3,2,6,5,8,7,10,9,12,11], float)
x3 = np.array([7,2,9,4,11,1,6,12,3,8,5,10], float)      # decoy
X  = np.column_stack([x1, x2, x3])
y  = 20 + 3*x1 - 2*x2 + np.array([1,-1,0,1,-1,0,1,-1,0,1,-1,0], float)

folds = [[0,4,8], [1,5,9], [2,6,10], [3,7,11]]          # interleaved, not blocks
grid  = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# where the grid should start
mu, sd = X.mean(0), X.std(0)
print("\nalpha_max =", np.max(np.abs(((X-mu)/sd).T @ (y-y.mean()))) / len(y))

rows = []
for a in grid:
    mses = []
    for f in folds:
        te = np.array(f)
        tr = np.array([i for i in range(12) if i not in f])
        m_, s_, ym = X[tr].mean(0), X[tr].std(0), y[tr].mean()   # TRAIN stats only
        fit  = Lasso(alpha=a, fit_intercept=False, max_iter=200000
                     ).fit((X[tr]-m_)/s_, y[tr]-ym)
        pred = ((X[te]-m_)/s_) @ fit.coef_ + ym
        mses.append(np.mean((y[te]-pred)**2))
    full = Lasso(alpha=a, fit_intercept=False, max_iter=200000
                 ).fit((X-mu)/sd, y-y.mean())
    cv, se = np.mean(mses), np.std(mses, ddof=1)/np.sqrt(len(folds))
    rows.append((a, cv, se, full.coef_))
    print(f"alpha={a:6}  folds={np.round(mses,3)}  CV={cv:9.4f}  SE={se:7.3f}  "
          f"coef={np.round(full.coef_,3)}")

best = min(rows, key=lambda r: r[1])
thr  = best[1] + best[2]
one_se = max(r[0] for r in rows if r[1] <= thr)
print(f"\nmin-CV alpha = {best[0]}   |   1-SE alpha = {one_se}")

# back-transform the 1-SE model to original units
f = Lasso(alpha=one_se, fit_intercept=False, max_iter=200000).fit((X-mu)/sd, y-y.mean())
b = f.coef_ / sd
print("coefficients:", np.round(b, 3), "  intercept:", round(y.mean() - mu@b, 3))
```

**The lazy (and correct) way to do the CV in practice:**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)

# the pipeline is what guarantees standardization happens INSIDE each fold
lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, n_alphas=100, eps=1e-3))
lasso.fit(X, y)
print("chosen alpha:", lasso[-1].alpha_)
print("coefficients:", lasso[-1].coef_)

enet = make_pipeline(StandardScaler(),
                     ElasticNetCV(cv=cv,
                                  l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1],
                                  n_alphas=100))
enet.fit(X, y)
print("alpha:", enet[-1].alpha_, " l1_ratio:", enet[-1].l1_ratio_)
```

---

## 14. Cheat sheet

### The three penalties

| | Ridge ($L_2$) | LASSO ($L_1$) | Elastic Net |
|:--|:--|:--|:--|
| Penalty | $\lambda\sum\beta_j^2$ | $\lambda\sum\lvert\beta_j\rvert$ | $\lambda_r\sum\beta_j^2+\lambda_\ell\sum\lvert\beta_j\rvert$ |
| Constraint region | circle / sphere | diamond (has corners) | rounded diamond |
| Closed-form solution? | **Yes**: $(X^\top X+\lambda I)^{-1}X^\top y$ | **No** — $\lvert\beta\rvert$ not differentiable at 0 | No |
| How to solve | direct linear algebra | coordinate descent / LARS | coordinate descent |
| Exact zeros? | **Never** | **Yes** | Yes |
| Does feature selection? | no | **yes** | yes |
| Correlated predictors | splits weight among them (grouping) | picks one, drops the rest (unstable) | keeps the group together |
| Works when $p>n$? | yes | yes, but selects at most $n$ features | yes, no $n$ limit |
| Coordinate update | $\dfrac{\rho_j}{x_j^\top x_j+\lambda}$ | $\dfrac{S(\rho_j,\lambda/2)}{x_j^\top x_j}$ | $\dfrac{S(\rho_j,\lambda_\ell/2)}{x_j^\top x_j+\lambda_r}$ |

### Which one do I use?

- Many features, believe **most are irrelevant**, want an interpretable short list → **LASSO**
- **All** features plausibly matter a little; predictors are collinear; want stability → **Ridge**
- $p\gg n$, or **groups of correlated features** you want kept or dropped together → **Elastic Net** (this is usually the safe default for descriptor-heavy scientific data)

### The workflow, start to finish

1. Split off a **test set** and put it away. Do not look at it again.
2. Choose a $k$-fold CV scheme on the remainder ($k=5$ or $10$; shuffle; split by group if grouped).
3. Build a **log-spaced $\lambda$ grid** from $\lambda_{\max}$ down about three decades.
4. Run CV, **standardizing inside each fold**. Get $\text{CV}(\lambda)$ and $\text{SE}(\lambda)$.
5. Pick $\lambda$ by **min-CV** (prediction) or **1-SE** (selection / interpretation).
6. Check the winner is not at a grid edge. If it is, widen the grid and redo step 4.
7. **Refit on all the training data** at the chosen $\lambda$.
8. Back-transform coefficients to original units; report the selected features.
9. *Now* evaluate once on the held-out test set. That number is your honest performance estimate.

### Numbers to remember from these notes

| quantity | Example A | Example B |
|:--|:--|:--|
| $\hat\beta_{\text{OLS}}$ | $(3.0,\;0.5)$ | $(4.0,\;1.0)$ |
| $\lambda$ at which $\beta_2$ hits 0 | $4$ | $36$ |
| $\lambda_{\max}$ (all coefficients 0) | $24$ | $96$ |
| ridge at that same $\lambda$ | $(1.5,\;0.25)$ — no zeros | $(0.912,\;0.754)$ — no zeros |

---

*Lecture 12 — LASSO, ridge, and elastic net. Every numerical result in this document was computed and cross-checked against `scikit-learn`; the code in Section 13 reproduces all of them.*
