# Lecture 13 — Bias, Variance, and the Bias–Variance Trade-off

**Source:** handwritten lecture board (NPTEL, IISc) — screenshots in `lec_13/`
**Goal of the lecture:** show *why* a machine-learning model makes errors, split that error into three named pieces, and prove the split:

$$
\text{Total (squared) error} \;=\; \big(\text{Bias}\big)^2 \;+\; \text{Variance} \;+\; \text{Irreducible error}
$$

Everything below is built from scratch: every symbol is defined before it is used, every algebraic step is shown, and each idea is followed by arithmetic you can check with a pen.

---

## 0. The question this lecture answers

You fit a model. It makes mistakes. The mistakes come from three completely different places:

1. **Your model is the wrong shape.** You tried to fit a curve with a straight line. No amount of data fixes this. → **bias**
2. **Your model wobbles.** Give it a slightly different training set and it produces a noticeably different prediction. → **variance**
3. **The world is noisy.** Even a perfect model cannot predict the measurement noise in the thing you are trying to predict. → **irreducible error**

The whole lecture is: define these precisely, then prove that the mean squared error is *exactly* their sum — no cross terms, no leftovers.

---

## 1. Vocabulary from scratch

Before any formula, here is every object we will use.

### 1.1 Data

- **Features** (also: inputs, predictors), written $x$. Example: temperature, pressure, a set of atomic coordinates.
- **Target** (also: output, label, response), written $y$. The thing you want to predict. Example: the measured solubility.
- A **data point** is a pair $(x, y)$.
- A **training set** (also: training dataset) $\mathcal{D} = \{(x_1,y_1), (x_2,y_2), \dots, (x_n,y_n)\}$ is the collection of $n$ data points you use to fit the model.

### 1.2 Models

- $f$ — the **true model**. The real, unknown function of nature that generates the target: if you knew $f$ and had no noise, you would predict perfectly. $f(x)$ is a fixed number for a fixed $x$; it is *not* random.
- $\hat{f}$ — the **fitted model** (read "f-hat"). This is what your learning algorithm produces *after* being trained on one particular training set $\mathcal{D}$. The hat always means "estimated from data".
- $\hat{f}(x)$ — the prediction of the fitted model at input $x$.

> **Key point that makes the whole lecture work:** $\hat{f}$ is a **random object**. Not because the algorithm is random, but because the *training set* is random — the $n$ points you happened to collect contain random measurement noise. Collect a different training set, get a different $\hat{f}$, get a different prediction $\hat{f}(x)$ at the *same* $x$.

### 1.3 Noise

- $\epsilon$ — the **error in the true model**, i.e. measurement/experimental noise. The lecture assumes it is **normally distributed** with mean $0$ and variance $\sigma^2$:

$$
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

  Read as: "$\epsilon$ is drawn from a Gaussian (normal) distribution whose mean is $0$ and whose variance is $\sigma^2$." Its two consequences we will use over and over:

$$
E[\epsilon] = 0, \qquad E[\epsilon^2] = \sigma^2
$$

  (The second follows from the definition of variance — proved in §1.4.)

- $\sigma^2$ — the **noise variance**. A property of the measurement, *not* of your model. You cannot reduce it by choosing a better model. Hence the name *irreducible error*.

### 1.4 Expectation and variance (the only probability you need)

- $E[\,\cdot\,]$ is the **expectation** = the average value over all the randomness. If a quantity $Z$ takes values $z_1, z_2, \dots, z_m$ with equal probability, then $E[Z] = \frac{1}{m}\sum_k z_k$. It is a *number*, not a random quantity.

- Two properties used constantly (both are just "the average of a sum is the sum of the averages"):

$$
E[A + B] = E[A] + E[B], \qquad E[cA] = c\,E[A]\ \ \text{for a constant } c
$$

- If $c$ is a **constant** (not random), then $E[c] = c$. In particular $E[f] = f$ for a fixed data point.

- **Variance** of a random quantity $Z$:

$$
\mathrm{Var}(Z) \;=\; E\big[(Z - E[Z])^2\big]
$$

  "the average squared distance from its own average" — a measure of spread/wobble.

- **Useful identity** (derived in §6.2 exactly as on the board):

$$
\mathrm{Var}(Z) = E[Z^2] - (E[Z])^2 \quad\Longleftrightarrow\quad E[Z^2] = \mathrm{Var}(Z) + (E[Z])^2
$$

  Applying it to $\epsilon$, where $E[\epsilon] = 0$: $\ \sigma^2 = \mathrm{Var}(\epsilon) = E[\epsilon^2] - 0^2 = E[\epsilon^2]$. ✔ (This is the line on the board: $\sigma^2 = E[(\epsilon - \cancel{E(\epsilon)})^2] = E[\epsilon^2]$.)

- **Independence.** Two random quantities $A$ and $B$ are independent if knowing one tells you nothing about the other. Then $E[AB] = E[A]\,E[B]$. We will use this for the noise at the *test* point and the model $\hat f$ fitted on the *training* set — they come from different measurements, so they are independent.

### 1.5 What exactly is the expectation taken over?

This is the single most common source of confusion. In this lecture, $E[\cdot]$ averages over **two** independent sources of randomness:

| Source | What it randomizes | Symbol affected |
|---|---|---|
| The random training set you happened to collect | the fitted model | $\hat f$, hence $\hat f(x)$ |
| The random noise in the target at the query point | the observed value you are judged against | $\epsilon$, hence $y$ |

The input $x$ is held **fixed** — the board says "the analysis is done for a given data point". So $f = f(x)$ is a plain constant throughout the derivation.

---

## 2. Bias — definition

> **Board:** *Bias: Error in model predictions as ML approximates a real-world variable with a simple, presupposed model architecture. Simpler models introduce higher bias since typically they have fewer parameters.*

Formally, for a fixed input $x$:

$$
\boxed{\ \mathrm{Bias}(\hat f) \;=\; E[\hat f] - f\ }
$$

In words: train the model on *many* different training sets, average all the predictions you get at this $x$, and compare that average with the truth. If the average is off, the model is **biased** — the error is systematic, baked into the *shape* of the model, and it does not average away.

**Where bias comes from:** you *presupposed* an architecture. You decided "the relationship is a straight line" or "a 3-parameter Arrhenius form" before looking. If reality is not in that family, no training set can save you.

**Consequence (board):** bias leads to **underfitting** — *the model misses important relationships between the target variable and the features*.

**Dartboard picture:** bias = your darts all land in a tight cluster, but 10 cm to the left of the bull's-eye.

---

## 3. Variance — definition

> **Board:** *Variance: Error in prediction due to a model's sensitivity to small fluctuations in the training dataset, since a model may be too complex and thus captures noise as a signal.*

Formally, for a fixed input $x$:

$$
\boxed{\ \mathrm{Var}(\hat f) \;=\; E\big[(\hat f - E[\hat f])^2\big]\ }
$$

In words: how much does the prediction at this $x$ jump around when you re-train on a different training set? A flexible model with many parameters will bend to follow the *noise* in whatever points it was given — so a new sample gives a very different curve.

**Consequence (board):** variance leads to **overfitting** — *the model performs poorly on test data due to an overly complex architecture*.

**Dartboard picture:** variance = your darts are scattered all over the board; their *average* position may be dead-centre, but no single throw is reliable.

---

## 4. The trade-off, stated

> **Board:** *Trade-off: to find an optimally complex model.*

- Make the model **more complex** (more parameters, higher polynomial degree, deeper network): bias ↓, variance ↑.
- Make the model **simpler** (fewer parameters, regularization, fewer features): bias ↑, variance ↓.

You cannot push both down by turning the single knob "complexity". The total error is a **U-shaped** curve in model complexity, and the bottom of the U is the *optimal model complexity*. §11 draws this curve with real numbers.

---

## 5. The statistical setup

Real-world measurements are the truth plus noise:

$$
\boxed{\ y = f + \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\sigma^2)\ }
$$

- $y$ = the **real-world measurement** of the target variable (what you actually observe).
- $f$ = the true model's value at this data point (a constant).
- $\epsilon$ = the noise — *"an estimate of the irreducible error that even a true model will incur"*.

The quantity we want to minimise is the **expected squared error** of the fitted model at this data point:

$$
\text{Error} \;=\; E\big[(y - \hat f\,)^2\big]
$$

where $\hat f$ is the **best-fit model obtained using a given dataset**.

Note the two random things inside: $y$ (through $\epsilon$) and $\hat f$ (through the training set). They are independent.

---

## 6. The derivation (every step)

### 6.1 Step 1 — expand the square

$$
\begin{aligned}
\text{Error} &= E\big[(y - \hat f)^2\big] \\[2pt]
&= E\big[y^2 + \hat f^{\,2} - 2y\hat f\,\big] \\[2pt]
&= E[y^2] \;+\; E[\hat f^{\,2}] \;-\; E[2y\hat f\,]
\end{aligned}
$$

(last line uses only $E[A+B] = E[A]+E[B]$).

We now evaluate the three pieces one at a time. **Piece 2 first.**

### 6.2 Step 2 — rewrite $E[\hat f^{\,2}]$ using the variance identity

Start from the *definition* of variance applied to $\hat f$ and expand the square:

$$
\begin{aligned}
\mathrm{Var}(\hat f) &= E\Big[\big(\hat f - E(\hat f)\big)^2\Big] \\[2pt]
&= E\Big[\big(E(\hat f)\big)^2 + \hat f^{\,2} - 2\,\hat f\,E(\hat f)\Big] \\[2pt]
&= \big(E(\hat f)\big)^2 + E\big(\hat f^{\,2}\big) - 2\,E\big(\hat f\,E(\hat f)\big) \\[2pt]
&= \big(E(\hat f)\big)^2 + E\big(\hat f^{\,2}\big) - 2\big(E(\hat f)\big)^2
\end{aligned}
$$

Two small justifications for the last two lines:
- $E(\hat f)$ is a **number** (an average), not random. So $\big(E(\hat f)\big)^2$ passes through the outer $E[\cdot]$ unchanged, and the constant $2E(\hat f)$ can be pulled out of the last term.
- $E\big(\hat f \cdot E(\hat f)\big) = E(\hat f)\cdot E(\hat f) = \big(E(\hat f)\big)^2$.

Collecting the two identical terms:

$$
\boxed{\ \mathrm{Var}(\hat f) = E\big(\hat f^{\,2}\big) - \big(E(\hat f)\big)^2 \ }
\qquad\Longrightarrow\qquad
\boxed{\ E\big(\hat f^{\,2}\big) = \mathrm{Var}(\hat f) + \big(E(\hat f)\big)^2\ }
$$

Substituting into Step 1:

$$
\text{Error} \;=\; \underbrace{E[y^2]}_{\text{Step 3}} \;+\; \mathrm{Var}(\hat f) + \big(E(\hat f)\big)^2 \;-\; \underbrace{E[2y\hat f\,]}_{\text{Step 4}}
$$

### 6.3 Step 3 — evaluate $E[y^2]$

Recall $y = f + \epsilon$:

$$
E[y^2] = E\big[(f+\epsilon)^2\big] = E[f^2] + E[2f\epsilon] + E[\epsilon^2]
$$

Since the analysis is done **for a given data point**, $f$ is a constant, so $E[f^2] = f^2$ and $E[2f\epsilon] = 2f\,E[\epsilon]$:

$$
E[y^2] = f^2 + 2f\,E(\epsilon) + E[\epsilon^2]
$$

Now use the two facts from $\epsilon \sim \mathcal N(0,\sigma^2)$: $E(\epsilon) = 0$ and $E[\epsilon^2] = \sigma^2$. Therefore

$$
\boxed{\ E[y^2] = f^2 + 0 + \sigma^2 = f^2 + \sigma^2\ }
$$

### 6.4 Step 4 — evaluate the cross term $E[2y\hat f\,]$

$$
\begin{aligned}
E[2y\hat f\,] &= 2\,E\big[(f + \epsilon)\,\hat f\,\big] \\[2pt]
&= 2\Big[E\big(f\hat f\big) + E\big(\epsilon \hat f\big)\Big] \\[2pt]
&= 2\Big[f\,E(\hat f) + E(\epsilon)\,E(\hat f)\Big]
\end{aligned}
$$

- $f$ is a constant → pulled out of the first expectation.
- The second term factorises because **the error for a data point and the model prediction are independent** (the noise at the query point had nothing to do with which training set you drew). This is the board's annotation.
- Finally $E(\epsilon) = 0$ kills the second term:

$$
\boxed{\ E[2y\hat f\,] = 2f\,E(\hat f)\ }
$$

### 6.5 Step 5 — put the three pieces back together

$$
\begin{aligned}
\text{Squared error}
&= E[y^2] + \mathrm{Var}(\hat f) + \big(E(\hat f)\big)^2 - E[2y\hat f\,] \\[4pt]
&= \big(f^2 + \sigma^2\big) + \mathrm{Var}(\hat f) + \big(E(\hat f)\big)^2 - 2f\,E(\hat f) \\[4pt]
&= \underbrace{\Big[f^2 - 2f\,E(\hat f) + \big(E(\hat f)\big)^2\Big]}_{\text{a perfect square}} + \mathrm{Var}(\hat f) + \sigma^2 \\[4pt]
&= \big(f - E[\hat f\,]\big)^2 + \mathrm{Var}(\hat f) + \sigma^2
\end{aligned}
$$

and since $f = E[f]$ for a fixed data point, the first bracket is exactly $\big(E[f] - E[\hat f]\big)^2 = \mathrm{Bias}(\hat f)^2$. Hence the result on the board:

$$
\boxed{\ \text{Squared error} \;=\; \big(\mathrm{Bias}(\hat f)\big)^2 \;+\; \mathrm{Var}(\hat f) \;+\; \sigma^2\ }
$$

### 6.6 Reading the result

- **Three non-negative terms.** Each is a square or a variance, so none can be negative and none can cancel another.
- **$\sigma^2$ is a floor.** Even with $\mathrm{Bias}=0$ and $\mathrm{Var}=0$ — a *perfect* model — the expected squared error is $\sigma^2$. If someone reports a test MSE below the noise level, be suspicious (leakage, or the test set is part of training).
- **Bias is squared, variance is not.** They are not in the same units before squaring: bias is measured in units of $y$, variance in units of $y^2$. Comparing "bias vs variance" always means comparing $\mathrm{Bias}^2$ with $\mathrm{Var}$.
- **Only the first two terms are yours to control.** Model choice moves bias and variance; nothing in the model moves $\sigma^2$.

---

## 7. Numerical example 1 — the decomposition on three candidate models (pure hand arithmetic)

**Setup.** Fix one query point $x_0$. Suppose the truth there is

$$
f = 4.00, \qquad \sigma^2 = 0.25 \ \ (\sigma = 0.5)
$$

Imagine that, because of the randomness of the training set, each candidate model produces one of **three equally likely predictions** at $x_0$ (probability $1/3$ each). This is a toy stand-in for "re-train on a new sample, get a new prediction".

| Model | complexity | predictions $\hat f$ over 3 training sets |
|---|---|---|
| **A** | high (flexible) | $3.0,\ 4.0,\ 5.0$ |
| **B** | low (very simple) | $2.9,\ 3.0,\ 3.1$ |
| **C** | medium | $3.6,\ 3.8,\ 4.0$ |

### Model A

$$
E[\hat f] = \tfrac{3.0+4.0+5.0}{3} = 4.00
$$
$$
\mathrm{Bias} = E[\hat f] - f = 4.00 - 4.00 = 0 \ \Rightarrow\ \mathrm{Bias}^2 = 0
$$
$$
\mathrm{Var}(\hat f) = \tfrac{(3.0-4)^2 + (4.0-4)^2 + (5.0-4)^2}{3} = \tfrac{1 + 0 + 1}{3} = 0.6667
$$
$$
\text{Total} = 0 + 0.6667 + 0.25 = \mathbf{0.9167}
$$

### Model B

$$
E[\hat f] = \tfrac{2.9+3.0+3.1}{3} = 3.00, \qquad \mathrm{Bias} = 3.00 - 4.00 = -1.00 \ \Rightarrow\ \mathrm{Bias}^2 = 1.00
$$
$$
\mathrm{Var}(\hat f) = \tfrac{(-0.1)^2 + 0^2 + (0.1)^2}{3} = \tfrac{0.02}{3} = 0.00667
$$
$$
\text{Total} = 1.00 + 0.00667 + 0.25 = \mathbf{1.2567}
$$

### Model C

$$
E[\hat f] = \tfrac{3.6+3.8+4.0}{3} = 3.80, \qquad \mathrm{Bias} = -0.20 \ \Rightarrow\ \mathrm{Bias}^2 = 0.04
$$
$$
\mathrm{Var}(\hat f) = \tfrac{(-0.2)^2 + 0^2 + (0.2)^2}{3} = \tfrac{0.08}{3} = 0.02667
$$
$$
\text{Total} = 0.04 + 0.02667 + 0.25 = \mathbf{0.3167}
$$

### Summary

| Model | $\mathrm{Bias}^2$ | $\mathrm{Var}$ | $\sigma^2$ | **Total** |
|---|---:|---:|---:|---:|
| A (complex) | 0.0000 | 0.6667 | 0.25 | 0.9167 |
| B (too simple) | 1.0000 | 0.0067 | 0.25 | 1.2567 |
| **C (medium)** | 0.0400 | 0.0267 | 0.25 | **0.3167** ← best |

**What to notice:**
- A is **unbiased** and still the second-worst: being right *on average* is worthless if every individual fit is far off.
- B is **rock-steady** and worst of all: stability around the wrong answer is not a virtue.
- C, which is neither unbiased nor the steadiest, wins. **This is the trade-off.** Accepting a little bias bought a large reduction in variance.
- No model can go below $0.25$. That is $\sigma^2$.

### Cross-check of the decomposition against a direct calculation

For model A, compute $E[(y-\hat f)^2]$ directly. Here $y = 4 + \epsilon$, and $\epsilon$ is independent of $\hat f$, so

$$
E[(y-\hat f)^2] = E\big[((4-\hat f) + \epsilon)^2\big] = E[(4-\hat f)^2] + 2\,E[(4-\hat f)]\,E[\epsilon] + E[\epsilon^2]
$$

- $E[(4-\hat f)^2] = \frac{(4-3)^2+(4-4)^2+(4-5)^2}{3} = \frac{2}{3} = 0.6667$
- middle term $= 0$ because $E[\epsilon]=0$
- $E[\epsilon^2] = \sigma^2 = 0.25$

Direct total $= 0.6667 + 0.25 = 0.9167$ — identical to $\mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$. ✔

---

## 8. Numerical example 2 — exact algebra with real regression models

Now a case where the fitted models are genuine least-squares fits, and bias and variance can be computed in closed form.

**Setup.**

- Fixed inputs (design points) $x \in \{1, 2, 3\}$.
- True model: $f(x) = 2x$, so $f(1)=2,\ f(2)=4,\ f(3)=6$.
- Observations: $y_i = 2x_i + \epsilon_i$, with $\epsilon_i \sim \mathcal N(0, \sigma^2)$, $\sigma^2 = 1$.
- Three candidate architectures, each fitted by ordinary least squares:
  - **M0** — constant: $\hat f(x) = c$ (1 parameter)
  - **M1** — straight line: $\hat f(x) = a + bx$ (2 parameters)
  - **M2** — quadratic: $\hat f(x) = a + bx + cx^2$ (3 parameters)

### M0 — the constant model

Least squares for a constant gives the sample mean: $\hat c = \frac{y_1+y_2+y_3}{3}$.

$$
E[\hat c] = \frac{E[y_1]+E[y_2]+E[y_3]}{3} = \frac{2+4+6}{3} = 4
$$

$$
\mathrm{Var}(\hat c) = \mathrm{Var}\!\left(\frac{\epsilon_1+\epsilon_2+\epsilon_3}{3}\right) = \frac{3\sigma^2}{9} = \frac{\sigma^2}{3} = 0.3333
$$

(using: variance of a sum of independent terms is the sum of variances, and $\mathrm{Var}(aZ) = a^2\mathrm{Var}(Z)$.)

Bias at each design point, $\mathrm{Bias}(x) = E[\hat f(x)] - f(x) = 4 - 2x$:

| $x$ | $f(x)$ | $E[\hat f]$ | Bias | $\mathrm{Bias}^2$ | Var | $\sigma^2$ | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 4 | $+2$ | 4.000 | 0.333 | 1 | 5.333 |
| 2 | 4 | 4 | $0$ | 0.000 | 0.333 | 1 | 1.333 |
| 3 | 6 | 4 | $-2$ | 4.000 | 0.333 | 1 | 5.333 |
| **avg** | | | | **2.667** | **0.333** | **1** | **4.000** |

### M1 — the straight line

The true function *is* a straight line, and it lies inside the model family, so least squares is **unbiased**: $E[\hat f(x)] = 2x$ for every $x$, hence $\mathrm{Bias} = 0$ everywhere.

The standard formula for the variance of a simple-regression prediction is

$$
\mathrm{Var}\big(\hat f(x_0)\big) = \sigma^2\left[\frac{1}{n} + \frac{(x_0-\bar x)^2}{S_{xx}}\right],
\qquad S_{xx} = \sum_i (x_i - \bar x)^2
$$

Here $n = 3$, $\bar x = 2$, $S_{xx} = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2$.

- $x_0 = 1$: $\ \mathrm{Var} = 1\left[\frac13 + \frac{1}{2}\right] = 0.8333$
- $x_0 = 2$: $\ \mathrm{Var} = 1\left[\frac13 + 0\right] = 0.3333$
- $x_0 = 3$: $\ \mathrm{Var} = 0.8333$

| $x$ | $\mathrm{Bias}^2$ | Var | $\sigma^2$ | Total |
|---|---:|---:|---:|---:|
| 1 | 0 | 0.833 | 1 | 1.833 |
| 2 | 0 | 0.333 | 1 | 1.333 |
| 3 | 0 | 0.833 | 1 | 1.833 |
| **avg** | **0** | **0.667** | **1** | **1.667** |

The average variance is $0.667 = \dfrac{p\,\sigma^2}{n}$ with $p = 2$ parameters and $n=3$ points — a useful rule of thumb: **for a linear least-squares fit, the average prediction variance over the design points is $p\sigma^2/n$.** Variance grows *linearly with the number of parameters*.

### M2 — the quadratic

Three parameters fitted to three points: the parabola passes exactly through every observation, so $\hat f(x_i) = y_i$. Consequences:

- Unbiased: $E[\hat f(x_i)] = E[y_i] = f(x_i)$ → $\mathrm{Bias} = 0$.
- $\mathrm{Var}(\hat f(x_i)) = \mathrm{Var}(y_i) = \sigma^2 = 1$ at every point (matching $p\sigma^2/n = 3\cdot1/3 = 1$).

| $x$ | $\mathrm{Bias}^2$ | Var | $\sigma^2$ | Total |
|---|---:|---:|---:|---:|
| all | 0 | 1.000 | 1 | 2.000 |

### The comparison

| Model | params $p$ | $\mathrm{Bias}^2$ (avg) | Var (avg) | $\sigma^2$ | **Total** |
|---|---:|---:|---:|---:|---:|
| M0 constant | 1 | 2.667 | 0.333 | 1 | 4.000 |
| **M1 line** | 2 | 0.000 | 0.667 | 1 | **1.667** ← best |
| M2 quadratic | 3 | 0.000 | 1.000 | 1 | 2.000 |

**Read this carefully — it contains the whole lecture:**

- M0 → M1: adding a parameter removed *all* the bias (2.667 → 0) at a variance cost of only 0.333. Worth it.
- M1 → M2: adding another parameter removed *no* bias (there was none left) but added 0.333 of variance. Pure loss. **This is overfitting, quantified.**
- Notice M2 has **zero training error** (it interpolates every point perfectly) and yet is *worse* than M1. Training error is not test error.

**Monte-Carlo verification.** Simulating 400,000 random training sets and computing $E[(y-\hat f)^2]$ by brute force at each design point gives:

| Model | MC total error at $x=1,2,3$ | predicted by decomposition |
|---|---|---|
| M0 | 5.33, 1.34, 5.33 | 5.33, 1.33, 5.33 |
| M1 | 1.84, 1.34, 1.84 | 1.83, 1.33, 1.83 |
| M2 | 2.00, 2.00, 2.00 | 2.00, 2.00, 2.00 |

Agreement to within Monte-Carlo noise. The identity is exact, not approximate. ✔

---

## 9. Numerical example 3 — the U-curve, measured

**Setup.** True function $f(x) = \sin(2\pi x)$ on $x \in [0,1]$; $n = 12$ evenly spaced training inputs; noise $\sigma = 0.30$ so $\sigma^2 = 0.09$; fit polynomials of degree $d = 0,1,\dots,9$; repeat over 4000 independent training sets; average $\mathrm{Bias}^2$ and $\mathrm{Var}$ over 49 test points.

| degree $d$ | params $p = d+1$ | $\mathrm{Bias}^2$ | Var | $\sigma^2$ | **Total** |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.5102 | 0.0076 | 0.09 | 0.6078 |
| 1 | 2 | 0.2087 | 0.0137 | 0.09 | 0.3124 |
| 2 | 3 | 0.2087 | 0.0193 | 0.09 | 0.3180 |
| **3** | 4 | 0.0062 | 0.0246 | 0.09 | **0.1208** ← minimum |
| 4 | 5 | 0.0062 | 0.0304 | 0.09 | 0.1265 |
| 5 | 6 | 0.0000 | 0.0371 | 0.09 | 0.1271 |
| 6 | 7 | 0.0000 | 0.0458 | 0.09 | 0.1358 |
| 7 | 8 | 0.0000 | 0.0574 | 0.09 | 0.1474 |
| 8 | 9 | 0.0000 | 0.0750 | 0.09 | 0.1650 |
| 9 | 10 | 0.0000 | 0.1194 | 0.09 | 0.2095 |

**Observations:**

1. **Bias² falls monotonically** with complexity and then sits at ~0. Once the model family can represent $\sin(2\pi x)$ well (degree 3 already captures it to 0.006), extra flexibility buys nothing.
2. **Variance rises monotonically**, roughly in proportion to the parameter count, exactly as $p\sigma^2/n$ predicts ($d=9$: $10\times0.09/12 = 0.075$; measured 0.119 — larger because the test points include regions between and outside the training points, where high-degree polynomials swing wildly).
3. **Total is U-shaped**, minimum at $d=3$. Both $d=0$ (underfit, bias-dominated) and $d=9$ (overfit, variance-dominated) are bad, for *opposite* reasons.
4. Bias² also plateaus in pairs ($d=1,2$ and $d=3,4$) because $\sin(2\pi x)$ is odd about $x=1/2$ — even-degree terms add no fitting power here. A nice reminder that "complexity" is about *useful* flexibility, not raw parameter count.
5. The total never drops below $0.09 = \sigma^2$. The floor is real.

Sketching column "Total" against $d$ reproduces the board's figure:

```
 error
   ^
   |  \                                   /   <- variance (rising)
   |   \                               /
   |    \            total          /
   |     \___                    /
   |         \___          ___/
   |             \___ ___/   <- minimum = optimal model complexity
   |  bias^2 -->      \____________________  (falls, then flat)
   |  ......................................  irreducible error sigma^2
   +---------------------------------------> model complexity
```

---

## 10. Underfitting vs overfitting — the diagnostic table

You almost never know $f$, so you diagnose from *training* and *validation* errors:

| Symptom | Training error | Validation error | Gap | Diagnosis | Dominant term |
|---|---|---|---|---|---|
| Model too rigid | high | high | small | **underfitting** | $\mathrm{Bias}^2$ |
| Model too flexible | very low | high | large | **overfitting** | $\mathrm{Var}$ |
| Just right | low | low | small | good fit | balanced |
| Both errors stuck near a floor | $\approx\sigma$ | $\approx\sigma$ | small | you are at the noise floor | $\sigma^2$ |

---

## 11. How to move each term (final board slide)

> **To decrease bias:** increase model complexity.
> **To decrease variance:** employ dimensionality reduction, regularization, or feature selection.

Expanded, with the mechanism in each case:

**Reducing bias** — give the model the capacity to represent the truth:
- higher polynomial degree / more basis functions / deeper or wider network
- add genuinely informative features (better descriptors)
- remove or weaken a constraint that was forcing the wrong shape
- boosting (each stage fits the residual left by the previous one)

**Reducing variance** — give the model less room to chase noise:
- **more training data** — the one move that reduces variance *without* raising bias (variance typically $\propto p/n$, so $n\uparrow$ shrinks it directly)
- **regularization** (ridge / L2, lasso / L1, weight decay, early stopping) — shrinks parameters toward zero; this *introduces* a little bias on purpose and is the clearest demonstration that the trade-off can be exploited, not just suffered
- **dimensionality reduction** (e.g. PCA) and **feature selection** — fewer inputs, so fewer effective parameters
- **averaging / bagging / ensembles** — averaging $k$ roughly independent fits divides the variance by about $k$ while leaving bias unchanged
- cross-validation to *choose* the complexity knob rather than guessing it

**Cannot be reduced by any model:** $\sigma^2$. Only better measurements — a cleaner experiment, a more accurate reference calculation, averaging repeated measurements of the same point — lower the noise floor.

---

## 12. Formula sheet

$$
y = f + \epsilon, \qquad \epsilon \sim \mathcal N(0, \sigma^2), \qquad E[\epsilon] = 0, \qquad E[\epsilon^2] = \sigma^2
$$

$$
\mathrm{Bias}(\hat f) = E[\hat f] - f
\qquad
\mathrm{Var}(\hat f) = E[\hat f^{\,2}] - \big(E[\hat f]\big)^2
$$

$$
E\big[(y-\hat f)^2\big] \;=\; \underbrace{\big(E[\hat f] - f\big)^2}_{\text{Bias}^2\ \text{(wrong on average)}} \;+\; \underbrace{E\big[(\hat f - E[\hat f])^2\big]}_{\text{Variance (unstable)}} \;+\; \underbrace{\sigma^2}_{\text{irreducible}}
$$

Supporting identities used in the proof:

| Identity | Reason |
|---|---|
| $E[y^2] = f^2 + \sigma^2$ | $f$ constant for a given data point; $E[\epsilon]=0$, $E[\epsilon^2]=\sigma^2$ |
| $E[\hat f^{\,2}] = \mathrm{Var}(\hat f) + (E[\hat f])^2$ | definition of variance, expanded |
| $E[2y\hat f] = 2f\,E[\hat f]$ | $\epsilon$ and $\hat f$ independent, $E[\epsilon] = 0$ |

---

## 13. Self-check questions

1. A model's predictions at a fixed $x$ over four training sets are $5, 7, 9, 11$; the truth is $f = 10$ and $\sigma^2 = 2$. Compute $\mathrm{Bias}^2$, $\mathrm{Var}$, and the total expected squared error.
2. Your test MSE is $0.05$ but you know the measurement noise has $\sigma^2 = 0.20$. What has gone wrong?
3. You double the training-set size and keep the model fixed. Which of the three terms change, and in which direction?
4. Why does the cross term $E[2y\hat f]$ not contribute a fourth piece to the decomposition?
5. True or false: a model with zero training error has zero bias.

<details>
<summary><b>Answers</b></summary>

1. $E[\hat f] = \frac{5+7+9+11}{4} = 8$. $\mathrm{Bias} = 8-10 = -2 \Rightarrow \mathrm{Bias}^2 = 4$. $\mathrm{Var} = \frac{9+1+1+9}{4} = 5$. Total $= 4+5+2 = \mathbf{11}$.
2. You cannot beat the noise floor — the expected error is at least $\sigma^2 = 0.20$. Something is leaking: the test points were seen in training, the test set is not independent, or $\sigma^2$ was overestimated.
3. $\mathrm{Bias}^2$: essentially unchanged (it is a property of the model *family*, not of $n$). $\mathrm{Var}$: falls, roughly as $p/n$, so about halved. $\sigma^2$: unchanged — it is a property of the measurement.
4. Because it collapses to $2f\,E[\hat f]$, which combines with $f^2$ and $(E[\hat f])^2$ into the perfect square $(f - E[\hat f])^2$. Its disappearance relies on the noise at the query point being independent of the fitted model and having zero mean.
5. **False.** M2 in §8 interpolates all training points (zero training error) and happens to be unbiased — but only because the truth was inside its family. Zero *training* error says nothing about bias: fit a degree-2 polynomial through 3 points generated by $f(x)=\sin x$ and you get zero training error with plenty of bias away from those 3 points. What zero training error does reliably signal is **high variance**.

</details>

---

## 14. One-paragraph summary

A model's expected squared error at a point splits, exactly and without remainder, into three non-negative pieces: **bias²**, how far the model is from the truth *on average over training sets* (caused by presupposing too simple an architecture, and showing up as underfitting); **variance**, how much the model's prediction jumps around as the training set changes (caused by an architecture so flexible it fits noise as if it were signal, and showing up as overfitting); and the **irreducible error** $\sigma^2$, the measurement noise, which no model can remove. Increasing model complexity lowers bias and raises variance, so the total error is U-shaped in complexity; the job of model selection is to find the bottom of that U — the optimally complex model — usually by cross-validation, with regularization, feature selection, dimensionality reduction or more data used to pull the variance side down.
