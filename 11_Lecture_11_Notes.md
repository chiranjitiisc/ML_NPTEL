# Lecture 11 — Overfitting, Underfitting & Ridge Regression

> Source: handwritten lecture slides (NPTEL, IISc), folder `ML_LECNOTES/11`
>
> Lecture 9 gave us $\hat{\beta} = (X^TX)^{-1}X^TY$. Lecture 10 asked *how much we trust it*. This lecture asks a different question: **is the model we fitted the right model at all?** — and then gives the first repair, **ridge regression**.

---

## 1. Overfitting — definition

**Overfitting** is a scenario when the **predictive power** of an ML model, as quantified by its performance on an **unseen test set**, is **compromised** due to overtraining of the model, which could still lead to an excellent performance on the training set.

In terms of the loss function:

| Set | Loss under overfitting |
|---|---|
| Training set | can take **very low** values |
| Test set | would have a **high** value |

> **Intuition.** The model has memorised the training points, noise and all, instead of learning the pattern that generated them. The noise in the training set is *not* present in the test set — it is different noise — so every wiggle the model added to chase a training point actively hurts it on new data. A perfect training score is therefore not good news on its own; it is only meaningful next to the test score.

---

## 2. Why overfitting occurs

This occurs when the model selected is **too complex** and places **too much importance on certain features**.

For example:

- the model has an **excessive number of parameters**
- the model **architecture is too complex**

> **The counting argument.** A polynomial of degree $n-1$ can pass exactly through any $n$ points. With enough free parameters, zero training error is always achievable — and it is worthless, because the fit is then determined by the data's noise rather than its structure. Complexity has to be paid for out of the information content of the data.
>
> "Places too much importance on certain features" is the concrete symptom, and it is the one ridge regression is built to attack: a few coefficients $\beta_j$ blow up to large magnitudes, often in cancelling pairs, so the fitted surface is violently sensitive to small changes in those features.

---

## 3. Underfitting — definition

**Underfitting** occurs when the chosen ML model is unable to capture the variations in data, such that it performs poorly on **even the training set**.

- **High** values of the loss function on **both** the training and test sets.
- Likely leading to **low $R^2$ values**.

> **How to tell the two apart in practice.** Look at the two losses together, not either alone:
>
> | Training loss | Test loss | Diagnosis |
> |---|---|---|
> | low | high | **overfitting** — model too complex |
> | high | high | **underfitting** — model too simple |
> | low | low (comparable) | **just right** |
> | high | low | essentially impossible — suspect a bug or a leak |
>
> Underfitting is the easier failure to spot, because the model cannot even hide it on the data it was trained on.

---

## 4. The three regimes

$$
\underbrace{\left(\text{Model too complex}\right)}_{\textbf{OVERFITTING}}
\qquad
\underbrace{\left(\text{Model just right}\right)}_{\text{target}}
\qquad
\underbrace{\left(\text{Model too simple}\right)}_{\textbf{UNDERFITTING}}
$$

**Model just right:** *comparable performance of the ML model on the training and the test sets, as quantified by the chosen loss function.*

> Note the criterion for "just right" is **comparability**, not smallness. Two models with training loss 0.05 / test loss 0.06 and 0.30 / test loss 0.31 are both well-specified in this sense; the second is simply working on harder or noisier data. What you never want is a *gap*.

---

## 5. The picture

```
   y
   ^
   |          ___                    .-'  <-- overfitting (wiggly, hits every point)
   |        /'   \        ,-.      ,'
   |    o  /  o   \      /   \   o/
   |      /        \    /     \  /
   |     /   _.-''--\--/--''-._\/        <-- underfitting (straight line, misses structure)
   |    o_.-'        \/          o
   |  .-'  o                  o
   | /                                   <-- optimally selected model
   |o                                        (smooth curve through the trend)
   +---------------------------------> x
```

Three curves are drawn through the same scatter of points:

| Curve | What it does | Verdict |
|---|---|---|
| Straight red line | too rigid to follow the trend | **underfitting** |
| Smooth dashed curve | follows the trend, ignores the scatter | **optimally selected model** |
| Wildly oscillating curve | passes through (nearly) every training point | **overfitting** |

> The overfitting curve has the *lowest* training error of the three — it is nearly interpolating — and would be the *worst* of the three at predicting a new point drawn between two training points, because between the points it swings far away from the trend. That gap between "fits the data I have" and "predicts the data I don't" is the entire subject of this lecture.

---

## 6. Dealing with the overfitting problem in linear regression

Three standard approaches, all **regularization** methods:

$$\text{Regularized linear regression} \begin{cases} \textbf{Ridge regression} \quad (\text{this lecture}) \\ \textbf{LASSO} \\ \textbf{Elastic net}\end{cases}$$

> All three work the same way: add a term to the loss that **penalises large coefficients**, so the fit has to *earn* every unit of coefficient magnitude by a corresponding reduction in error. They differ only in how the penalty is measured — ridge uses $\sum\beta_j^2$ (an $\ell_2$ penalty), LASSO uses $\sum|\beta_j|$ (an $\ell_1$ penalty), and elastic net mixes the two.

---

## 7. Ridge regression — the penalised loss function

To avoid the problem of the model overfitting by assigning **inordinately high values to the weights for some features**, we introduce a **penalty for very high weighting coefficients**.

$$L \;=\; \underbrace{\sum_{i=1}^{n}\left(y_i - \left(\beta_0 + \sum_{j=1}^{p} x_{ij}\beta_j\right)\right)^{2}}_{\text{usual loss function (SSE) for linear regression}} \;+\; \underbrace{\lambda\sum_{j=1}^{p}\beta_j^{2}}_{\substack{\text{penalty term to stop some weights}\\\text{from becoming too high}}}$$

where $\lambda$ is the **regularization parameter**.

$$\boxed{\hat{\beta}_{\text{ridge}} \;=\; \underset{\beta}{\arg\min}\; \left(L(\beta)\right)}$$

### Reading the two terms against each other

| | pulls towards | wins when |
|---|---|---|
| SSE term | whatever fits the training data best | $\lambda$ small |
| $\lambda\sum\beta_j^2$ | all coefficients towards **zero** | $\lambda$ large |

> $\lambda$ is a **knob you choose**, not something the data hands you — it is a *hyperparameter*, tuned by cross-validation, not fitted by minimising $L$. (Minimising $L$ over $\lambda$ as well would just drive $\lambda \to 0$, since the penalty only ever adds to the loss.)
>
> Two details worth noticing in the formula:
> - The sum in the penalty runs $j = 1 \dots p$, **not** from $j=0$: the intercept $\beta_0$ is deliberately left unpenalised. Shrinking $\beta_0$ would tie the prediction to the arbitrary origin of the $y$ scale.
> - Because the penalty is measured in the units of $\beta_j^2$, and $\beta_j$ carries the inverse units of feature $j$, ridge is **not scale-invariant**. Features are standardised before fitting, otherwise a feature measured in metres and the same feature measured in kilometres would be penalised a thousandfold differently.

---

## 8. Alternative implementation — in terms of a constraint

The same estimator can be written as a **constrained** optimisation instead of a penalised one:

$$\hat{\beta}_{\text{ridge}} \;=\; \underset{\beta}{\arg\min}\left\{\sum_{i=1}^{n}\left(y_i - \left(\beta_0 + \sum_{j=1}^{p}\beta_j x_{ij}\right)\right)^{2}\right\}$$

$$\text{subject to} \qquad \boxed{\sum_{j=1}^{p}\beta_j^{2} \;\le\; t}$$

> **Why these are the same problem.** This is Lagrange duality: minimising $\text{SSE} + \lambda\sum\beta_j^2$ is equivalent to minimising SSE inside a ball $\sum\beta_j^2 \le t$, with $\lambda$ playing the role of the Lagrange multiplier attached to the constraint. There is a one-to-one, *decreasing* correspondence between the two knobs:
>
> | | no regularization | heavy regularization |
> |---|---|---|
> | penalty form | $\lambda = 0$ | $\lambda \to \infty$ |
> | constraint form | $t \to \infty$ (ball is all of space) | $t \to 0$ (ball shrinks to the origin) |
>
> The constraint picture is the more geometric one: the unconstrained least-squares solution $\hat{\beta}_{\text{OLS}}$ sits at the centre of a set of elliptical SSE contours, and the ridge solution is the point where the smallest of those ellipses first touches the sphere of radius $\sqrt{t}$. Because the constraint region is a *sphere* (smooth, no corners), the touching point generically has **all** coefficients non-zero but shrunken. Swapping the sphere for the diamond $\sum|\beta_j| \le t$ gives LASSO, whose corners lie on the axes — which is exactly why LASSO sets coefficients to *exactly* zero and ridge does not.

---

## 9. Matrix form and the closed-form solution

Collect the parameters into a vector:

$$\beta = \begin{bmatrix}\beta_1 & \beta_2 & \cdots & \beta_p\end{bmatrix}$$

Then

$$\hat{\beta}_{\text{ridge}} = \underset{\beta}{\arg\min}\Big[\underbrace{\left(Y - X\beta\right)^{T}\left(Y-X\beta\right) \;+\; \lambda\,\beta^{T}\beta}_{L(\beta)}\Big]$$

> $(Y-X\beta)^T(Y-X\beta)$ is just $\sum_i (y_i - \hat{y}_i)^2$ written as a dot product of the residual vector with itself, and $\beta^T\beta = \sum_j \beta_j^2$ is the squared length of the coefficient vector. Nothing new — only compressed.

### Setting the derivative to zero

$$\frac{\partial L(\beta)}{\partial \beta} = 0 \qquad \Longrightarrow \qquad -2X^{T}\left(Y - X\beta\right) + 2\lambda\beta = 0$$

$$\Longrightarrow \qquad \left(X^{T}X + \lambda I\right)\beta = X^{T}Y$$

$$\boxed{\hat{\beta}_{\text{ridge}} = \left(X^{T}X + \lambda I\right)^{-1}X^{T}Y}$$

> **Working the derivative in full.** Expand
> $$L(\beta) = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta + \lambda\beta^T\beta$$
> and use the two standard vector-calculus results $\dfrac{\partial}{\partial\beta}\left(\beta^Ta\right) = a$ and $\dfrac{\partial}{\partial\beta}\left(\beta^TA\beta\right) = 2A\beta$ for symmetric $A$:
> $$\frac{\partial L}{\partial\beta} = -2X^TY + 2X^TX\beta + 2\lambda\beta = -2X^T(Y - X\beta) + 2\lambda\beta$$
> Divide by 2 and gather the $\beta$ terms: $X^TX\beta + \lambda\beta = X^TY$, and since $\lambda\beta = \lambda I\beta$, the two terms combine into $\left(X^TX + \lambda I\right)\beta$. The identity matrix $I$ is doing nothing more than making the dimensions match so the $\lambda$ can be absorbed into the bracket — but that is precisely where the whole benefit comes from, as §11 shows.
>
> Because $L(\beta)$ is a sum of two convex quadratics, this stationary point is the **unique global minimum** — no local-minimum worries here.

---

## 10. The $\lambda = 0$ limit — sanity check

$$\hat{\beta}_{\text{ridge}} = \left(X^{T}X + \underbrace{\lambda I}_{\substack{\text{if } \lambda = 0}}\right)^{-1}X^{T}Y$$

If $\lambda = 0$, this expression reduces to

$$\hat{\beta} = \left(X^{T}X\right)^{-1}X^{T}Y$$

which is exactly the **ordinary least-squares result from Lecture 9**.

> Always worth doing: ridge regression is not a different model, it is the *same* model with a dial on it, and turning the dial to zero must give back what we had. At the other extreme, $\lambda \to \infty$ makes the $\lambda I$ term dominate and $\hat{\beta}_{\text{ridge}} \to 0$ — the model collapses to predicting the mean, i.e. maximal underfitting. So sweeping $\lambda$ from $0$ to $\infty$ traverses the whole axis of §4, from overfitting to underfitting, and the job of cross-validation is to stop at "just right".

---

## 11. Advantages of ridge regression

1. It can **prevent over-reliance on some parameters**, to improve the **generalizability** of the model.

2. $\lambda$ can be **tuned** to achieve comparable / best performance on **both train and test sets**.

3. If $X^TX$ is **singular** — i.e. $\left(X^TX\right)^{-1}$ does not exist — regularization can still enable the calculation of $\left(X^TX + \lambda I\right)^{-1}$.

> **On the third advantage — this is the one to hold on to.** Lecture 10 §8 gave $\text{Var}(\hat\beta) = \sigma^2\,\text{diag}\left((X^TX)^{-1}\right)$ and warned that nearly-collinear features make $X^TX$ close to singular, its inverse blow up, and the parameter estimates become wildly uncertain. Ridge fixes exactly that failure, and the mechanism is visible in the eigenvalues.
>
> $X^TX$ is symmetric positive semi-definite, so it has real eigenvalues $d_1, d_2, \dots, d_p \ge 0$. Adding $\lambda I$ shifts **every** eigenvalue up by the same amount:
> $$d_k \;\longrightarrow\; d_k + \lambda$$
> A singular $X^TX$ has some $d_k = 0$, which is why it cannot be inverted; after the shift that eigenvalue is $\lambda > 0$ and the matrix is invertible. A *nearly* singular $X^TX$ has some tiny $d_k$, and $1/d_k$ is enormous — that is the variance blow-up; after the shift the worst factor is $1/(d_k+\lambda)$, which is bounded by $1/\lambda$.
>
> This also means ridge works when $p > n$ (more features than data points), where $X^TX$ is *always* singular and ordinary least squares has no unique solution at all.
>
> **The price.** Ridge is a **biased** estimator: $E[\hat\beta_{\text{ridge}}] \ne \beta$, unlike the OLS estimator we proved unbiased in Lecture 10 §7. What it buys with that bias is a large reduction in variance, and since prediction error is roughly $\text{bias}^2 + \text{variance} + \text{noise}$, a small deliberate bias can lower the total. That trade is the **bias–variance trade-off**, and it is the formal statement of the overfitting–underfitting axis: a complex model is low-bias / high-variance (overfits), a simple model is high-bias / low-variance (underfits), and $\lambda$ tunes between them.

---

## Summary — the whole lecture in one chain

1. **Overfitting**: low training loss, high test loss. The model memorised the noise; it happens when the model is too complex (too many parameters, architecture too rich).
2. **Underfitting**: high loss on *both* sets, low $R^2$. The model is too simple to capture the variation.
3. "Just right" is defined by **comparable** train and test loss, not by low loss alone.
4. Three regularization cures for overfitting in linear regression: **ridge**, **LASSO**, **elastic net**.
5. Ridge adds a penalty on coefficient size: $L = \text{SSE} + \lambda\sum_{j=1}^{p}\beta_j^2$, with $\lambda$ the **regularization parameter**.
6. Equivalently, minimise SSE **subject to** $\sum_j\beta_j^2 \le t$ — the same problem, seen through Lagrange duality.
7. In matrix form, $L(\beta) = (Y-X\beta)^T(Y-X\beta) + \lambda\beta^T\beta$.
8. Setting $\partial L/\partial\beta = 0$ gives $\left(X^TX + \lambda I\right)\beta = X^TY$, hence $\boxed{\hat\beta_{\text{ridge}} = \left(X^TX+\lambda I\right)^{-1}X^TY}$.
9. $\lambda = 0$ recovers OLS; $\lambda \to \infty$ shrinks all coefficients to zero.
10. Benefits: better generalization, a tunable knob, and — crucially — invertibility even when $X^TX$ is singular or ill-conditioned, at the cost of a little bias.
