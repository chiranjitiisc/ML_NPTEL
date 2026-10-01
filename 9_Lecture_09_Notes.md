# Lecture 9 — Multiple Linear Regression: Least Squares, Pseudoinverse and $R^2$

> Source: handwritten lecture slides (NPTEL, IISc), folder `ML_LECNOTES/9`

---

## 1. Setting: multiple linear regression

We have a model that depends on **several input variables (features)** and predicts a **continuous target variable**.

**Objective:** find the optimal parameters of such a model.

$$\hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$

| Symbol | Meaning |
|---|---|
| $\hat{y}_i$ | predicted value of the target variable for the $i^{\text{th}}$ data point |
| $\beta_0$ | **intercept** (bias term) |
| $\beta_j$ | weight of the $j^{\text{th}}$ feature in the model |
| $x_{ij}$ | value of the $j^{\text{th}}$ feature in the $i^{\text{th}}$ data point |
| $p$ | total number of features in the model |
| $n$ | number of data points |

So the model has $p+1$ parameters: $\beta_0, \beta_1, \dots, \beta_p$.

---

## 2. Matrix form

Write out the prediction for all $n$ data points at once:

$$
\begin{pmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \hat{y}_3 \\ \vdots \\ \hat{y}_n \end{pmatrix}
=
\begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
1 & \ddots &        &        & \vdots \\
\vdots &   & \ddots &        & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
\begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_p \end{pmatrix}
$$

**In matrix notation:**

$$\boxed{\hat{Y} = X\beta}$$

Key points about the shapes:

- $\hat{Y}$ is $n \times 1$
- $X$ is $n \times (p+1)$ — the **design matrix**
- $\beta$ is $(p+1) \times 1$

The **column of 1's** in the first column of $X$ is the trick that lets the intercept $\beta_0$ be absorbed into the same matrix product (it multiplies $\beta_0$ by 1 for every data point).

---

## 3. The loss function: MSE / residual sum of squares

**Mean-squared error (MSE)**, also called the residual sum of squares:

$$\text{MSE}(\beta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2$$

- $y_i$ — **true** value of the target variable
- $\hat{y}_i$ — **predicted** value of the target

Substituting the model:

$$\boxed{\text{MSE}(\beta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \left\{\beta_0 + \sum_{j=1}^{p}\beta_j x_{ij}\right\}\right)^2}$$

### Optimality conditions

To find the optimal set of parameters, set the derivative w.r.t. each parameter to zero (condition for **any extremum**):

$$\frac{\partial\, \text{MSE}(\beta)}{\partial \beta_j} = 0$$

To ensure the extremum is a **minimum**:

$$\frac{\partial^2 \text{MSE}}{\partial \beta_j^2} > 0$$

---

## 4. Minimising in matrix form

Work with the **sum of squared errors** instead of the mean (the factor $n$ does not change where the minimum is):

$$\text{SSE} = n \times \text{MSE}$$

$$
\begin{aligned}
\text{SSE}(\beta) &= (Y - X\beta)^T (Y - X\beta) \\
&= \left(Y^T - (X\beta)^T\right)(Y - X\beta) \\
&= \left(Y^T - \beta^T X^T\right)(Y - X\beta)
\end{aligned}
$$

Expanding:

$$\boxed{\text{SSE}(\beta) = Y^T Y - Y^T X\beta - \beta^T X^T Y + \beta^T X^T X \beta}$$

### First derivative

Setting $\dfrac{\partial(\text{SSE})}{\partial \beta} = 0$, with

$$\boxed{\frac{\partial}{\partial \beta}(\text{SSE}) = -2X^T Y + 2X^T X \beta}$$

(The two cross terms $-Y^TX\beta$ and $-\beta^TX^TY$ are scalars and equal to each other, which is why they combine into a single $-2X^TY$.)

### Second derivative

$$\frac{\partial^2 (\text{SSE})}{\partial \beta^2} = 2\,X^T X$$

$X^T X$ is a **positive semi-definite matrix** → this ensures a minimum in most cases.

> Why positive semi-definite: for any vector $v$, $v^T X^T X v = \|Xv\|^2 \ge 0$. It is strictly positive definite (a *strict* minimum, unique solution) only when $X$ has full column rank — i.e. no feature is an exact linear combination of others.

---

## 5. The normal equation — best-fit linear model

For the optimal $\beta$, denoted $\hat{\beta}$:

$$
\begin{aligned}
-2X^T Y + 2X^T X \hat{\beta} &= 0 \\
2X^T X\hat{\beta} &= 2X^T Y
\end{aligned}
$$

$$\boxed{\hat{\beta} = \left(X^T X\right)^{-1} X^T Y}$$

This is the expression for the optimal set of parameters — the **"best-fit linear model"**. It is a *closed-form* (analytic) solution: no iterative optimisation needed.

### Complex-valued data

If $X$ or $Y$ have **complex entries**:

$$\hat{\beta} = \left(X^H X\right)^{-1} X^H Y$$

where

$$X^H = \overline{X^T}$$

is the **Hermitian transpose** = complex-conjugate transpose.

---

## 6. The pseudoinverse (Moore–Penrose inverse)

The matrix

$$\boxed{X^{+} = \left(X^H X\right)^{-1} X^H}$$

is referred to as the **pseudoinverse** or **Moore–Penrose inverse** of the matrix $X$. It is the **generalisation of the concept of an inverse to a rectangular matrix**.

So the least-squares solution is simply $\hat{\beta} = X^{+} Y$.

### Connection to linear systems

For a linear system

$$A x = b \quad \longrightarrow \quad x = A^{+} b$$

$$x = \left(A^H A\right)^{-1} A^H b$$

i.e. solving an over-determined system in the least-squares sense is exactly the same algebra as fitting a linear regression.

---

## 7. Goodness of fit: coefficient of determination $R^2$

$R^2$ is a measure of the **goodness of fit** of a model.

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$

- $SS_{\text{res}}$ — residual sum of squares
- $SS_{\text{tot}}$ — total sum of squares

$$SS_{\text{res}} = \sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2 \equiv \text{SSE}$$

$$SS_{\text{tot}} = \sum_{i=1}^{n}\left(y_i - \bar{y}\right)^2$$

where $y_i$ is the true value, $\hat{y}_i$ the predicted value, and $\bar{y}$ the **mean of the true values**.

$$\boxed{R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

*Note: on the slide $SS_{\text{res}}$ was written once without the square — the squared form above (matching SSE and the boxed final formula) is the correct one.*

### Interpretation via limiting cases

| Case | Condition | $R^2$ |
|---|---|---|
| **Perfect model** | $\hat{y}_i = y_i$ for all $i$ → $SS_{\text{res}} = 0$ | $R^2 = 1 - 0 = \boxed{1}$ |
| **Mean predictor** | $\hat{y}_i = \bar{y}$ always → $SS_{\text{res}} = SS_{\text{tot}}$ | $R^2 = 1 - 1 = \boxed{0}$ |
| **Imperfect model** | general case | $R^2 < 1$ |

So $R^2$ compares your model against the dumbest possible baseline — **always predicting the mean value**. $R^2 = 0$ means the model is no better than that baseline. (A model *worse* than the mean predictor gives $R^2 < 0$.)

---

## 8. Parity plot

A **parity plot** puts true values on the $x$-axis and predicted values on the $y$-axis:

```
        ^
   ŷ    |                    ,'  <- perfect model (R² = 1): ŷ = y
predicted|                 ,'         points lie on the 45° diagonal
 values  |              ,'
         |           ,'
         | · · · · · · · · · · ·  <- R² = 0: ŷ = ȳ
         |        ,'                  a flat horizontal line
         |     ,'
         |  ,'
         +------------------------->
                y ≡ true values
```

- **Perfect model ($R^2 = 1$, $\hat{y} = y$):** all points fall on the diagonal $y = x$.
- **$R^2 = 0$ model ($\hat{y} = \bar{y}$):** all points fall on a **horizontal line** at the mean, regardless of the true value.
- Real models scatter around the diagonal; the tighter the scatter about the 45° line, the better the fit.

---

## Summary — the whole lecture in one chain

1. Model: $\hat{Y} = X\beta$, with a column of 1's in $X$ for the intercept.
2. Loss: $\text{MSE}(\beta) = \frac{1}{n}\|Y - X\beta\|^2$.
3. Set $\partial \text{SSE}/\partial\beta = -2X^TY + 2X^TX\beta = 0$.
4. Solve: $\hat{\beta} = (X^TX)^{-1}X^TY = X^{+}Y$ (pseudoinverse; use $X^H$ for complex data).
5. Second derivative $2X^TX \succeq 0$ guarantees it's a minimum.
6. Assess with $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$ and visualise with a parity plot.
