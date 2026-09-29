# Lecture 14 — Logistic Regression and How to Judge a Binary Classifier

**Source:** handwritten lecture board (NPTEL, IISc, "Lec14 – Windows Journal", pages 1–9) — screenshots in `lec_14/` (pages 1–6) and six more Lec-14 screenshots dated 29-09-2026 that are sitting in `lec_13/` (confusion matrix, accuracy, balanced accuracy).

**Goal of the lecture:**

1. Build a model that answers a **yes/no question** (e.g. "is this material a metal?") instead of predicting a number.
2. Train it with a loss suited to probabilities — the **binary cross-entropy**.
3. Measure how good it is — **confusion matrix, accuracy, sensitivity, specificity, balanced accuracy**.

Every symbol is defined before it is used, and each idea is followed by arithmetic you can check with a pen and a calculator.

---

## 0. Why we need a new model

In linear regression (earlier lectures) the target $y$ is a **continuous number** — a solubility, an energy, a band gap — and the model is a straight line (or plane):

$$
\hat y = \beta_0 + \beta_1 x
$$

where $x$ is the input feature, $\beta_0$ is the intercept (value of $\hat y$ at $x = 0$), $\beta_1$ is the slope, and $\hat y$ ("y-hat") is the model's prediction.

Now suppose the target can only take **two values**. Examples:

| Question | $y = 0$ | $y = 1$ |
|---|---|---|
| Is the material a metal? | semiconductor | metal |
| Does the reaction proceed? | no | yes |
| Is this atom in the crystal phase? | liquid-like | crystal-like |

A variable that can only be 0 or 1 is called a **binary variable**, and sorting inputs into one of two groups is **binary classification**. The lecture names the groups:

- **Class I:** $y = 0$
- **Class II:** $y = 1$

**What goes wrong if we just use a straight line?** A line $\beta_0 + \beta_1 x$ runs from $-\infty$ to $+\infty$. Take $\beta_0 = -1$, $\beta_1 = 0.5$:

| $x$ | 0 | 2 | 4 | 6 |
|---|---|---|---|---|
| $\beta_0 + \beta_1 x$ | −1.0 | 0.0 | 1.0 | 2.0 |

A "prediction" of −1 or 2 is meaningless when the answer must be 0 or 1. What we really want is a **probability** — a number between 0 and 1 that says how confident the model is that $y = 1$. Logistic regression does exactly this: it keeps the linear part and squeezes its output into $(0, 1)$. That is why the board calls it **"a modification of linear regression used to predict the value of a binary variable."**

---

## 1. Vocabulary from scratch

- **Feature** $x$ — the input you measure (temperature, a bond length, an order parameter). With several features you stack them into a **feature vector** $\mathbf{x} = [x_1, x_2, \dots, x_p]$, where $p$ is the number of features.
- **Target** $y$ — the true class of a data point, either 0 or 1.
- **Data set** — $n$ pairs $(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)$. The subscript $i$ labels the data point.
- **Probability** $P(\text{event})$ — a number between 0 (never happens) and 1 (always happens).
- **Conditional probability** $P(y = 1 \mid \mathbf{x})$ — read "the probability that $y$ equals 1 **given** that the features are $\mathbf{x}$". The vertical bar means "given". (The board writes it with a slash, $P(y=1/x)$ — same thing.) This boxed quantity is **what logistic regression models.**
- Since $y$ is either 0 or 1, the two probabilities must add to 1:

$$
P(y = 0 \mid \mathbf{x}) = 1 - P(y = 1 \mid \mathbf{x})
$$

- $\exp(z)$ — the exponential function $e^{z}$, with $e \approx 2.71828$. Key values: $e^{0} = 1$; $e^{z} \to \infty$ as $z \to \infty$; $e^{z} \to 0$ as $z \to -\infty$. It is **always positive**.
- $\log(z)$ — throughout this lecture, the **natural logarithm** (base $e$), the inverse of $\exp$: $\log(e^{z}) = z$. Key values: $\log(1) = 0$; $\log(z) < 0$ for $0 < z < 1$; $\log(z) \to -\infty$ as $z \to 0^{+}$.

---

## 2. The logistic function (one feature)

### 2.1 The formula

The lecture's model relies on the **logistic function**:

$$
p(x) = \frac{1}{1 + \exp\!\left(-\dfrac{x - \alpha}{\gamma}\right)}
$$

Symbols:

- $x$ — the single input feature.
- $\alpha$ (alpha) — a **location** parameter: the value of $x$ where the curve crosses its midpoint.
- $\gamma$ (gamma) — a **width** parameter, $\gamma > 0$: how gradually the curve rises.
- $p(x)$ — the output, interpreted as $P(y = 1 \mid x)$.

The lecture also writes the curve as $\sigma(\cdot)$ (sigma). In general, for any number $z$,

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \text{so } p(x) = \sigma\!\left(\frac{x - \alpha}{\gamma}\right)
$$

The **logistic function is one member of the family of S-shaped curves called sigmoids** — hence the symbol $\sigma$.

### 2.2 Why the output is always between 0 and 1

Let $z = (x - \alpha)/\gamma$.

- $e^{-z}$ is always positive, so the denominator $1 + e^{-z}$ is always **greater than 1**, so $\sigma(z)$ is always **less than 1**.
- The denominator is also finite and positive, so $\sigma(z)$ is always **greater than 0**.
- $z \to +\infty$: $e^{-z} \to 0$, so $\sigma \to 1/(1+0) = 1$.
- $z \to -\infty$: $e^{-z} \to \infty$, so $\sigma \to 0$.
- $z = 0$: $e^{0} = 1$, so $\sigma = 1/(1+1) = 0.5$.

So the output is a legitimate probability, whatever $x$ is.

### 2.3 Reading the two parameters (board: page 3)

**$x = \alpha$ is where $p = 0.5$.** Put $x = \alpha$: then $z = 0$ and $p = 0.5$. So $\alpha$ is the "tipping point": below it the model leans towards class I, above it towards class II.

**$\gamma$ is the width of the logistic function.** A small $\gamma$ makes $z$ change fast with $x$ → a steep, almost step-like switch. A large $\gamma$ → a slow, gradual rise.

The graph on the board plots $p$ against $(x-\alpha)/\gamma$: it starts near 0 on the far left, passes through 0.5 at the origin, and flattens towards 1 on the right (dashed line).

### 2.4 Numerical example 1 — the curve by hand

Take $\alpha = 5$. Compute $p(x)$ for two widths.

Worked point, $\gamma = 1$, $x = 6$: $z = (6-5)/1 = 1$; $e^{-1} = 0.3679$; $p = 1/1.3679 = 0.7311$.

Worked point, $\gamma = 1$, $x = 3$: $z = -2$; $e^{2} = 7.389$; $p = 1/8.389 = 0.1192$.

| $x$ | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| $p(x)$, $\gamma = 1$ | 0.047 | 0.119 | 0.269 | **0.500** | 0.731 | 0.881 | 0.953 |
| $p(x)$, $\gamma = 2$ | 0.182 | 0.269 | 0.378 | **0.500** | 0.622 | 0.731 | 0.818 |

What to notice:

- Both rows equal 0.5 at $x = \alpha = 5$.
- Symmetry: $p(\alpha + d) = 1 - p(\alpha - d)$. E.g. $0.731 = 1 - 0.269$.
- With $\gamma = 1$ the probability goes from 0.05 to 0.95 over $x = 2 \to 8$; with $\gamma = 2$ it only goes from 0.18 to 0.82 — the wider curve.

### 2.5 Why this is still "linear regression" underneath

Define the **odds** of class II as $\dfrac{p}{1-p}$ (e.g. $p = 0.8$ gives odds $0.8/0.2 = 4$, "4 to 1"). Solve the logistic formula for the exponent:

$$
p = \frac{1}{1 + e^{-z}} \;\Rightarrow\; 1 + e^{-z} = \frac{1}{p} \;\Rightarrow\; e^{-z} = \frac{1-p}{p} \;\Rightarrow\; z = \log\frac{p}{1-p}
$$

So the **log of the odds (the "logit") is a straight-line function of $x$**:

$$
\log\frac{p}{1-p} = \frac{x - \alpha}{\gamma}
$$

Check with the table: at $x = 6$, $\gamma = 1$, $p = 0.7311$: $\log(0.7311/0.2689) = \log(2.718) = 1.0 = (6-5)/1$. ✓

Logistic regression = linear regression on the log-odds, followed by the sigmoid to turn it back into a probability.

---

## 3. From probability to a class label (board: pages 3–4)

The model outputs a probability, but eventually we must say "class I" or "class II". Use a cut-off:

$$
P(y=1 \mid \mathbf{x}) < 0.5 \;\Rightarrow\; \text{Class I}, \;\; \hat y = 0
$$

$$
P(y=1 \mid \mathbf{x}) \ge 0.5 \;\Rightarrow\; \text{Class II}, \;\; \hat y = 1
$$

Here $\hat y$ is the **predicted class** (the model's answer), as opposed to $y$, the **true class**.

The number 0.5 is called the **classification threshold probability**, written $p^{*}$. It is a choice, not a law — you can move it (see §7.5).

**Example (from §2.4, $\alpha = 5$, $\gamma = 1$, $p^* = 0.5$):** $x = 4 \Rightarrow p = 0.269 < 0.5 \Rightarrow \hat y = 0$. $x = 6 \Rightarrow p = 0.731 \ge 0.5 \Rightarrow \hat y = 1$. With one feature and $p^{*} = 0.5$ the rule is simply "$\hat y = 1$ if $x \ge \alpha$".

---

## 4. Many features: the $\beta$ form (board: pages 4–5)

### 4.1 The formula

With $p$ features $\mathbf{x} = [x_1, \dots, x_p]$, each with its own centre $\alpha_i$ and width $\gamma_i$, add up the scaled distances in the exponent:

$$
z = \sum_{i=1}^{p} \frac{x_i - \alpha_i}{\gamma_i}
$$

(The symbol $\sum_{i=1}^{p}$ means "add the following term for $i = 1, 2, \dots, p$".) Split the sum into the part that multiplies $x_i$ and the part that does not:

$$
z = \sum_{i=1}^{p} \frac{1}{\gamma_i}\,x_i \;-\; \sum_{i=1}^{p} \frac{\alpha_i}{\gamma_i}
$$

Now give the pieces names:

$$
\beta_0 = -\sum_{i=1}^{p} \frac{\alpha_i}{\gamma_i}, \qquad \boldsymbol\beta = \left[\frac{1}{\gamma_1}, \frac{1}{\gamma_2}, \dots, \frac{1}{\gamma_p}\right]
$$

- $\beta_0$ — the **intercept** (one number).
- $\boldsymbol\beta$ — the **coefficient vector** (one number per feature).
- $\boldsymbol\beta^{T}\mathbf{x}$ — the **dot product** $\beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$. (The superscript $T$, "transpose", just means "row times column, multiply matching entries and add".)

Then $z = \beta_0 + \boldsymbol\beta^{T}\mathbf{x}$, and the model on the board is

$$
P(y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp\!\big(-(\beta_0 + \boldsymbol\beta^{T}\mathbf{x})\big)}
$$

This is the standard form you will see in every textbook and in `scikit-learn`. In practice $\beta_0, \beta_1, \dots, \beta_p$ are simply the **unknown parameters that training finds**; the $\alpha, \gamma$ form is a way to interpret them ($\gamma_i = 1/\beta_i$ is the width along feature $i$). A negative $\beta_i$ just means the probability *falls* as $x_i$ grows.

### 4.2 Numerical example 2 — two features

Say we classify materials as metal ($y=1$) using $x_1$ = coordination number and $x_2$ = temperature in K, with $\alpha_1 = 5$, $\gamma_1 = 1$, $\alpha_2 = 300$, $\gamma_2 = 50$ (made-up numbers for arithmetic).

Parameters:

$$
\beta_0 = -\left(\frac{5}{1} + \frac{300}{50}\right) = -(5 + 6) = -11, \qquad \boldsymbol\beta = \left[\frac{1}{1}, \frac{1}{50}\right] = [1,\; 0.02]
$$

| $\mathbf{x} = (x_1, x_2)$ | $z = -11 + 1\cdot x_1 + 0.02\,x_2$ | $P(y=1\mid\mathbf{x}) = \sigma(z)$ | $\hat y$ |
|---|---|---|---|
| (6, 350) | $-11 + 6 + 7 = 2.0$ | 0.881 | 1 |
| (4, 250) | $-11 + 4 + 5 = -2.0$ | 0.119 | 0 |
| (5.5, 280) | $-11 + 5.5 + 5.6 = 0.1$ | 0.525 | 1 (barely) |

The third point shows why the probability is useful: $\hat y = 1$, but 0.525 warns that the model is nearly undecided.

---

## 5. The loss function: binary cross-entropy (board: pages 5–7)

### 5.1 What a loss function is

To train a model we need a single number that says **how wrong** the current parameters are on the training data. That number is the **loss** $L$. Training = finding the $\beta_0, \boldsymbol\beta$ that make $L$ as small as possible. (In linear regression the loss was the mean squared error.)

### 5.2 The formula

Let

$$
\hat p_i = P(y=1 \mid \mathbf{x} = \mathbf{x}_i)
$$

be the model's predicted probability of class II for data point $i$, and $y_i$ (0 or 1) be the **true class of that data point**. The **binary cross-entropy loss** (also called **log loss**, hence the subscript "log") is

$$
L_{\log} = -\frac{1}{n}\sum_{i=1}^{n}\Big[\, y_i \log(\hat p_i) + (1 - y_i)\log(1 - \hat p_i) \,\Big]
$$

The loss for **one** point is written $l_{\log, i}$:

$$
l_{\log,i} = -\Big[\, y_i \log(\hat p_i) + (1 - y_i)\log(1 - \hat p_i) \,\Big], \qquad L_{\log} = \frac{1}{n}\sum_{i=1}^{n} l_{\log,i}
$$

### 5.3 Reading it — only one term is ever "on"

Because $y_i$ is 0 or 1, exactly one of the two terms survives:

- If $y_i = 1$: the second term has factor $(1 - 1) = 0$, so $l_{\log,i} = -\log(\hat p_i)$. → penalise a **low** probability for a true class-II point.
- If $y_i = 0$: the first term has factor $0$, so $l_{\log,i} = -\log(1 - \hat p_i)$. → penalise a **high** probability for a true class-I point.

In words: $l_{\log,i} = -\log(\text{probability the model gave to the correct answer})$.

**Why the minus sign (circled in red on the board)?** Probabilities are ≤ 1, so their logs are ≤ 0. The minus sign flips the loss to be **≥ 0**, so "smaller loss = better model" and a perfect model has loss 0.

### 5.4 The four extreme cases (board: pages 7–8)

Convention: $0 \cdot \log(0)$ is taken as 0 (because $t\log t \to 0$ as $t \to 0$). The factor in front is zero, so the term is simply switched off.

| Case | $y_i$ | $\hat p_i$ | Calculation | $l_{\log,i}$ | Meaning |
|---|---|---|---|---|---|
| (a) | 1 | 0 | $-(1\cdot\log 0 + 0\cdot\log 1) = -(-\infty + 0)$ | $\to \infty$ | confidently wrong |
| (b) | 0 | 1 | $-(0\cdot\log 1 + 1\cdot\log 0) = -(0 + (-\infty))$ | $\to \infty$ | confidently wrong |
| (c) | 0 | 0 | $-(0\cdot\log 0 + 1\cdot\log 1) = -(0 + 0)$ | **0** | confidently right |
| (d) | 1 | 1 | $-(1\cdot\log 1 + 0\cdot\log 0) = -(0+0)$ | **0** | confidently right |

A confidently wrong prediction is punished **without limit**. That is the key design feature of the loss.

### 5.5 Numerical example 3 — how the penalty grows

For a true class-II point ($y_i = 1$), $l_{\log,i} = -\log(\hat p_i)$:

| $\hat p_i$ | 0.99 | 0.9 | 0.5 | 0.1 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|
| $l_{\log,i}$ | 0.010 | 0.105 | 0.693 | 2.303 | 4.605 | 6.908 |

Going from 0.1 to 0.01 (ten times more confident in the wrong answer) adds another 2.3 to the loss. Compare with squared error $(y - \hat p)^2$, which can never exceed 1: cross-entropy cares much more about confident mistakes. Note $0.693 = \log 2$ is the loss of a model that just says "50/50" for everything.

### 5.6 Numerical example 4 — loss on a small data set

| $i$ | $y_i$ | $\hat p_i$ | active term | $l_{\log,i}$ |
|---|---|---|---|---|
| 1 | 1 | 0.9 | $-\log(0.9)$ | 0.1054 |
| 2 | 1 | 0.6 | $-\log(0.6)$ | 0.5108 |
| 3 | 0 | 0.2 | $-\log(1 - 0.2) = -\log(0.8)$ | 0.2231 |
| 4 | 0 | 0.7 | $-\log(1 - 0.7) = -\log(0.3)$ | 1.2040 |

$$
L_{\log} = \frac{0.1054 + 0.5108 + 0.2231 + 1.2040}{4} = \frac{2.0433}{4} = 0.511
$$

Point 4 (true class I, but the model said 70 % class II) contributes more than the other three combined.

---

## 6. Training by numerical optimization (board: page 6)

### 6.1 Why "numerical"

In linear regression, setting the derivative of the squared-error loss to zero gives a closed-form answer (the normal equations). For logistic regression, the sigmoid inside the log makes the equations **non-linear in $\beta$ — no closed-form solution exists**. So "the model is trained by numerical optimization": start from a guess and improve it step by step.

### 6.2 The gradient (what the optimizer needs)

The **gradient** is the list of partial derivatives $\partial L / \partial \beta_j$ — how much the loss changes if you nudge $\beta_j$ slightly. Two facts give it:

1. Derivative of the sigmoid: $\dfrac{d\sigma}{dz} = \sigma(z)\,\big(1 - \sigma(z)\big)$.
   (Proof: $\sigma = (1+e^{-z})^{-1}$, so $\sigma' = e^{-z}(1+e^{-z})^{-2} = \sigma \cdot \dfrac{e^{-z}}{1+e^{-z}} = \sigma(1-\sigma)$.)
2. Chain rule applied to $l_{\log,i}$ with $\hat p_i = \sigma(z_i)$, $z_i = \beta_0 + \boldsymbol\beta^T\mathbf{x}_i$:

$$
\frac{\partial l_{\log,i}}{\partial z_i} = -\left[\frac{y_i}{\hat p_i} - \frac{1-y_i}{1-\hat p_i}\right]\hat p_i(1-\hat p_i) = \hat p_i - y_i
$$

Since $\partial z_i/\partial \beta_0 = 1$ and $\partial z_i/\partial \beta_j = x_{ij}$ (feature $j$ of point $i$):

$$
\frac{\partial L_{\log}}{\partial \beta_0} = \frac{1}{n}\sum_{i=1}^{n}(\hat p_i - y_i), \qquad
\frac{\partial L_{\log}}{\partial \beta_j} = \frac{1}{n}\sum_{i=1}^{n}(\hat p_i - y_i)\,x_{ij}
$$

Beautifully simple: "prediction minus truth", weighted by the feature.

### 6.3 Gradient descent

Repeat: move each parameter a small step **against** its gradient,

$$
\beta_j \leftarrow \beta_j - \eta\,\frac{\partial L_{\log}}{\partial \beta_j}
$$

where $\eta$ (eta), the **learning rate**, is the step size you choose.

### 6.4 Numerical example 5 — training a one-feature model

Data ($n = 6$), one feature:

| $x_i$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $y_i$ | 0 | 0 | 1 | 0 | 1 | 1 |

Start at $\beta_0 = 0$, $\beta_1 = 0$, learning rate $\eta = 0.5$.

**Step 0.** $z_i = 0$ for all points, so every $\hat p_i = 0.5$. Loss $= -\log(0.5) = 0.6931$.
Gradients: $\hat p_i - y_i = [0.5, 0.5, -0.5, 0.5, -0.5, -0.5]$.

$$
\frac{\partial L}{\partial \beta_0} = \frac{0.5+0.5-0.5+0.5-0.5-0.5}{6} = 0
$$

$$
\frac{\partial L}{\partial \beta_1} = \frac{0.5(1) + 0.5(2) - 0.5(3) + 0.5(4) - 0.5(5) - 0.5(6)}{6} = \frac{-3.5}{6} = -0.5833
$$

Update: $\beta_0 = 0 - 0.5(0) = 0$; $\beta_1 = 0 - 0.5(-0.5833) = 0.2917$.

**Step 1.** New predictions $\hat p = \sigma(0.2917\,x_i) = [0.572, 0.642, 0.706, 0.763, 0.811, 0.852]$, loss $= 0.672$ — already lower.

Continuing (computed with a short script):

| iteration | $\beta_0$ | $\beta_1$ | $L_{\log}$ |
|---|---|---|---|
| 0 | 0 | 0 | 0.693 |
| 1 | 0 | 0.292 | 0.672 |
| 10 | −0.558 | 0.283 | 0.579 |
| 100 | −2.883 | 0.866 | 0.429 |
| 1000 | −4.247 | 1.214 | 0.413 |
| 5000 | −4.249 | 1.214 | 0.413 (converged, gradient = 0) |

Final probabilities: $\hat p = [0.046, 0.139, 0.353, 0.647, 0.861, 0.954]$.

Translate back to the lecture's $\alpha, \gamma$ form ($\beta_1 = 1/\gamma$, $\beta_0 = -\alpha/\gamma$):

$$
\gamma = \frac{1}{1.214} = 0.82, \qquad \alpha = -\frac{\beta_0}{\beta_1} = \frac{4.249}{1.214} = 3.5
$$

The tipping point sits at $x = 3.5$ — exactly halfway through the overlap region where the classes are mixed (the class-I point at 4 and class-II point at 3). The loss does not reach 0 because the data themselves are not perfectly separable.

---

## 7. Assessing the accuracy of a binary classifier (board: page 8 onwards)

### 7.1 The four outcomes

Every prediction falls in one of four boxes. "Positive" means the model said $\hat y = 1$; "True/False" says whether it was right.

| Name | True $y$ | Predicted $\hat y$ | Verdict |
|---|---|---|---|
| **True positive (TP)** | 1 | 1 | correct |
| **True negative (TN)** | 0 | 0 | correct |
| **False positive (FP)** | 0 | 1 | error — **Type I error** (false alarm) |
| **False negative (FN)** | 1 | 0 | error — **Type II error** (a miss) |

TP, TN, FP, FN are also used as the **counts** of data points in each box.

### 7.2 The confusion matrix

Arranged in a 2 × 2 table, as drawn in the lecture (**rows = predicted $\hat y$, columns = true $y$**):

| | true $y = 1$ | true $y = 0$ |
|---|---|---|
| **predicted $\hat y = 1$** | TP ✓ | FP ✗ |
| **predicted $\hat y = 0$** | FN ✗ | TN ✓ |

Correct predictions sit on the diagonal. ⚠️ `scikit-learn`'s `confusion_matrix` uses the **opposite** layout (rows = true, columns = predicted) and puts class 0 first — always check which convention a source uses.

### 7.3 Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{correct predictions}}{\text{all predictions}}
$$

**Numerical example 6 — the model from §6.4.** With $p^{*} = 0.5$: $\hat p = [0.046, 0.139, 0.353, 0.647, 0.861, 0.954]$ gives $\hat y = [0, 0, 0, 1, 1, 1]$. Compare with $y = [0, 0, 1, 0, 1, 1]$:

- $x = 1, 2$: $y = 0, \hat y = 0$ → TN, TN
- $x = 3$: $y = 1, \hat y = 0$ → FN
- $x = 4$: $y = 0, \hat y = 1$ → FP
- $x = 5, 6$: $y = 1, \hat y = 1$ → TP, TP

| | true 1 | true 0 |
|---|---|---|
| pred 1 | TP = 2 | FP = 1 |
| pred 0 | FN = 1 | TN = 2 |

Accuracy $= (2 + 2)/6 = 0.667$.

### 7.4 Why accuracy fails on imbalanced data, and the fix

**The trap (board, page 9):** suppose 95 % of materials in your data set are semiconductors ($y = 0$) and 5 % are metals ($y = 1$). A useless model that **always predicts "semiconductor"** is right 95 % of the time — high accuracy, zero skill.

The fix is to measure performance **on each class separately**, then average.

**Sensitivity** (also called **recall**, or **true positive rate, TPR**) — of all truly positive points, what fraction did we catch?

$$
\text{Sensitivity} = TPR = \frac{TP}{TP + FN}
$$

**Specificity** (**true negative rate, TNR**) — of all truly negative points, what fraction did we correctly reject?

$$
\text{Specificity} = TNR = \frac{TN}{TN + FP}
$$

**Balanced accuracy** — the plain average of the two:

$$
\text{Balanced accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{TPR + TNR}{2}
$$

It is **useful for imbalanced data sets** because each class gets equal weight regardless of how many points it has. A coin-flip or always-one-answer model scores 0.5; a perfect model scores 1.

**Numerical example 7 — 1000 materials: 950 semiconductors ($y=0$), 50 metals ($y=1$).**

*Model A: always predicts semiconductor.* Every metal is missed, every semiconductor is correct: TP = 0, FN = 50, TN = 950, FP = 0.

- Accuracy $= (0 + 950)/1000 = \mathbf{0.950}$
- TPR $= 0/(0+50) = 0$; TNR $= 950/(950+0) = 1$
- Balanced accuracy $= (0 + 1)/2 = \mathbf{0.500}$

*Model B: a real classifier.* It catches 40 of the 50 metals and wrongly flags 95 semiconductors: TP = 40, FN = 10, TN = 855, FP = 95.

- Accuracy $= (40 + 855)/1000 = \mathbf{0.895}$
- TPR $= 40/50 = 0.80$; TNR $= 855/950 = 0.90$
- Balanced accuracy $= (0.80 + 0.90)/2 = \mathbf{0.850}$

| | Accuracy | Balanced accuracy |
|---|---|---|
| Model A (useless) | 0.950 | 0.500 |
| Model B (useful) | 0.895 | 0.850 |

Plain accuracy ranks them **the wrong way round**; balanced accuracy exposes Model A.

### 7.5 Moving the threshold $p^{*}$ (beyond the board, but follows directly)

Lowering $p^{*}$ makes the model say "1" more often → more TP, fewer FN (sensitivity ↑), but more FP (specificity ↓). Raising it does the reverse.

On the §6.4 model with $p^{*} = 0.3$: $\hat p \ge 0.3$ for $x = 3, 4, 5, 6$ → $\hat y = [0, 0, 1, 1, 1, 1]$. Now TP = 3, FN = 0, FP = 1, TN = 2: sensitivity $= 3/3 = 1.00$ (up from 2/3), specificity $= 2/3 = 0.67$ (unchanged here). Choose $p^{*}$ according to which error is costlier — e.g. missing a toxic compound (FN) is usually worse than a false alarm (FP).

---

## 8. Formula sheet

| Quantity | Formula |
|---|---|
| Logistic function (1 feature) | $p(x) = \dfrac{1}{1 + \exp(-(x-\alpha)/\gamma)}$; $p(\alpha) = 0.5$; $\gamma$ = width |
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$; $\sigma'(z) = \sigma(1-\sigma)$ |
| Log-odds | $\log\dfrac{p}{1-p} = \beta_0 + \boldsymbol\beta^T\mathbf{x}$ |
| Multi-feature model | $P(y=1\mid\mathbf{x}) = \sigma(\beta_0 + \boldsymbol\beta^T\mathbf{x})$ |
| Link to $\alpha,\gamma$ | $\beta_0 = -\sum_i \alpha_i/\gamma_i$, $\;\boldsymbol\beta = [1/\gamma_1, \dots, 1/\gamma_p]$ |
| Decision rule | $\hat y = 1$ if $P(y=1\mid\mathbf{x}) \ge p^{*}$ (default $p^{*} = 0.5$), else $\hat y = 0$ |
| Binary cross-entropy | $L_{\log} = -\frac1n\sum_i \big[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\big]$ |
| Gradient | $\partial L/\partial\beta_0 = \frac1n\sum(\hat p_i - y_i)$; $\;\partial L/\partial\beta_j = \frac1n\sum(\hat p_i - y_i)x_{ij}$ |
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ |
| Sensitivity = recall = TPR | $TP/(TP+FN)$ |
| Specificity = TNR | $TN/(TN+FP)$ |
| Balanced accuracy | $(TPR + TNR)/2$ |
| Type I / Type II error | FP / FN |

---

## 9. Self-check questions

1. For $\alpha = 2$, $\gamma = 0.5$, compute $p(3)$. *(Answer: $z = 2$, $p = 0.881$.)*
2. A model has $\beta_0 = -3$, $\beta_1 = 1.5$. At what $x$ is $p = 0.5$? *(Answer: $x = -\beta_0/\beta_1 = 2$.)*
3. True class 0, predicted $\hat p = 0.95$. What is $l_{\log,i}$? *(Answer: $-\log(0.05) = 3.00$.)*
4. Why can't you just fit $y \in \{0,1\}$ with ordinary linear regression? *(Outputs outside [0, 1]; no probability interpretation.)*
5. A cancer screen: TP = 90, FN = 10, TN = 800, FP = 100. Compute accuracy, sensitivity, specificity, balanced accuracy. *(Answer: 0.890, 0.900, 0.889, 0.894.)*
6. Which error type is a false negative? *(Type II.)*
7. In a data set with 99 % class 0, what balanced accuracy does "always predict 0" get? *(0.5.)*

---

## 10. One-paragraph summary

Logistic regression turns a linear combination of features, $z = \beta_0 + \boldsymbol\beta^T\mathbf{x}$, into a probability with the S-shaped logistic (sigmoid) function $\sigma(z) = 1/(1+e^{-z})$, which models $P(y=1\mid\mathbf{x})$; for one feature the curve is centred at $\alpha$ (where $p = 0.5$) with width $\gamma$. A threshold $p^{*}$ (usually 0.5) converts the probability into a class label. The parameters are found by numerically minimising the binary cross-entropy loss, which is zero for confident correct predictions and grows without bound for confident wrong ones; its gradient is simply "prediction minus truth" times the feature. Performance is summarised by the confusion matrix (TP, TN, FP = Type I, FN = Type II). Plain accuracy is misleading when one class dominates, so use sensitivity (TPR), specificity (TNR), and their average, the balanced accuracy.
