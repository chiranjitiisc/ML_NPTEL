# Lecture 6 — Conversion of Random Variables to Other Distributions

## Motivation

Programming languages only provide built-in functions to sample from a few **specific** probability distributions (e.g. the **uniform** or **normal** distribution).

To generate samples from other distributions, we use **variable transformations**: we take a random variable sampled from a distribution we *can* generate and apply a function to it so that the result follows the distribution we *want*.

$$
X \sim U(0,1) \quad \xrightarrow{\ Y = f(X)\ } \quad Y \sim \text{Exp}(\lambda)
$$

The question is: **what function $Y = f(X)$** transforms $X$ into the target distribution?

---

## General Transformation Rule 

Given a random variable $X$ with:
- pdf $f(x)$
- cdf $F(x)$

Let $Y = r(X)$, and let:
- $g(y)$ denote the pdf of $Y$
- $G(y)$ denote the cdf of $Y$
- $r^{-1}(y)$ denote the inverse of $r(x)$

Then the distribution of $Y$ depends on whether $r^{-1}$ is increasing or decreasing.

### Case 1 — $r^{-1}$ is an *increasing* function

$$
\text{cdf:} \quad G(y) = F\!\left[r^{-1}(y)\right]
$$

$$
\text{pdf:} \quad g(y) = f\!\left(r^{-1}(y)\right) \cdot \frac{d}{dy}\left(r^{-1}(y)\right)
$$

### Case 2 — $r^{-1}$ is a *decreasing* function

$$
\text{cdf:} \quad G(y) = 1 - F\!\left[r^{-1}(y)\right]
$$

$$
\text{pdf:} \quad g(y) = -\,f\!\left(r^{-1}(y)\right) \cdot \frac{d}{dy}\left(r^{-1}(y)\right)
$$

> The two cases combine into the single expression using the absolute value of the derivative (Jacobian):
> $g(y) = f\!\left(r^{-1}(y)\right)\,\left|\dfrac{d}{dy}\,r^{-1}(y)\right|$

---

## Worked Example: Uniform → Exponential

**Problem.** Let $X \sim U(0,1)$. Find the distribution followed by

$$
y = \frac{1}{\alpha}\ln\!\left(\frac{1}{x}\right), \qquad \alpha > 0.
$$

**Step 1 — pdf of $X$.** Since $X \sim U(0,1)$:

$$
f(x) =
\begin{cases}
1, & 0 \le x \le 1 \\
0, & \text{otherwise}
\end{cases}
$$

**Step 2 — Find the inverse $r^{-1}(y)$.**

$$
r(x) = \frac{1}{\alpha}\ln\!\left(\frac{1}{x}\right) = y
\;\Rightarrow\; \ln\!\left(\frac{1}{x}\right) = \alpha y
\;\Rightarrow\; \frac{1}{x} = e^{\alpha y}
$$

$$
\boxed{\,r^{-1}(y) = x = e^{-\alpha y}\,}
$$

**Step 3 — Apply the transformation formula.**

Here $r^{-1}(y) = e^{-\alpha y}$ is a *decreasing* function, so use the decreasing case:

$$
g(y) = -\,f\!\left(r^{-1}(y)\right) \cdot \frac{d}{dy}\left(r^{-1}(y)\right)
$$

$$
= -\,f\!\left(e^{-\alpha y}\right) \cdot \frac{d}{dy}\left(e^{-\alpha y}\right)
$$

$$
= -\,f\!\left(e^{-\alpha y}\right) \times \left(-\alpha e^{-\alpha y}\right)
$$

$$
= \alpha\, e^{-\alpha y}\, f\!\left(e^{-\alpha y}\right)
$$

**Step 4 — Substitute the pdf of $X$.**

$$
f\!\left(e^{-\alpha y}\right) =
\begin{cases}
1, & 0 \le e^{-\alpha y} \le 1 \\
0, & \text{otherwise}
\end{cases}
$$

**Determine the valid range of $y$** from the condition $0 \le e^{-\alpha y} \le 1$. Taking logs:

$$
\log(0) \le -\alpha y \le \log(1)
$$

$$
-\infty < -\alpha y \le 0
$$

$$
\infty > \alpha y \ge 0
$$

$$
\boxed{\,\infty > y \ge 0\,} \quad (\alpha > 0)
$$

So $f(e^{-\alpha y}) = 1$ for $0 \le y < \infty$ and $0$ otherwise.

**Step 5 — Final result.**

$$
g(y) =
\begin{cases}
\alpha\, e^{-\alpha y}, & 0 \le y < \infty \\
0, & \text{otherwise}
\end{cases}
$$

This is exactly the **pdf of an exponentially distributed random variable** $Y$:

$$
Y \sim \text{Exp}(\alpha)
$$

---

## Key Takeaway

$$
X \sim U(0,1) \quad \xrightarrow{\; y = \frac{1}{\lambda}\ln\left(\frac{1}{x}\right)\;} \quad Y \sim \text{Exp}(\lambda)
$$

By applying the transformation $y = \frac{1}{\lambda}\ln(1/x)$ to a uniform random variable, we obtain an exponentially distributed random variable. This is the basis of the **inverse transform (inverse CDF) sampling method** — a general technique for generating samples from a target distribution using only uniform samples.
