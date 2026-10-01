# Lecture 07 — Population vs. Sample Parameters

## Population vs. Sample

**Population:** the entire set of data points that we are interested in studying.

**Sample:** a subset of the population that is selected for analysis in a statistical study. It should be **representative** of the population to ensure the accuracy of the statistical analysis.

### Parameters vs. properties

**Population parameters:** mean ($\mu$), variance ($\sigma^2$), standard deviation ($\sigma$), etc.

**Sample properties:** sample mean ($\bar{X}$), sample variance ($s^2$), sample standard deviation ($s$).

A population's statistical properties can be inferred from the underlying **probability density function (pdf)**, written $f_X(x)$ or $f(x)$.

---

## 1. Population Mean / Expectation / Expected Value

$$\mu = \langle X \rangle = E(X) = \int_{-\infty}^{\infty} x\, f(x)\, dx$$

- This is the **first moment** of the pdf.
- The term $f(x)\,dx$ is the probability that $X$ lies between $x$ and $(x + dx)$.

---

## 2. Population Variance

Quantifies the extent of spread in a random variable following a certain distribution.

$$\sigma^2 = \text{Var}(X) = \boxed{E(X^2) - [E(X)]^2}$$

where

$$E(X^2) = \int_{-\infty}^{\infty} x^2 f(x)\, dx \quad \text{(second moment of the pdf)}$$

$$E(X) = \int_{-\infty}^{\infty} x\, f(x)\, dx = \mu$$

---

## 3. Standard Deviation

Square root of the variance, and thus has the **same unit** as the random variable.

$$\sigma = \sqrt{\text{Var}(X)} = \sqrt{E(X^2) - (E(X))^2}$$

### Normal / Gaussian distribution

$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\, \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$\mu = E(X) = \int_{-\infty}^{\infty} x \cdot f(x)\, dx$$

The pdf is a bell-shaped curve centered at $\mu$.

---

## 4. Covariance

For two random variables $X$ and $Y$, the covariance provides an understanding of the relationship between them.

$$\text{Cov}(X, Y) = E\big[(X - E(X))\cdot(Y - E(Y))\big]$$

Expanding (using that the **expectation operator is a linear operator**):

$$= E\big[XY - X\,E(Y) - E(X)\,Y + E(X)E(Y)\big]$$
$$= E(XY) - E(Y)E(X) - E(X)E(Y) + E(X)E(Y)$$

$$\boxed{\text{Cov}(X, Y) = E(XY) - E(X)\,E(Y)}$$

### Special case: $Y = X$

$$\text{Cov}(X, X) = E(X^2) - (E(X))^2 = \text{Var}(X)$$

### Example — independent random variables

Let $X$ and $Y$ be two **independent** random variables with pdfs $f_X(x)$ and $f_Y(y)$. Find the covariance of $X$ and $Y$.

For independent variables, the joint pdf factorizes:

$$f_{XY}(x, y) = f_X(x)\, f_Y(y)$$

Then

$$\text{Cov}(X, Y) = E(XY) - E(X)\cdot E(Y)$$
$$= \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} xy\, f_X(x) f_Y(y)\, dx\, dy - \left(\int_{-\infty}^{\infty} x f_X(x)\, dx\right)\!\left(\int_{-\infty}^{\infty} y f_Y(y)\, dy\right)$$

The double integral separates into the product of the two single integrals, so:

$$= \left(\int_{-\infty}^{\infty} x f_X(x)\, dx\right)\!\left(\int_{-\infty}^{\infty} y f_Y(y)\, dy\right) - \left(\int_{-\infty}^{\infty} x f_X(x)\, dx\right)\!\left(\int_{-\infty}^{\infty} y f_Y(y)\, dy\right) = \boxed{0}$$

**Independent random variables have zero covariance.**

### Sign of the covariance

- $\text{Cov}(X, Y) > 0$ — larger values of $X$ correspond to larger values of $Y$ (positive/upward trend).
- $\text{Cov}(X, Y) \approx 0$ — no linear relationship (scattered cloud).
- $\text{Cov}(X, Y) < 0$ — larger values of $X$ correspond to lower values of $Y$ (negative/downward trend).

---

## 5. Correlation Coefficient

A **normalized** measure of the correlation between two random variables.

**Pearson's correlation coefficient:**

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\,\text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X\,\sigma_Y}$$

$$-1 \le \rho_{XY} \le 1$$

### Example — perfectly linearly related variables

Let $Y = mX + c$.

**Variance of $Y$:**

$$\text{Var}(Y) = \text{Var}(mX + c) = \text{Var}(mX) = m^2\,\text{Var}(X)$$

(adding a constant $c$ does not change the variance).

**Covariance of $X$ and $Y$:**

$$\text{Cov}(X, Y) = E(XY) - E(X)E(Y)$$
$$= E\big(X(mX + c)\big) - E(X)\cdot E(mX + c)$$
$$= E(mX^2 + cX) - E(X)\cdot\big[m E(X) + c\big]$$
$$= m E(X^2) + c E(X) - m (E(X))^2 - c E(X)$$
$$= m\big(E(X^2) - (E(X))^2\big)$$
$$= m\,\text{Var}(X)$$

**Correlation coefficient:**

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X\,\sigma_Y} = \frac{m\,\text{Var}(X)}{\sqrt{\text{Var}(X)\cdot m^2\,\text{Var}(X)}} = \frac{m\,\text{Var}(X)}{m\,\text{Var}(X)} = \boxed{1}$$

So $\rho_{XY} = 1$ **if two random variables are perfectly linearly related.**
