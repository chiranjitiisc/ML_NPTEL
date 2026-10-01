# Machine Learning — Lecture Notes

## Probability Distributions (NPTEL, IISc — Lecture 5)

---

## Overview

A **probability distribution** describes how the probabilities are spread across the possible values of a random variable. Distributions come in two families:

- **Discrete random variables** — take *countable, separate* values (e.g., 0, 1, 2, …). Described by a **Probability Mass Function (PMF)**, $P(X = k)$.
- **Continuous random variables** — take *any value* in a range. Described by a **Probability Density Function (PDF)**, $f(x)$.

**In easy terms:** A random variable is a number whose value depends on chance (e.g., the outcome of a coin toss). The distribution is the "rulebook" that tells you how likely each outcome is.

---

# Part A — Discrete Random Variables

## 1. Bernoulli Distribution

A random variable with **only two possible outcomes** — say **success (1)** or **failure (0)**.

$$X \sim \text{Bern}(p), \quad p \in [0, 1]$$

where **p = probability of success.**

**Probability Mass Function:**

$$
P(X = x) =
\begin{cases}
p & x = 1 \\
1 - p & x = 0 \\
0 & \text{otherwise}
\end{cases}
$$

**In easy terms:** This is the simplest possible random experiment — a *single* trial with a yes/no (or success/failure) outcome.

*Example:* A single coin toss (Heads = success), a single "will the customer click the ad? (yes/no)", one exam question (pass/fail).

---

## 2. Binomial Distribution

**Generalizes the Bernoulli distribution** to **n independent and identical trials.** Each trial has the same probability **p** of success.

$$X \sim \text{Bin}(n, p)$$

Here **X represents the number of successes in n independent Bernoulli trials**, each with success probability p.

**Probability Mass Function:**

$$P(X = k) = \binom{n}{k}\, p^{k}\, (1 - p)^{\,n-k}, \quad k = 0, 1, 2, \dots, n$$

Where $\binom{n}{k} = {}^{n}C_{k}$ counts the number of ways to choose *k* successes out of *n* trials.

**In easy terms:** If Bernoulli is *one* coin toss, Binomial answers "if I toss the coin **n** times, what's the probability of getting exactly **k** heads?" The three pieces of the formula are: number of ways to arrange the successes ($^{n}C_{k}$), probability of the k successes ($p^k$), and probability of the remaining failures ($(1-p)^{n-k}$).

*Example:* Tossing a coin 10 times and asking for the probability of exactly 6 heads; number of defective items in a batch of 100.

---

## 3. Poisson Distribution

Models the **number of events occurring in a fixed interval of time (or space)**, assuming each event occurs **independently** and at a **constant average rate.**

$$X \sim \text{Poisson}(\lambda)$$

where **λ (lambda) = rate parameter** = the average rate of the process or event.

**Probability Mass Function:**

$$P(X = k) = \frac{\lambda^{k}\, e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots$$

**In easy terms:** Use this when you're counting how many times something happens in a set window, and you know the *average* rate. There's no fixed "number of trials" — events just happen over time.

*Example:* Number of emails received per hour, number of calls at a call center per minute, number of typos per page, number of customers arriving per hour.

---

## 4. Geometric Distribution

Models the **number of Bernoulli trials needed to get the first success.**

$$X \sim \text{Geom}(p), \quad 0 < p \le 1$$

**Probability Mass Function:**

$$P(X = k) = (1 - p)^{\,k-1}\, p, \quad k = 1, 2, 3, \dots$$

**In easy terms:** "How many attempts until I finally succeed?" The formula says: fail the first $(k-1)$ times, each with probability $(1-p)$, then succeed on the k-th try with probability $p$.

*Example:* Number of coin tosses until the first head; number of sales calls until the first sale; number of attempts until a machine part passes inspection.

---

# Part B — Continuous Random Variables

## 1. Uniform Distribution

Models a continuous random variable **X ~ U(a, b)** which has **equal probability** of lying **anywhere** in the domain $x \in (a, b)$.

**Probability Density Function (PDF):**

$$
f(x) =
\begin{cases}
\dfrac{1}{b - a} & a \le x \le b \\[2mm]
0 & x < a \\
0 & x > b
\end{cases}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x) =
\begin{cases}
0 & x < a \\[1mm]
\dfrac{x - a}{b - a} & a \le x \le b \\[2mm]
1 & x > b
\end{cases}
$$

**In easy terms:** Every value between *a* and *b* is *equally likely* — the "flat" distribution. The PDF is a flat line of constant height. The **CDF** (cumulative distribution function) tells you the probability that X is *less than or equal to* a given value; it rises steadily from 0 to 1.

*Example:* A random number generator producing values evenly between 0 and 1; a bus that could arrive at any moment in a 10-minute window.

---

## 2. Normal / Gaussian Distribution

$$X \sim N(\mu, \sigma^{2})$$

where **μ (mu) = mean** and **σ² (sigma squared) = variance.**

Commonly known as the **"bell curve"** — one of the most commonly encountered distributions in nature and statistics.

**Central Limit Theorem (CLT):** The **normalized sum of many independent and identical distributions tends to the normal distribution.** *(This is why the bell curve appears everywhere — sums/averages of random effects naturally become Gaussian.)*

**Probability Density Function:**

$$f_X(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\, \exp\!\left(-\frac{(x - \mu)^{2}}{2\sigma^{2}}\right)$$

Where:

- **μ** = mean of the distribution (center of the bell — the peak sits at $x = \mu$).
- **σ** = standard deviation (controls the width/spread of the bell).

**In easy terms:** The bell curve is symmetric around its mean. Most values cluster near the center, and values far from the mean become increasingly rare (the thin "tails"). The CLT is the deep reason it's so common: whenever an outcome is the sum of many small independent random influences, it ends up looking Gaussian.

*Examples:* Student marks in an exam, heights of people in an institution, atomic velocities in a fluid.

---

## 3. Student's t-Distribution

Used for **determining the confidence intervals** for a linear regression fit — the range in which the **true mean of a parameter** lies. It **quantifies the extent of deviation from the true mean**, especially useful when the sample size is small.

**The t-statistic (a random variable following Student's t-distribution):**

$$t = \frac{\bar{X} - \mu}{s / \sqrt{n}}$$

Where:

- $\bar{X}$ = **sample mean** = $\dfrac{1}{n}\sum_{i=1}^{n} x_i$
- $s^{2}$ = **sample variance** = $\dfrac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{X})^{2}$
- $\nu$ (nu) = **number of degrees of freedom** = $(n - 1)$

**Probability Density Function:**

$$f(t) = \frac{\Gamma\!\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi}\;\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^{2}}{\nu}\right)^{-\left(\frac{\nu + 1}{2}\right)}$$

Where **Γ (Gamma function)** generalizes the factorial:

$$\Gamma(x) = \int_{0}^{\infty} u^{\,x-1} e^{-u}\, du = (x - 1)!$$

**In easy terms:** When you only have a *small sample* of data, you can't be fully sure of the true average. The t-distribution looks like a bell curve but with **fatter tails**, accounting for that extra uncertainty. As the sample grows large, it approaches the normal distribution. "Degrees of freedom" $(n-1)$ reflects how much independent information you have.

*Example:* Estimating the true average effect of a drug from a small clinical trial and reporting a confidence interval.

---

## 4. Exponential Distribution

Represents the probability underlying the **time interval between two incidences of an event.**

$$X \sim \text{Exp}(\lambda)$$

where **λ = rate parameter.**

**Probability Density Function:**

$$
f_X(x) =
\begin{cases}
\lambda\, e^{-\lambda x} & x \ge 0 \\
0 & x < 0
\end{cases}
$$

The curve starts at height λ (at x = 0) and **decays** toward zero as x increases.

**In easy terms:** While the **Poisson** distribution counts *how many* events happen in an interval, the **Exponential** distribution models *how long you wait* between consecutive events. Short waiting times are most likely; long waits become exponentially rarer. (They are two sides of the same coin — both governed by the rate λ.)

*Example:* Time between two successive customer arrivals, time until a radioactive atom decays, time between failures of a machine.

---

## Quick Recap Table

| Distribution | Type | Models | Key parameter(s) |
|---|---|---|---|
| **Bernoulli** | Discrete | Single success/failure trial | p |
| **Binomial** | Discrete | # successes in n trials | n, p |
| **Poisson** | Discrete | # events in a fixed interval | λ (rate) |
| **Geometric** | Discrete | # trials until first success | p |
| **Uniform** | Continuous | Equal chance over (a, b) | a, b |
| **Normal / Gaussian** | Continuous | Bell curve; sums of random effects | μ, σ² |
| **Student's t** | Continuous | Confidence intervals, small samples | ν = n−1 |
| **Exponential** | Continuous | Time between events | λ (rate) |

**Two big connections to remember:**

- **Bernoulli → Binomial:** one trial generalized to n trials.
- **Poisson ↔ Exponential:** Poisson counts events per interval; Exponential measures the waiting time between them.
- **Central Limit Theorem:** sums of many independent random things → Normal distribution.

---

*End of notes — Probability Distributions for AI/ML.*
