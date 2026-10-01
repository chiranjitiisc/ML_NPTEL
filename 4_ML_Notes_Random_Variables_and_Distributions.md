# Machine Learning — Lecture Notes

## Random Variables, Distributions & a Bayes' Theorem Example (NPTEL, IISc)

---

## 1. Bayes' Theorem — Worked Example

**Problem.** In an ML class, **40%** of the students are from a **science** background and the remaining **60%** are from **engineering**. Of the science students, **80%** have *no prior exposure* to ML, and this number is **70%** for engineering students. Find the probability of a student being from a **science background given that they have some exposure to ML**.

### Setting up the events

- **S** = student is from a science background → $P(S) = 0.40$
- **E** = student is from an engineering background → $P(E) = 0.60$
- **N** = *no* prior exposure to ML, **X** = *some* exposure to ML (X is the complement of N).

From the data (careful — the percentages are for *no* exposure):

- $P(N \mid S) = 0.80 \Rightarrow P(X \mid S) = 1 - 0.80 = 0.20$
- $P(N \mid E) = 0.70 \Rightarrow P(X \mid E) = 1 - 0.70 = 0.30$

**In easy terms:** We're told how many have *no* exposure, but the question asks about students who *do* have exposure — so we flip each percentage first.

### Applying Bayes' theorem

We want $P(S \mid X)$ — the probability the student is from science, *given* they have some ML exposure:

$$P(S \mid X) = \frac{P(X \mid S)\,P(S)}{P(X \mid S)\,P(S) + P(X \mid E)\,P(E)}$$

The denominator is just the **total probability** of the evidence X (a student having exposure), summed over both backgrounds.

$$P(S \mid X) = \frac{(0.20)(0.40)}{(0.20)(0.40) + (0.30)(0.60)} = \frac{0.08}{0.08 + 0.18} = \frac{0.08}{0.26} \approx 0.3077$$

$$\boxed{P(S \mid X) \approx 30.8\%}$$

**Takeaway:** Even though science students make up 40% of the class, most of them have *no* ML exposure, so among the students who *do* have exposure, the science share drops to about 31%. This is Bayes' theorem doing exactly what it's meant to do — updating a prior belief (40%) in light of evidence (has exposure) to give a posterior (~31%).

---

## 2. Random Variables

**Definition:** A **random variable** is a variable that denotes the **outcome of a stochastic (random) experiment**.

Random variables come in two types:

$$\text{Random variable} \longrightarrow \begin{cases} \textbf{Discrete} \\ \textbf{Continuous} \end{cases}$$

**In easy terms:** A random variable is just a number attached to the result of something random. Roll a die → the number you get is a discrete random variable. Measure the exact height of a person → that's a continuous random variable.

- **Discrete** = takes *countable*, separate values (e.g. 1, 2, 3, …).
- **Continuous** = can take *any value* in a range (e.g. any real number in an interval).

---

## 3. Two Functions Associated with a Random Variable

Two functions can be associated with a random variable that follows a given **probability distribution**:

$$\text{Probability distribution} \longrightarrow \begin{cases} \textbf{Probability density function (pdf)} \\ \textbf{Cumulative distribution function (cdf)} \end{cases}$$

**In easy terms:** The **pdf** tells you how densely probability is packed around each value; the **cdf** tells you the total probability accumulated *up to* a value. They are two views of the same distribution.

---

## 4. Continuous Random Variables and the pdf

If a continuous random variable **X** follows a probability distribution with pdf **f(x)**, then the probability of X lying in the small interval between $x$ and $(x + dx)$ is:

$$P(x \le X \le x + dx) = f(x)\, dx$$

Rearranging gives the meaning of the density itself:

$$f(x) = \frac{P(x \le X \le x + dx)}{dx}$$

**In easy terms:** For a *continuous* variable, the probability of hitting any single exact value is zero, so we talk about the probability of landing in a *tiny interval*. The pdf $f(x)$ is that probability *per unit length* — a density, not a probability. To get an actual probability you must multiply by a width (or integrate over one).

### Probability over an interval

The probability of X lying between two possible values $a$ and $b$ is the **area under the pdf** between them:

$$\boxed{P(a \le X \le b) = \int_a^b f(x)\, dx}$$

*(Picture: the pdf curve $f(x)$ with the shaded region between $x=a$ and $x=b$ — that shaded area **is** the probability.)*

---

## 5. The Cumulative Distribution Function (cdf)

The interval probability can also be written using the cdf $F(x)$:

$$P(a \le X \le b) = \int_a^b f(x)\, dx = F(b) - F(a)$$

### Deriving the cdf

Consider the special case $a \to -\infty$ and $b$ replaced by $x$. Then we obtain the **cdf**:

$$P(-\infty < X \le x) = \int_{-\infty}^{x} f(x')\, dx' = F(x) - F(-\infty) = F(x)$$

Here $F(-\infty) = 0$, because **there is no possible value of $x$ below $-\infty$** — no probability can accumulate there.

**Meaning of the cdf:** The cdf $F(x)$ gives the probability that the random variable X has a value **less than or equal to the argument** of the cdf, i.e. $\le x$.

$$F(x) = P(X \le x)$$

**In easy terms:** The cdf is a "running total." $F(x)$ answers *"what's the chance X comes out at $x$ or lower?"* It starts at 0 far to the left and climbs to 1 far to the right.

### The total probability must be 1

What is $F(\infty)$?

$$F(\infty) = \int_{-\infty}^{\infty} f(x)\, dx = \boxed{1}$$

This says the **total probability of X taking *some* value is 1** — for $x \in \mathbb{R}$, i.e. $x \in (-\infty, \infty)$. Every distribution's pdf must integrate to 1 over all possible values.

---

## 6. Multiple Random Variables — Joint Distributions

Often in ML we deal with **many more variables** at once. For multiple random variables, we can define the **joint probability density function (jpdf)**.

For two random variables X and Y:

$$P(a \le X \le b,\; c \le Y \le d) = \int_a^b \int_c^d f(x, y)\, dy\, dx$$

where $f(x,y)$ is the **jpdf**.

### Evaluating via the joint cdf

Carrying out the integration in terms of the joint cdf $F(x,y)$:

$$= \Big[\,[F(x,y)]_{y=c}^{y=d}\,\Big]_{x=a}^{x=b} = \Big[\,F(x,d) - F(x,c)\,\Big]_{x=a}^{x=b}$$

$$= F(b,d) - F(a,d) - F(b,c) + F(a,c)$$

Here $F(x,y)$ is called the **joint cumulative distribution function**, defined as:

$$F(x, y) = P(-\infty < X \le x,\; -\infty < Y \le y)$$

**In easy terms:** The joint cdf accumulates probability over a *rectangle* in the $(x,y)$ plane. To get the probability inside the box $[a,b]\times[c,d]$, you take the corner values of $F$ and combine them with alternating signs — the 2-D version of $F(b) - F(a)$.

### Independent random variables

If X and Y are **independent** random variables, the joint pdf factorizes into the product of the individual (marginal) pdfs:

$$\boxed{f_{XY}(x, y) = f_X(x)\, f_Y(y)}$$

**In easy terms:** "Independent" means knowing one tells you nothing about the other — so their joint density is just the two densities multiplied together.

### Getting the pdf back from the cdf

Given the joint cdf, one can obtain the pdf by **computing the derivative**:

$$f_X(x) = \frac{dF_X(x)}{dx} \qquad ; \qquad f_X(x) = \frac{\partial F_{XY}(x, \infty)}{\partial x}$$

**In easy terms:** The pdf is the *slope* (derivative) of the cdf. Integrating the pdf builds up the cdf; differentiating the cdf recovers the pdf. Taking $y \to \infty$ in the joint cdf "sums out" Y, leaving the **marginal** cdf of X, whose derivative is the marginal pdf $f_X(x)$.

---

## Quick Recap

1. **Bayes' example:** flip the "no exposure" figures, then apply $P(S\mid X) = \dfrac{P(X\mid S)P(S)}{P(X\mid S)P(S) + P(X\mid E)P(E)} = \dfrac{0.08}{0.26} \approx 30.8\%$.
2. **Random variable** = numeric outcome of a random experiment; **discrete** or **continuous**.
3. A distribution has two views: **pdf** $f(x)$ (density) and **cdf** $F(x)$ (accumulated probability).
4. **pdf:** $P(x \le X \le x+dx) = f(x)\,dx$; interval probability $P(a\le X\le b) = \int_a^b f(x)\,dx$.
5. **cdf:** $F(x) = P(X \le x)$, with $F(-\infty)=0$ and $F(\infty)=\int_{-\infty}^{\infty} f(x)\,dx = 1$.
6. **Joint distributions:** $P(a\le X\le b, c\le Y\le d) = \int_a^b\int_c^d f(x,y)\,dy\,dx$; joint cdf $F(x,y)=P(X\le x, Y\le y)$.
7. **Independence:** $f_{XY}(x,y) = f_X(x)f_Y(y)$.
8. **pdf ↔ cdf:** differentiate the cdf to get the pdf; $f_X(x) = \dfrac{dF_X(x)}{dx}$.

---

*End of notes — Random Variables, Distributions & Joint Distributions for AI/ML.*
