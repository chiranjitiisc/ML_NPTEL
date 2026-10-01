# Probability Distributions — Complete Notes

## Concepts, Easy Derivations, and Worked Examples

---

# 1. Probability Distributions

A **random variable** can take different values. A probability distribution tells us:

> What values can the random variable take, and how likely is each value?

## Types

### Discrete
Countable values:

\[
0,1,2,3,\ldots
\]

Examples: number of heads, number of defects, number of adsorption events.

### Continuous
Infinitely many possible values in an interval.

Examples: temperature, pressure, height, lifetime.

---

# 2. Bernoulli Distribution

A Bernoulli experiment has exactly two outcomes:

\[
\boxed{1=\text{Success},\qquad 0=\text{Failure}}
\]

If:

\[
X\sim\operatorname{Bern}(p)
\]

then:

\[
\boxed{P(X=1)=p}
\]

and:

\[
\boxed{P(X=0)=1-p}
\]

## Example

A molecule adsorbs with probability:

\[
p=0.7
\]

Define:

\[
X=
\begin{cases}
1,&\text{adsorbs}\\
0,&\text{does not adsorb}
\end{cases}
\]

Therefore:

\[
\boxed{P(X=1)=0.7}
\]

\[
\boxed{P(X=0)=0.3}
\]

**Memory trick:**

\[
\boxed{\text{Bernoulli = ONE trial with TWO outcomes}}
\]

---

# 3. Binomial Distribution

Bernoulli describes one trial. Binomial describes a fixed number of independent Bernoulli trials.

Let:

\[
X=\text{number of successes}
\]

Then:

\[
\boxed{X\sim\operatorname{Bin}(n,p)}
\]

## Conditions

1. Fixed number of trials \(n\)
2. Two outcomes per trial
3. Same probability \(p\)
4. Independent trials

## Formula

\[
\boxed{
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
}
\]

where:

\[
\boxed{
\binom nk=\frac{n!}{k!(n-k)!}
}
\]

## Easy derivation

For exactly \(k\) successes:

- probability of successes: \(p^k\)
- probability of failures: \((1-p)^{n-k}\)
- number of possible arrangements: \(\binom nk\)

Therefore:

\[
\boxed{
P(X=k)=\binom nk p^k(1-p)^{n-k}
}
\]

## Example: 3 adsorptions out of 5

\[
n=5,\quad p=0.7,\quad k=3
\]

\[
P(X=3)=\binom53(0.7)^3(0.3)^2
\]

\[
=10(0.343)(0.09)
\]

\[
\boxed{P(X=3)=0.3087}
\]

or:

\[
\boxed{30.87\%}
\]

## Example: 2 heads in 4 fair coin tosses

\[
P(X=2)=\binom42(0.5)^2(0.5)^2
\]

\[
=6(0.5)^4
\]

\[
\boxed{P(X=2)=0.375}
\]

**Memory trick:**

\[
\boxed{\text{Binomial = How many successes in fixed }n\text{ trials?}}
\]

---

# 4. Geometric Distribution

Repeat independent trials until the **first success**.

Let:

\[
X=\text{number of trials until first success}
\]

Then:

\[
\boxed{X\sim\operatorname{Geom}(p)}
\]

## Formula

For the first success on trial \(k\):

- first \(k-1\) trials fail
- the \(k\)-th trial succeeds

Therefore:

\[
\boxed{
P(X=k)=(1-p)^{k-1}p
}
\]

## Example: first adsorption on third trial

\[
p=0.7
\]

Required sequence:

\[
F,F,S
\]

Therefore:

\[
P(X=3)=(0.3)^2(0.7)
\]

\[
=0.09(0.7)
\]

\[
\boxed{P(X=3)=0.063=6.3\%}
\]

**Memory trick:**

\[
\boxed{\text{Geometric = When does the FIRST success occur?}}
\]

---

# 5. Poisson Distribution

Poisson answers:

> How many events occur in a fixed interval?

The interval may be time, area, distance, or volume.

\[
\boxed{X\sim\operatorname{Poisson}(\lambda)}
\]

where \(\lambda\) is the average number of events.

## Formula

\[
\boxed{
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}
}
\]

## Example

Average:

\[
\lambda=2
\]

Find probability of exactly 3 events:

\[
P(X=3)=\frac{e^{-2}(2)^3}{3!}
\]

\[
\boxed{P(X=3)=\frac{8e^{-2}}{6}}
\]

Using \(e^{-2}\approx0.1353\):

\[
\boxed{P(X=3)\approx0.1804=18.04\%}
\]

**Memory trick:**

\[
\boxed{\text{Poisson = How many events in an interval?}}
\]

---

# 6. Uniform Distribution

For:

\[
X\sim U(a,b)
\]

all parts of the interval have equal density.

## PDF

\[
\boxed{
f(x)=\frac1{b-a}
}
\]

for:

\[
a\le x\le b
\]

## Easy derivation

The graph is a rectangle.

Total area must be 1:

\[
\text{height}\times\text{width}=1
\]

Thus:

\[
h(b-a)=1
\]

So:

\[
\boxed{h=\frac1{b-a}}
\]

## Example

\[
X\sim U(0,10)
\]

Find:

\[
P(2\le X\le5)
\]

Desired interval length:

\[
5-2=3
\]

Total length:

\[
10-0=10
\]

Therefore:

\[
P(2\le X\le5)=\frac3{10}
\]

\[
\boxed{0.3}
\]

## CDF

\[
\boxed{
F(x)=
\begin{cases}
0,&x<a\\
\dfrac{x-a}{b-a},&a\le x\le b\\
1,&x>b
\end{cases}
}
\]

### Example

For \(X\sim U(0,10)\):

\[
F(4)=\frac4{10}
\]

\[
\boxed{0.4}
\]

### Important

For a continuous variable:

\[
\boxed{P(X=x)=0}
\]

The PDF gives density, not point probability.

---

# 7. Normal Distribution

We write:

\[
\boxed{X\sim N(\mu,\sigma^2)}
\]

where:

\[
\mu=\text{mean}
\]

\[
\sigma^2=\text{variance}
\]

\[
\sigma=\text{standard deviation}
\]

## Meaning

\[
\boxed{\mu=\text{center}}
\]

\[
\boxed{\sigma=\text{spread}}
\]

## 68–95–99.7 rule

Approximately:

\[
\boxed{\mu\pm\sigma\rightarrow68\%}
\]

\[
\boxed{\mu\pm2\sigma\rightarrow95\%}
\]

\[
\boxed{\mu\pm3\sigma\rightarrow99.7\%}
\]

## Example

\[
X\sim N(50,16)
\]

Therefore:

\[
\mu=50
\]

\[
\sigma=\sqrt{16}=4
\]

About 68% lies between:

\[
50-4=46
\]

and:

\[
50+4=54
\]

So:

\[
\boxed{46\le X\le54}
\]

---

# 8. Adding and Subtracting Normal Variables

For independent:

\[
X\sim N(\mu_X,\sigma_X^2)
\]

and:

\[
Y\sim N(\mu_Y,\sigma_Y^2)
\]

## Addition

\[
\boxed{
X+Y\sim N(\mu_X+\mu_Y,\sigma_X^2+\sigma_Y^2)
}
\]

## Subtraction

\[
\boxed{
X-Y\sim N(\mu_X-\mu_Y,\sigma_X^2+\sigma_Y^2)
}
\]

### Important

We add **variances**, not standard deviations.

## Product and division

\[
\boxed{XY\text{ is generally not Normal}}
\]

\[
\boxed{X/Y\text{ is generally not Normal}}
\]

---

# 9. Z-score and Standardization

The Z-score tells us how many standard deviations a value is from the mean.

\[
\boxed{
Z=\frac{X-\mu}{\sigma}
}
\]

## Example

\[
X\sim N(50,16)
\]

So:

\[
\mu=50,\quad\sigma=4
\]

For \(X=46\):

\[
Z=\frac{46-50}{4}
\]

\[
\boxed{Z=-1}
\]

From the Z-table:

\[
P(Z\le-1)=0.1587
\]

Therefore:

\[
\boxed{15.87\%}
\]

---

# 10. Probability Between Two Values

Suppose:

\[
X\sim N(100,25)
\]

Thus:

\[
\mu=100,\quad\sigma=5
\]

Find:

\[
P(100\le X\le105)
\]

Convert:

\[
Z_1=\frac{100-100}{5}=0
\]

\[
Z_2=\frac{105-100}{5}=1
\]

Therefore:

\[
P(0\le Z\le1)
\]

From the Z-table:

\[
P(Z\le1)=0.8413
\]

and:

\[
P(Z\le0)=0.5000
\]

Thus:

\[
0.8413-0.5000
\]

\[
\boxed{0.3413=34.13\%}
\]

---

# 11. Central Limit Theorem (CLT)

Suppose we repeatedly take samples of size \(n\) and calculate:

\[
\bar X=\frac{X_1+X_2+\cdots+X_n}{n}
\]

The CLT says that, under its conditions and for sufficiently large samples, the distribution of sample means becomes approximately Normal:

\[
\boxed{
\bar X\approx N\left(\mu,\frac{\sigma^2}{n}\right)
}
\]

## Very important

The original variable \(X\) does not have to become Normal.

It is the **sample mean** \(\bar X\) whose distribution becomes approximately Normal.

## Mean and standard error

\[
\boxed{\mu_{\bar X}=\mu}
\]

\[
\boxed{
\operatorname{Var}(\bar X)=\frac{\sigma^2}{n}
}
\]

\[
\boxed{
SE=\frac{\sigma}{\sqrt n}
}
\]

As \(n\) increases:

\[
\boxed{SE\text{ decreases}}
\]

---

# 12. CLT Example Using a Z-table

Suppose:

\[
\mu=100,\quad\sigma=20,\quad n=100
\]

Find:

\[
P(\bar X>104)
\]

## Step 1: Standard error

\[
SE=\frac{20}{\sqrt{100}}
\]

\[
\boxed{SE=2}
\]

Thus:

\[
\bar X\approx N(100,4)
\]

## Step 2: Convert to Z

\[
Z=\frac{104-100}{2}
\]

\[
\boxed{Z=2}
\]

## Step 3: Use Z-table

\[
P(Z\le2)=0.9772
\]

Therefore:

\[
P(Z>2)=1-0.9772
\]

\[
\boxed{0.0228=2.28\%}
\]

---

# 13. CLT Example Between Two Values

Using:

\[
\mu=100,\quad SE=2
\]

Find:

\[
P(98\le\bar X\le102)
\]

For 98:

\[
Z_1=\frac{98-100}{2}=-1
\]

For 102:

\[
Z_2=\frac{102-100}{2}=1
\]

Thus:

\[
P(-1\le Z\le1)
\]

From the table:

\[
P(Z\le1)=0.8413
\]

\[
P(Z\le-1)=0.1587
\]

Therefore:

\[
0.8413-0.1587
\]

\[
\boxed{0.6826=68.26\%}
\]

---

# 14. Student's t-Distribution

If population standard deviation \(\sigma\) is known:

\[
\boxed{
Z=\frac{\bar X-\mu}{\sigma/\sqrt n}
}
\]

If \(\sigma\) is unknown, use sample standard deviation \(s\):

\[
\boxed{
t=\frac{\bar X-\mu}{s/\sqrt n}
}
\]

The t-distribution accounts for the additional uncertainty from estimating \(\sigma\) using \(s\).

## Degrees of freedom

For a one-sample problem:

\[
\boxed{df=n-1}
\]

---

# 15. t-Distribution Numerical Example

Given:

\[
n=16,\quad\bar X=55,\quad s=8,\quad\mu=51
\]

## Standard error

\[
SE=\frac8{\sqrt{16}}
\]

\[
\boxed{SE=2}
\]

## t-value

\[
t=\frac{55-51}{2}
\]

\[
\boxed{t=2}
\]

## Degrees of freedom

\[
df=16-1
\]

\[
\boxed{df=15}
\]

---

# 16. How to Use a t-table

A typical t-table gives **critical t-values**.

You need:

1. Degrees of freedom
2. One-tailed or two-tailed test
3. Significance level \(\alpha\)

Example row:

| df | One-tail 0.05 | One-tail 0.025 |
|---|---:|---:|
| 15 | 1.753 | 2.131 |

So:

\[
\boxed{\text{Row}=df}
\]

\[
\boxed{\text{Column}=\text{tail probability}}
\]

---

# 17. One-tailed and Two-tailed Tests

## One-tailed

Use when we care about one specific direction.

### Right-tailed

\[
\boxed{H_1:\mu>\mu_0}
\]

### Left-tailed

\[
\boxed{H_1:\mu<\mu_0}
\]

Memory rule:

\[
\boxed{>\text{ or }<\Rightarrow\text{one-tailed}}
\]

## Two-tailed

Use when we care about either direction.

\[
\boxed{H_1:\mu\neq\mu_0}
\]

Memory rule:

\[
\boxed{\neq\Rightarrow\text{two-tailed}}
\]

---

# 18. Why Is Alpha Split?

Suppose:

\[
\alpha=0.05
\]

## One-tailed

All 5% is in one tail:

\[
\boxed{0.05}
\]

## Two-tailed

Split equally:

\[
\boxed{0.025\text{ left}+0.025\text{ right}}
\]

---

# 19. t-table Example

Suppose:

\[
t_{\text{calculated}}=2,\quad df=15
\]

## One-tailed at \(\alpha=0.05\)

From the table:

\[
t_{\text{critical}}=1.753
\]

Since:

\[
2>1.753
\]

the value is in the rejection region.

## Two-tailed at \(\alpha=0.05\)

Split:

\[
0.05/2=0.025
\]

From the table:

\[
t_{\text{critical}}=2.131
\]

Critical boundaries:

\[
-2.131,\quad+2.131
\]

Since:

\[
|2|<2.131
\]

the value is not in either rejection region.

---

# 20. Master Formula Sheet

## Bernoulli

\[
\boxed{P(X=1)=p,\quad P(X=0)=1-p}
\]

## Binomial

\[
\boxed{P(X=k)=\binom nkp^k(1-p)^{n-k}}
\]

## Geometric

\[
\boxed{P(X=k)=(1-p)^{k-1}p}
\]

## Poisson

\[
\boxed{P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}}
\]

## Uniform

\[
\boxed{f(x)=\frac1{b-a}}
\]

## Z-score

\[
\boxed{Z=\frac{X-\mu}{\sigma}}
\]

## CLT Standard Error

\[
\boxed{SE=\frac{\sigma}{\sqrt n}}
\]

## Z-score for sample mean

\[
\boxed{Z=\frac{\bar X-\mu}{\sigma/\sqrt n}}
\]

## t-statistic

\[
\boxed{t=\frac{\bar X-\mu}{s/\sqrt n}}
\]

## Degrees of freedom

\[
\boxed{df=n-1}
\]

---

# 21. Final Memory Map

\[
\boxed{\text{Bernoulli = ONE trial}}
\]

\[
\boxed{\text{Binomial = HOW MANY successes?}}
\]

\[
\boxed{\text{Geometric = WHEN is the first success?}}
\]

\[
\boxed{\text{Poisson = HOW MANY events in an interval?}}
\]

\[
\boxed{\text{Uniform = equal density}}
\]

\[
\boxed{\text{Normal = bell-shaped distribution}}
\]

\[
\boxed{\text{CLT = sample means become approximately Normal}}
\]

\[
\boxed{\text{Known }\sigma\rightarrow Z}
\]

\[
\boxed{\text{Unknown }\sigma\rightarrow t}
\]

\[
\boxed{>\text{ or }<\rightarrow\text{one-tailed}}
\]

\[
\boxed{\neq\rightarrow\text{two-tailed}}
\]
