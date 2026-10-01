# Machine Learning — Random Variables, Distributions & Joint Distributions

## Expanded Lecture Notes with Easy Explanations, Derivations & Worked Examples

---

# 1. Bayes' Theorem — Worked Example

## Problem

In an ML class:

- **40%** of students are from a **science** background.
- **60%** are from an **engineering** background.
- Of the science students, **80% have no prior exposure** to ML.
- Of the engineering students, **70% have no prior exposure** to ML.

Find the probability that a student is from a **science background given that they have some exposure to ML**.

---

## Step 1 — Define the events

Let:

\[
S = \text{student is from a science background}
\]

\[
E = \text{student is from an engineering background}
\]

\[
N = \text{student has no prior ML exposure}
\]

\[
X = \text{student has some prior ML exposure}
\]

Given:

\[
P(S)=0.40,\qquad P(E)=0.60
\]

Also:

\[
P(N\mid S)=0.80
\]

\[
P(N\mid E)=0.70
\]

The question asks about students with **some exposure**, not no exposure.

Since \(X\) is the complement of \(N\):

\[
P(X\mid S)=1-P(N\mid S)
\]

Therefore:

\[
P(X\mid S)=1-0.80=0.20
\]

Similarly:

\[
P(X\mid E)=1-P(N\mid E)
\]

\[
P(X\mid E)=1-0.70=0.30
\]

---

## Step 2 — What do we need?

We need:

\[
P(S\mid X)
\]

This means:

> Probability that a student is from science **given that** the student has some ML exposure.

---

## Step 3 — Apply Bayes' theorem

\[
P(S\mid X)
=
\frac{P(X\mid S)P(S)}{P(X)}
\]

There are only two background groups: science and engineering.

Therefore, using total probability:

\[
P(X)
=
P(X\mid S)P(S)+P(X\mid E)P(E)
\]

Substitute this into Bayes' theorem:

\[
\boxed{
P(S\mid X)
=
\frac{P(X\mid S)P(S)}
{P(X\mid S)P(S)+P(X\mid E)P(E)}
}
\]

---

## Step 4 — Substitute the numbers

\[
P(S\mid X)
=
\frac{(0.20)(0.40)}
{(0.20)(0.40)+(0.30)(0.60)}
\]

Calculate the numerator:

\[
(0.20)(0.40)=0.08
\]

Calculate the engineering contribution:

\[
(0.30)(0.60)=0.18
\]

Therefore:

\[
P(S\mid X)
=
\frac{0.08}{0.08+0.18}
=
\frac{0.08}{0.26}
\]

\[
\boxed{P(S\mid X)\approx0.3077}
\]

Thus:

\[
\boxed{P(S\mid X)\approx30.8\%}
\]

### Interpretation

Science students make up 40% of the class initially. However, science students are less likely to have prior ML exposure than engineering students. Therefore, after we learn that a student **does have exposure**, the probability that the student is from science drops to about **30.8%**.

---

# 2. Random Variables

## 2.1 What is a random experiment?

A **random experiment** is an experiment whose outcome is uncertain before the experiment is performed.

Examples:

- Rolling a die
- Tossing a coin
- Measuring the temperature of a system
- Selecting a student randomly

---

## 2.2 What is a random variable?

A **random variable** is a numerical variable whose value is determined by the outcome of a random experiment.

For example, roll a die and define:

\[
X=\text{number obtained on the die}
\]

Then:

\[
X\in\{1,2,3,4,5,6\}
\]

Before rolling the die, we do not know which value \(X\) will take. After rolling it, the value becomes known.

For example, if the die shows 4:

\[
X=4
\]

---

## 2.3 Example — Number of heads

Suppose a coin is tossed three times.

Define:

\[
X=\text{number of heads}
\]

Examples:

| Outcome | Value of \(X\) |
|---|---:|
| HHH | 3 |
| HHT | 2 |
| HTH | 2 |
| HTT | 1 |
| TTT | 0 |

Therefore:

\[
\boxed{X\in\{0,1,2,3\}}
\]

---

# 3. Types of Random Variables

\[
\text{Random Variable}
\longrightarrow
\begin{cases}
\text{Discrete}\\
\text{Continuous}
\end{cases}
\]

---

## 3.1 Discrete random variable

A **discrete random variable** takes separate, countable values.

Examples:

\[
X=\text{number obtained when rolling a die}
\]

Possible values:

\[
1,2,3,4,5,6
\]

Other examples:

- Number of heads in 10 tosses
- Number of students in a class
- Number of Mo atoms in a simulation box
- Number of defects in a crystal

### Simple rule

> **If you count it, it is usually discrete.**

---

## 3.2 Continuous random variable

A **continuous random variable** can take any value within a range.

Examples:

- Temperature
- Pressure
- Bond length
- Particle position
- Height
- Time

For example, a temperature can be:

\[
300\text{ K},\quad300.1\text{ K},\quad300.12345\text{ K}
\]

### Simple rule

> **If you measure it, it is usually continuous.**

---

# 4. Probability Distribution

A probability distribution tells us:

> **What values a random variable can take and how probability is distributed among those values.**

---

## Example — Fair die

Let:

\[
X=\text{number obtained when rolling a fair die}
\]

| \(X\) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Probability | \(1/6\) | \(1/6\) | \(1/6\) | \(1/6\) | \(1/6\) |

The total probability is:

\[
\frac16+\frac16+\frac16+\frac16+\frac16+\frac16=1
\]

Thus:

\[
\boxed{\text{Total probability}=1}
\]

---

## Example — Another discrete distribution

Suppose:

| \(X\) | 0 | 1 | 2 |
|---|---:|---:|---:|
| \(P(X)\) | 0.2 | 0.5 | 0.3 |

Then:

\[
P(X=1)=0.5
\]

Check:

\[
0.2+0.5+0.3=1
\]

Therefore, this is a valid probability distribution.

---

# 5. Probability Density Function (PDF)

For a continuous random variable, there are infinitely many possible values.

Therefore, instead of assigning probability to each individual point, we use a **Probability Density Function**, written as:

\[
\boxed{f(x)}
\]

The pdf describes how probability is **densely distributed around different values of \(x\)**.

---

## 5.1 Probability in a very small interval

For a tiny interval from \(x\) to \(x+dx\):

\[
\boxed{P(x\le X\le x+dx)\approx f(x)\,dx}
\]

The notes write this as:

\[
P(x\le X\le x+dx)=f(x)\,dx
\]

The idea is:

\[
\boxed{\text{Probability}\approx\text{density}\times\text{width}}
\]

---

## Example

Suppose:

\[
f(2)=0.5
\]

and we consider a small interval of width:

\[
dx=0.1
\]

Then approximately:

\[
P(2\le X\le2.1)
\approx f(2)\times0.1
\]

\[
=0.5\times0.1
\]

\[
\boxed{=0.05}
\]

---

# 6. Probability Over an Interval

For a continuous random variable, the probability that \(X\) lies between \(a\) and \(b\) is the area under the pdf:

\[
\boxed{
P(a\le X\le b)=\int_a^b f(x)\,dx
}
\]

### Important idea

> **For one continuous random variable, probability = area under the pdf curve.**

---

## Example

Suppose:

\[
f(x)=0.5,\qquad0\le x\le2
\]

and \(f(x)=0\) elsewhere.

Find:

\[
P(0.5\le X\le1.5)
\]

Using:

\[
P(a\le X\le b)=\int_a^b f(x)\,dx
\]

we get:

\[
P(0.5\le X\le1.5)
=
\int_{0.5}^{1.5}0.5\,dx
\]

Since \(0.5\) is constant:

\[
=0.5(1.5-0.5)
\]

\[
\boxed{=0.5}
\]

---

# 7. Why \(P(X=x)=0\) for a Continuous Random Variable

This is one of the most important concepts.

For a continuous random variable:

\[
\boxed{P(X=x)=0}
\]

For example:

\[
P(X=5)=0
\]

Why?

A single point has zero width.

Since probability corresponds to area:

\[
\text{Area}=\text{height}\times\text{width}
\]

At one exact point:

\[
\text{width}=0
\]

Therefore:

\[
\text{Area}=f(5)\times0=0
\]

Hence:

\[
\boxed{P(X=5)=0}
\]

This does **not** mean that the value 5 is impossible. It means that a single point has zero area.

Therefore, for continuous variables:

\[
P(4<X<6)
=
P(4\le X\le6)
\]

because:

\[
P(X=4)=P(X=6)=0
\]

---

# 8. Cumulative Distribution Function (CDF)

The CDF is written as:

\[
\boxed{F(x)}
\]

and is defined as:

\[
\boxed{F(x)=P(X\le x)}
\]

In simple words:

> **The CDF gives the total probability accumulated up to \(x\).**

Think of it as a running total of probability.

---

# 9. Deriving the CDF from the PDF

We know:

\[
P(a\le X\le b)=\int_a^b f(x)\,dx
\]

To find the probability that:

\[
X\le x
\]

we start from the far left and accumulate probability until \(x\).

Therefore:

\[
P(-\infty<X\le x)
=
\int_{-\infty}^{x}f(x')\,dx'
\]

By definition:

\[
F(x)=P(X\le x)
\]

Therefore:

\[
\boxed{
F(x)=\int_{-\infty}^{x}f(x')\,dx'
}
\]

Here \(x'\) is simply a dummy integration variable. It is used so that the upper limit can remain \(x\).

---

## 9.1 CDF at the limits

At the far left:

\[
\boxed{F(-\infty)=0}
\]

because no probability has yet accumulated.

At the far right:

\[
F(\infty)
=
\int_{-\infty}^{\infty}f(x)\,dx
\]

Since total probability must be 1:

\[
\boxed{F(\infty)=1}
\]

Thus, a CDF always satisfies:

\[
\boxed{0\le F(x)\le1}
\]

---

# 10. Example — Calculating a CDF

Suppose:

\[
f(x)=0.5,\qquad0\le x\le2
\]

Find:

\[
F(1)
\]

By definition:

\[
F(1)=P(X\le1)
\]

Since the pdf exists from 0 to 2:

\[
F(1)=\int_0^1 0.5\,dx
\]

\[
=0.5(1-0)
\]

\[
\boxed{F(1)=0.5}
\]

Therefore:

\[
\boxed{P(X\le1)=0.5}
\]

---

# 11. Finding Interval Probability Using the CDF

We know:

\[
F(b)=P(X\le b)
\]

and:

\[
F(a)=P(X\le a)
\]

To obtain only the probability between \(a\) and \(b\), subtract:

\[
\boxed{
P(a\le X\le b)=F(b)-F(a)
}
\]

---

## Example

Suppose:

\[
F(2)=0.30
\]

and:

\[
F(5)=0.85
\]

Find:

\[
P(2\le X\le5)
\]

Using:

\[
P(2\le X\le5)=F(5)-F(2)
\]

\[
=0.85-0.30
\]

\[
\boxed{=0.55}
\]

Thus, there is a 55% probability that \(X\) lies between 2 and 5.

---

# 12. PDF and CDF Relationship

The CDF is obtained by integrating the PDF:

\[
\boxed{
F(x)=\int_{-\infty}^{x}f(x')\,dx'
}
\]

Therefore:

\[
\boxed{\text{PDF}\xrightarrow{\text{integration}}\text{CDF}}
\]

The reverse relationship is obtained by differentiation:

\[
\boxed{
f(x)=\frac{dF(x)}{dx}
}
\]

Therefore:

\[
\boxed{\text{CDF}\xrightarrow{\text{differentiation}}\text{PDF}}
\]

### Easy analogy

Velocity integrated over time gives displacement:

\[
v(t)\xrightarrow{\text{integrate}}\text{displacement}
\]

Similarly:

\[
f(x)\xrightarrow{\text{integrate}}F(x)
\]

---

# 13. Multiple Random Variables

So far, we considered one random variable:

\[
X
\]

For example:

\[
X=\text{temperature}
\]

Now suppose we also have:

\[
Y=\text{pressure}
\]

A single observation gives a pair:

\[
(X,Y)
\]

For example:

\[
(X,Y)=(300\text{ K},5\text{ bar})
\]

The word **joint** simply means **together**.

---

# 14. Joint Probability Density Function (JPDF)

For two continuous random variables, the joint pdf is written as:

\[
\boxed{f_{XY}(x,y)}
\]

or simply:

\[
f(x,y)
\]

This describes how probability density is distributed over combinations of \(X\) and \(Y\).

For one variable:

> Probability is obtained from area under a curve.

For two variables:

> Probability is obtained from the volume under a probability-density surface over a region.

---

# 15. Probability in a Rectangular Region

Suppose we want:

\[
a\le X\le b
\]

and simultaneously:

\[
c\le Y\le d
\]

Then:

\[
\boxed{
P(a\le X\le b,\;c\le Y\le d)
=
\int_a^b\int_c^d f(x,y)\,dy\,dx
}
\]

We integrate over both variables because the probability is accumulated over a two-dimensional region.

---

## Example

Let:

\[
X=\text{temperature}
\]

and:

\[
Y=\text{pressure}
\]

Find the probability that:

\[
300\le X\le310
\]

and:

\[
4\le Y\le5
\]

If the joint pdf is \(f(x,y)\), then:

\[
\boxed{
P(300\le X\le310,\;4\le Y\le5)
=
\int_{300}^{310}\int_4^5f(x,y)\,dy\,dx
}
\]

---

# 16. Joint CDF

For one random variable:

\[
F_X(x)=P(X\le x)
\]

For two random variables:

\[
\boxed{
F(x,y)=P(X\le x,\;Y\le y)
}
\]

This is called the **Joint Cumulative Distribution Function**.

It means:

> The probability that \(X\) is less than or equal to \(x\) and \(Y\) is less than or equal to \(y\), at the same time.

---

## Example

Let:

\[
X=\text{temperature}
\]

\[
Y=\text{pressure}
\]

Then:

\[
F(300,5)
\]

means:

\[
\boxed{
P(X\le300,\;Y\le5)
}
\]

In words:

> The probability that temperature is less than or equal to 300 and pressure is less than or equal to 5 simultaneously.

---

# 17. Probability from the Joint CDF

Suppose we want the probability inside the rectangle:

\[
a\le X\le b
\]

and:

\[
c\le Y\le d
\]

The result is:

\[
\boxed{
P(a\le X\le b,\;c\le Y\le d)
=
F(b,d)-F(a,d)-F(b,c)+F(a,c)
}
\]

This is the two-dimensional version of:

\[
F(b)-F(a)
\]

The four terms arise because we must remove the probability outside the required rectangle, while adding back the overlap that gets subtracted twice.

---

## Worked Example

Suppose:

\[
F(2,4)=0.30
\]

\[
F(1,4)=0.15
\]

\[
F(2,3)=0.20
\]

\[
F(1,3)=0.10
\]

Find:

\[
P(1\le X\le2,\;3\le Y\le4)
\]

Here:

\[
a=1,\quad b=2,\quad c=3,\quad d=4
\]

Therefore:

\[
P
=
F(2,4)-F(1,4)-F(2,3)+F(1,3)
\]

Substitute:

\[
=0.30-0.15-0.20+0.10
\]

\[
=0.05
\]

Therefore:

\[
\boxed{
P(1\le X\le2,\;3\le Y\le4)=0.05
}
\]

---

# 18. Independent Random Variables

Two random variables are **independent** if knowing the value of one does not provide information about the other.

For independent continuous random variables:

\[
\boxed{
f_{XY}(x,y)=f_X(x)f_Y(y)
}
\]

This means the joint pdf factorizes into the product of the individual pdfs.

---

## Example — Two independent dice

Let:

\[
X=\text{result of die 1}
\]

\[
Y=\text{result of die 2}
\]

For fair dice:

\[
P(X=2)=\frac16
\]

and:

\[
P(Y=5)=\frac16
\]

Because the dice are independent:

\[
P(X=2,Y=5)
=
P(X=2)P(Y=5)
\]

Therefore:

\[
=
\frac16\times\frac16
\]

\[
\boxed{=\frac1{36}}
\]

For discrete variables:

\[
\boxed{
P(X=x,Y=y)=P(X=x)P(Y=y)
}
\]

For continuous variables, the corresponding pdf relationship is:

\[
\boxed{
f_{XY}(x,y)=f_X(x)f_Y(y)
}
\]

---

# 19. Getting the Marginal Distribution from a Joint Distribution

Suppose we have the joint CDF:

\[
F_{XY}(x,y)
\]

If we let \(y\to\infty\), we include all possible values of \(Y\). This leaves only the distribution of \(X\):

\[
F_X(x)=F_{XY}(x,\infty)
\]

Differentiating gives the marginal pdf:

\[
\boxed{
f_X(x)
=
\frac{\partial F_{XY}(x,\infty)}{\partial x}
}
\]

For a one-variable CDF:

\[
\boxed{
f_X(x)=\frac{dF_X(x)}{dx}
}
\]

---

# 20. Complete Formula Sheet

## Bayes' Theorem

\[
\boxed{
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
}
\]

For two mutually exclusive cases \(A\) and \(C\):

\[
\boxed{
P(A\mid B)
=
\frac{P(B\mid A)P(A)}
{P(B\mid A)P(A)+P(B\mid C)P(C)}
}
\]

---

## Random Variable Types

\[
\boxed{\text{Random variable}=
\begin{cases}
\text{Discrete}\\
\text{Continuous}
\end{cases}}
\]

---

## PDF

Small interval:

\[
\boxed{
P(x\le X\le x+dx)\approx f(x)\,dx
}
\]

Interval probability:

\[
\boxed{
P(a\le X\le b)=\int_a^b f(x)\,dx
}
\]

Total probability:

\[
\boxed{
\int_{-\infty}^{\infty}f(x)\,dx=1
}
\]

---

## CDF

Definition:

\[
\boxed{
F(x)=P(X\le x)
}
\]

From the pdf:

\[
\boxed{
F(x)=\int_{-\infty}^{x}f(x')\,dx'
}
\]

Interval probability:

\[
\boxed{
P(a\le X\le b)=F(b)-F(a)
}
\]

PDF from CDF:

\[
\boxed{
f(x)=\frac{dF(x)}{dx}
}
\]

Limits:

\[
\boxed{F(-\infty)=0,\qquad F(\infty)=1}
\]

---

## Joint PDF

\[
\boxed{
P(a\le X\le b,\;c\le Y\le d)
=
\int_a^b\int_c^d f(x,y)\,dy\,dx
}
\]

---

## Joint CDF

\[
\boxed{
F(x,y)=P(X\le x,\;Y\le y)
}
\]

Rectangle probability:

\[
\boxed{
P(a\le X\le b,\;c\le Y\le d)
=
F(b,d)-F(a,d)-F(b,c)+F(a,c)
}
\]

---

## Independence

\[
\boxed{
f_{XY}(x,y)=f_X(x)f_Y(y)
}
\]

---

# 21. Final Concept Map

\[
\boxed{
\text{Random Experiment}
\rightarrow
\text{Random Variable}
}
\]

\[
\boxed{
\text{Random Variable}
\rightarrow
\begin{cases}
\text{Discrete}\\
\text{Continuous}
\end{cases}
}
\]

For a continuous random variable:

\[
\boxed{
\text{PDF }f(x)
\xrightarrow{\text{integrate}}
\text{CDF }F(x)
}
\]

\[
\boxed{
F(x)
\xrightarrow{\text{differentiate}}
f(x)
}
\]

For two variables:

\[
\boxed{
(X,Y)
\rightarrow
\text{Joint PDF }f(x,y)
\rightarrow
\text{Joint CDF }F(x,y)
}
\]

If independent:

\[
\boxed{
f_{XY}(x,y)=f_X(x)f_Y(y)
}
\]

---

# Key Takeaways

1. A **random variable** assigns a numerical value to the outcome of a random experiment.
2. **Discrete** variables are countable; **continuous** variables can take any value in a range.
3. A **probability distribution** describes the possible values and their probabilities.
4. For continuous variables, \(f(x)\) is a **density**, not the probability at a point.
5. Probability over an interval is the **area under the pdf**.
6. The **CDF** is the accumulated probability:

\[
F(x)=P(X\le x)
\]

7. PDF and CDF are related by integration and differentiation.
8. A **joint distribution** describes two random variables together.
9. A joint CDF gives cumulative probability for both variables simultaneously.
10. Independent random variables have a joint pdf equal to the product of their individual pdfs.

---

*End of expanded notes.*
