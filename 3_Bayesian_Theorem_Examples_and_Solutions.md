# Bayesian Theorem — Examples & Step-by-Step Solutions

## Probability and Statistics for AI/ML

---

## 1. Bayes' Theorem — The Basic Idea

Bayes' theorem helps us calculate the probability of a **hypothesis after observing some evidence**.

\[
\boxed{
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
}
\]

### Meaning of the four terms

| Term | Name | Meaning |
|---|---|---|
| \(P(A)\) | Prior | Probability of \(A\) before seeing evidence \(B\) |
| \(P(B|A)\) | Likelihood | Probability of observing \(B\) if \(A\) is true |
| \(P(B)\) | Evidence | Overall probability of observing \(B\) |
| \(P(A|B)\) | Posterior | Updated probability of \(A\) after observing \(B\) |

### Easy way to remember

\[
\boxed{
\text{Posterior}
=
\frac{\text{Likelihood}\times\text{Prior}}
{\text{Evidence}}
}
\]

> **Bayesian thinking:** Start with a prior belief → observe evidence → update the belief → obtain the posterior probability.

---

# Example 1 — Medical Test

## Question

A disease affects **1% of the population**. A diagnostic test correctly gives a positive result for **95% of people who have the disease**. However, the test also gives a positive result for **5% of healthy people**.

If a randomly selected person tests positive, what is the probability that the person actually has the disease?

---

## Solution

### Step 1 — Define the events

Let

\[
D=\text{person has the disease}
\]

\[
P=\text{person tests positive}
\]

We need to find:

\[
\boxed{P(D|P)}
\]

---

### Step 2 — Write the given probabilities

Probability of having the disease:

\[
P(D)=0.01
\]

Therefore, probability of being healthy:

\[
P(\bar D)=1-0.01=0.99
\]

Probability of a positive test if the person has the disease:

\[
P(P|D)=0.95
\]

Probability of a positive test if the person is healthy:

\[
P(P|\bar D)=0.05
\]

---

### Step 3 — Calculate the evidence \(P(P)\)

A positive test can occur in two ways:

1. The person has the disease and tests positive.
2. The person is healthy but tests positive.

Therefore,

\[
P(P)
=
P(P|D)P(D)
+
P(P|\bar D)P(\bar D)
\]

Substituting:

\[
P(P)
=
(0.95)(0.01)+(0.05)(0.99)
\]

\[
P(P)=0.0095+0.0495
\]

\[
\boxed{P(P)=0.059}
\]

---

### Step 4 — Apply Bayes' theorem

\[
P(D|P)
=
\frac{P(P|D)P(D)}
{P(P)}
\]

Therefore,

\[
P(D|P)
=
\frac{(0.95)(0.01)}
{0.059}
\]

\[
P(D|P)=0.161
\]

Hence,

\[
\boxed{P(D|P)=16.1\%}
\]

### Answer

The probability that the person actually has the disease, given that the test is positive, is:

\[
\boxed{16.1\%}
\]

### Important lesson

Do not confuse:

\[
P(P|D)=95\%
\]

with

\[
P(D|P)=16.1\%
\]

They are **not the same probability**.

Bayes' theorem allows us to go from:

\[
P(P|D)
\]

to:

\[
P(D|P)
\]

---

# Example 2 — Spam Email Classification

## Question

Suppose **20% of all emails are spam**. The word **"lottery"** appears in **40% of spam emails** and in **2% of non-spam emails**.

If an email contains the word "lottery", what is the probability that the email is spam?

---

## Solution

### Step 1 — Define the events

Let

\[
S=\text{email is spam}
\]

\[
L=\text{email contains the word "lottery"}
\]

We need:

\[
\boxed{P(S|L)}
\]

---

### Step 2 — Given probabilities

Probability that an email is spam:

\[
P(S)=0.20
\]

Probability that an email is not spam:

\[
P(\bar S)=0.80
\]

Probability that "lottery" appears in spam:

\[
P(L|S)=0.40
\]

Probability that "lottery" appears in non-spam:

\[
P(L|\bar S)=0.02
\]

---

### Step 3 — Calculate \(P(L)\)

The word "lottery" can occur in either a spam or non-spam email:

\[
P(L)
=
P(L|S)P(S)
+
P(L|\bar S)P(\bar S)
\]

Substituting:

\[
P(L)
=
(0.40)(0.20)+(0.02)(0.80)
\]

\[
P(L)=0.08+0.016
\]

\[
\boxed{P(L)=0.096}
\]

---

### Step 4 — Apply Bayes' theorem

\[
P(S|L)
=
\frac{P(L|S)P(S)}
{P(L)}
\]

\[
P(S|L)
=
\frac{(0.40)(0.20)}
{0.096}
\]

\[
P(S|L)=0.8333
\]

Therefore,

\[
\boxed{P(S|L)=83.33\%}
\]

### Answer

The probability that the email is spam, given that it contains the word "lottery", is:

\[
\boxed{83.33\%}
\]

### ML connection

This is closely related to the idea behind **Naive Bayes classification**.

A machine-learning model can use words/features as evidence and calculate probabilities such as:

\[
P(\text{Spam}|\text{observed words})
\]

The class with the higher posterior probability can then be selected.

---

# Example 3 — Medical Diagnosis with Multiple Possible Causes

## Question

A patient has a fever. There are three possible causes:

- Flu: \(40\%\) of the population
- COVID: \(20\%\) of the population
- Other infection: \(40\%\) of the population

The probability of having a fever for each condition is:

\[
P(F|\text{Flu})=0.80
\]

\[
P(F|\text{COVID})=0.90
\]

\[
P(F|\text{Other})=0.30
\]

If the patient has a fever, what is the probability that the patient has COVID?

---

## Solution

### Step 1 — Define the events

Let

\[
A_1=\text{Flu}
\]

\[
A_2=\text{COVID}
\]

\[
A_3=\text{Other infection}
\]

and

\[
F=\text{Fever}
\]

We need:

\[
\boxed{P(A_2|F)}
\]

---

### Step 2 — Given probabilities

\[
P(A_1)=0.40
\]

\[
P(A_2)=0.20
\]

\[
P(A_3)=0.40
\]

and

\[
P(F|A_1)=0.80
\]

\[
P(F|A_2)=0.90
\]

\[
P(F|A_3)=0.30
\]

---

### Step 3 — Calculate the evidence \(P(F)\)

Since Flu, COVID and Other infection are mutually exclusive and exhaustive possibilities, use the **law of total probability**:

\[
P(F)
=
P(A_1)P(F|A_1)
+
P(A_2)P(F|A_2)
+
P(A_3)P(F|A_3)
\]

Substituting:

\[
P(F)
=
(0.40)(0.80)
+
(0.20)(0.90)
+
(0.40)(0.30)
\]

\[
P(F)=0.32+0.18+0.12
\]

\[
\boxed{P(F)=0.62}
\]

---

### Step 4 — Apply Bayes' theorem

For COVID:

\[
P(A_2|F)
=
\frac{P(F|A_2)P(A_2)}
{P(F)}
\]

Therefore,

\[
P(A_2|F)
=
\frac{(0.90)(0.20)}
{0.62}
\]

\[
P(A_2|F)=0.2903
\]

Hence,

\[
\boxed{P(\text{COVID}|\text{Fever})=29.03\%}
\]

### Answer

The probability that the patient has COVID, given that the patient has a fever, is:

\[
\boxed{29.03\%}
\]

### Important lesson

When there are several possible causes of an observed event, the denominator is obtained by adding the contribution from **every possible cause**:

\[
\boxed{
P(B)=\sum_i P(A_i)P(B|A_i)
}
\]

---

# Example 4 — Rain and a Wet Road

## Question

Suppose there is a **30% chance of rain** on a particular day. If it rains, there is a **90% chance that the road will be wet**. If it does not rain, there is still a **10% chance that the road will be wet** because of sprinklers or other reasons.

If you observe that the road is wet, what is the probability that it actually rained?

---

## Solution

### Step 1 — Define the events

Let

\[
R=\text{It rained}
\]

\[
W=\text{Road is wet}
\]

We need:

\[
\boxed{P(R|W)}
\]

---

### Step 2 — Given probabilities

Probability of rain:

\[
P(R)=0.30
\]

Therefore,

\[
P(\bar R)=1-0.30=0.70
\]

Probability of a wet road if it rains:

\[
P(W|R)=0.90
\]

Probability of a wet road if it does not rain:

\[
P(W|\bar R)=0.10
\]

---

### Step 3 — Calculate \(P(W)\)

The road can be wet because it rained or because it did not rain but some other event made it wet:

\[
P(W)
=
P(W|R)P(R)
+
P(W|\bar R)P(\bar R)
\]

Substituting:

\[
P(W)
=
(0.90)(0.30)+(0.10)(0.70)
\]

\[
P(W)=0.27+0.07
\]

\[
\boxed{P(W)=0.34}
\]

---

### Step 4 — Apply Bayes' theorem

\[
P(R|W)
=
\frac{P(W|R)P(R)}
{P(W)}
\]

Therefore,

\[
P(R|W)
=
\frac{(0.90)(0.30)}
{0.34}
\]

\[
P(R|W)=0.7941
\]

Hence,

\[
\boxed{P(R|W)=79.41\%}
\]

### Answer

If the road is wet, there is approximately a:

\[
\boxed{79.41\%}
\]

probability that it rained.

---

# 5. The Most Important Concept — \(P(A|B)\) vs \(P(B|A)\)

This is the most common source of confusion in Bayes' theorem.

Consider:

\[
A=\text{Disease}
\]

\[
B=\text{Positive test}
\]

Then:

### \(P(B|A)\)

\[
P(\text{Positive}|\text{Disease})
\]

means:

> If the person has the disease, how likely is the test to be positive?

This is the **likelihood**.

---

### \(P(A|B)\)

\[
P(\text{Disease}|\text{Positive})
\]

means:

> If the test is positive, how likely is it that the person has the disease?

This is the **posterior**.

---

Therefore:

\[
\boxed{
P(A|B)\neq P(B|A)
}
\]

in general.

Bayes' theorem connects the two:

\[
\boxed{
P(A|B)=
\frac{P(B|A)P(A)}
{P(B)}
}
\]

---

# 6. General Bayes' Theorem for Multiple Events

Suppose there are several mutually exclusive and exhaustive hypotheses:

\[
A_1,A_2,\ldots,A_n
\]

and we observe evidence \(B\).

Then:

\[
\boxed{
P(A_i|B)
=
\frac{P(B|A_i)P(A_i)}
{\sum_j P(B|A_j)P(A_j)}
}
\]

The denominator is:

\[
\boxed{
P(B)=
\sum_j P(B|A_j)P(A_j)
}
\]

### Why?

Because the observed evidence \(B\) could have been produced by **any one of the possible hypotheses**.

---

# 7. A Universal Procedure for Solving Bayes Problems

Whenever you see a Bayesian theorem problem, follow these steps.

## Step 1 — Define the events

For example:

\[
D=\text{Disease}
\]

\[
P=\text{Positive test}
\]

---

## Step 2 — Identify what the question asks

Look carefully at the phrase **"given that"**.

If the question asks:

> What is the probability of disease **given that** the test is positive?

then:

\[
\boxed{P(D|P)}
\]

---

## Step 3 — Identify the prior

Find:

\[
P(A)
\]

This represents what you know **before observing the evidence**.

---

## Step 4 — Identify the likelihood

Find:

\[
P(B|A)
\]

This tells you how likely the observed evidence is if the hypothesis is true.

---

## Step 5 — Calculate the evidence

For two possibilities:

\[
P(B)
=
P(B|A)P(A)
+
P(B|\bar A)P(\bar A)
\]

For multiple possibilities:

\[
P(B)
=
\sum_i P(B|A_i)P(A_i)
\]

---

## Step 6 — Apply Bayes' theorem

\[
\boxed{
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
}
\]

---

# 8. Quick Comparison of the Four Terms

| Symbol | Name | Question it answers |
|---|---|---|
| \(P(A)\) | Prior | How likely is \(A\) before seeing evidence? |
| \(P(B|A)\) | Likelihood | If \(A\) is true, how likely is the evidence \(B\)? |
| \(P(B)\) | Evidence | How likely is the evidence \(B\) overall? |
| \(P(A|B)\) | Posterior | After seeing \(B\), how likely is \(A\)? |

---

# 9. Connection to Machine Learning

Bayesian reasoning is directly relevant to classification.

Suppose a machine-learning model must decide whether an object is a **Cat** or **Dog**.

Let:

\[
C=\text{Cat}
\]

\[
D=\text{Dog}
\]

and let \(X\) represent the observed features of the image.

The model can calculate:

\[
P(C|X)
\]

and

\[
P(D|X)
\]

Using Bayes' theorem:

\[
P(C|X)
=
\frac{P(X|C)P(C)}
{P(X)}
\]

and

\[
P(D|X)
=
\frac{P(X|D)P(D)}
{P(X)}
\]

The model can then compare the posterior probabilities.

For example, if:

\[
P(C|X)=0.85
\]

and

\[
P(D|X)=0.15
\]

the model would classify the image as:

\[
\boxed{\text{Cat}}
\]

This is the basic Bayesian idea behind probabilistic classification methods such as **Naive Bayes**.

---

# 10. Final Summary

Bayes' theorem is fundamentally about **updating probability using evidence**.

\[
\boxed{
\text{Prior}
+
\text{Evidence}
\longrightarrow
\text{Posterior}
}
\]

Mathematically:

\[
\boxed{
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
}
\]

Remember:

- \(P(A)\) → **Prior**
- \(P(B|A)\) → **Likelihood**
- \(P(B)\) → **Evidence**
- \(P(A|B)\) → **Posterior**

### The most important point

> **Bayes' theorem tells us how to update our belief about a hypothesis after observing new evidence.**

And always remember:

\[
\boxed{
P(A|B)\neq P(B|A)
}
\]

Bayes' theorem is what allows us to correctly move from one to the other.
