# Machine Learning — Lecture Notes

## Probability and Statistics (NPTEL, IISc)

---

## 1. Why Probability and Statistics for AI/ML?

AI/ML → **involves the use of large amounts of data** → and to *understand and analyze* that data, we need **Probability and Statistics.**

**In easy terms:** ML lives on data, and data is full of randomness, noise, and uncertainty. Probability gives us the language to describe uncertainty ("how likely is this?"), and statistics gives us the tools to draw conclusions from data. Without them, a machine can't reason sensibly about what the data is telling it.

> **Data (AI/ML)** → needs **Probability & Statistics** → to understand & analyze it.

---

## 2. Two Schools of Statistics

Statistics is broadly split into two approaches — two different *philosophies* of what "probability" means:

| | **Frequentist** | **Bayesian** |
|---|---|---|
| **How probability is evaluated** | From **data collected** by repeating an experiment, regarding a certain hypothesis or event | From **two perspectives** — *prior* to knowing a piece of evidence, and *posterior* to it |
| **Other events** | **No importance** given to other events that may or may not influence the event under consideration | Uses evidence to update beliefs |
| **Foundation** | Repeated experiments / observed frequencies | The celebrated **Bayes' theorem** |

**In easy terms:**

- **Frequentist** = "Probability is just the long-run frequency of an event." You find it by *repeating an experiment many times* and counting. It only cares about the data for *that specific event* — it ignores outside information.
- **Bayesian** = "Probability is a degree of belief that gets *updated* as new evidence arrives." You start with a **prior** belief, then observe evidence, and end with an updated **posterior** belief.

---

## 3. Frequentist Statistics

**Definition:** Given an event **A**, the probability of the event occurring, **P(A)**, is determined by **repeatedly conducting a relevant experiment** and noting down its outcome in each case.

$$P(A) = \frac{n(A)}{n(\Omega)}$$

Where:

- **n(A)** = the number of times event A occurred.
- **n(Ω)** = the number of times the experiment was conducted = the **size of the sample space**.
- **Ω (Omega) = the sample space** = the *set of all possible outcomes*.

**In easy terms:** Just run the experiment lots of times and see how often A happens. The fraction "times A happened ÷ total trials" is the frequentist probability.

**Example — Coin toss:**

A coin is tossed **50 times** and a head is obtained **24 times**.

$$P(\text{Heads}) = \frac{24}{50} = \frac{12}{25}$$

This is the **experimental (observed)** probability.

Compare with the **ideal (theoretical)** probability, where the sample space is {H, T}:

$$P(\text{Head})_{\text{ideal}} = \frac{n(H)}{n(\Omega)} = \frac{1}{n(\{H, T\})} = \frac{1}{2}$$

**Takeaway:** The observed value (12/25 = 0.48) is close to, but not exactly, the ideal 1/2. As you toss the coin *more and more* times, the frequentist estimate gets closer to the true probability. This is the essence of the frequentist view — probability emerges from repeated trials.

---

## 4. Bayesian Statistics

**Definition:** This form of statistics is based on the concept of **conditional probability**, which provides a way to account for probabilities **before (prior to)** and **after (posterior to)** knowing a piece of evidence.

**In easy terms:** Bayesian thinking mirrors how humans actually reason. You have an initial belief, then you see some evidence, and you *update* your belief. It's all about learning from new information.

### Conditional Probability & Bayes' Theorem

Given two events **A** and **B**, the **conditional probability** of event A occurring, *provided event B has occurred*, is:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

This is the celebrated **Bayes' Theorem.**

### The four terms explained

- **P(A) — Prior Probability:** the probability of event A occurring *prior to knowing* whether event B has occurred or not. Here event A is treated as a **proposition (a hypothesis)** whose likelihood we are estimating, given some **evidence** represented by event B.

- **P(B) — Evidential Probability:** the probability of the evidential event B (the evidence itself).

- **P(A | B) — Posterior Probability:** the probability of the hypothesis / proposition A **after (post)** the occurrence of the evidential event B.

- **P(B | A) — Likelihood Function:** *how likely* the evidence B is, given that proposition A has occurred.

### The intuitive form

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

- $P(A\mid B)$ → **Posterior probability**
- $P(B\mid A)$ → **Likelihood function**
- $P(A)$ → **Prior probability**
- $P(B)$ → **Evidential probability**

Which gives the famous, easy-to-remember phrasing:

$$\boxed{\text{POSTERIOR} = \frac{\text{LIKELIHOOD} \times \text{PRIOR}}{\text{EVIDENCE}}}$$

**In easy terms:** Start with what you believed (**prior**), weigh it by how well the new evidence fits your hypothesis (**likelihood**), scale by how common that evidence is overall (**evidence**), and you get your updated belief (**posterior**).

---

## 5. Derivation of Bayes' Theorem

Bayes' theorem comes directly from the definition of joint probability. The probability of both A **and** B happening can be written two ways:

$$P(A \cap B) = P(B \cap A)$$

$$P(A)\cdot P(B \mid A) = P(B)\cdot P(A \mid B)$$

Rearranging for $P(A\mid B)$:

$$\Rightarrow P(A \mid B) = \frac{P(A)\cdot P(B \mid A)}{P(B)}$$

That's Bayes' theorem — it's simply a rearrangement of how joint probability is defined. Neat and powerful.

---

## 6. Application of Bayes' Theorem to Multiple Events

If there are several **mutually exclusive and exhaustive** events $A_i$ that can occur following the evidential event B, then:

**Total probability of the evidence B** (Law of Total Probability):

$$P(B) = \sum_i P(A_i \cap B) = \sum_i P(A_i)\cdot P(B \mid A_i)$$

**Bayes' theorem for the specific event $A_i$:**

$$P(A_i \mid B) = \frac{P(A_i)\cdot P(B \mid A_i)}{P(B)} = \frac{P(A_i)\cdot P(B \mid A_i)}{\sum_j P(A_j)\cdot P(B \mid A_j)}$$

**What "mutually exclusive and exhaustive" means:**

- **Mutually exclusive** = no two of the events can happen at the same time (they don't overlap).
- **Exhaustive** = together they cover *all* possibilities (one of them must happen).

**In easy terms:** When there are many possible "causes" ($A_1, A_2, \dots$) that could explain an observed evidence B, this formula lets you find the probability of *each specific cause* given that you saw B. The denominator just adds up the evidence-probability over *all* possible causes, so everything scales to a proper probability.

*Example:* A patient tests positive (evidence B). There are several possible conditions ($A_i$) that could cause a positive test. This formula tells you the probability the patient has a *specific* condition given the positive test — the foundation of medical diagnosis, spam filtering, and much of ML classification.

---

## Quick Recap

1. **AI/ML needs Probability & Statistics** to make sense of large, noisy datasets.
2. **Frequentist** = probability from repeated experiments: $P(A) = n(A)/n(\Omega)$.
3. **Bayesian** = probability as an *updatable belief* using evidence.
4. **Bayes' theorem:** $P(A\mid B) = \dfrac{P(B\mid A)P(A)}{P(B)}$, i.e. **Posterior = (Likelihood × Prior) / Evidence.**
5. Four terms: **Prior** P(A), **Evidence** P(B), **Likelihood** P(B|A), **Posterior** P(A|B).
6. **Multiple events:** use the law of total probability in the denominator to handle many mutually exclusive, exhaustive causes.

---

*End of notes — Probability and Statistics for AI/ML.*
