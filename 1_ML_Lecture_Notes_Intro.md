# Machine Learning — Lecture Notes

*Introduction to AI & ML (NPTEL, IISc)*

---

## 1. What is Artificial Intelligence (AI)?

**Definition:** Artificial Intelligence involves imparting *human-like cognitive functions* to machines so that they can **make inferences and decisions based on input data**.

**In easy terms:** Your brain constantly perceives, reasons, learns, and decides — these are *cognitive functions*. AI is the effort to copy these mental abilities into machines, so a computer can do "smart" tasks on its own instead of blindly following fixed instructions. The key idea is that the machine acts **based on input data**.

**Simple flow:**

> Input data → AI processes it → Inference / Decision

**Traditional program vs. AI:** A normal program does exactly what a programmer hard-codes. AI is designed to *look at data, figure something out (infer), and choose an action (decide)* — resembling human thinking.

**Examples:**

- **Spam filter** — reads an email, infers spam vs. genuine, decides which folder.
- **Netflix / YouTube recommendations** — infers your taste from history, decides what to suggest.
- **Face unlock** — infers whether the face matches you, decides to unlock or not.
- **Google Maps** — infers the fastest route from traffic data, decides which path to show.
- **Self-driving car** — infers "pedestrian ahead," decides to brake.

---

## 2. What is Machine Learning (ML)?

**Definition:** Machine Learning is the use of computers to **learn** (rather than *memorize*) datasets involving a number of variables, so that they can **make predictions corresponding to unseen data points**.

**In easy terms — learning vs. memorizing:**

- **Memorizing** = a student who crams last year's exact answers. Change the numbers and they're stuck, because they never grasped the pattern.
- **Learning** = a student who understands the concept and can solve *any* new problem of that type.

ML aims for the second kind. It extracts the **underlying pattern** from data and applies it to brand-new situations. Performing well on **unseen data** is called **generalization** — the goal of every good ML model.

**"A number of variables":** Real datasets have many *features*. To predict a house price, the variables might be size, bedrooms, location, and age — ML studies how all of them together relate to the price.

**The Predicted vs. Actual graph:**

- **X-axis (Actual data):** the true real-world values.
- **Y-axis (Predicted data):** what the model guessed.
- **Dashed diagonal line:** the "perfect prediction" line (predicted = actual).

When the points fall close to the dashed line, predictions ≈ real values → a **good model**. Points scattered far from the line → a **poor model**.

**Examples:**

- **House price prediction** — learns from past sales, predicts a new house's price.
- **Weather forecasting** — learns from years of data, predicts tomorrow's weather.
- **Exam score prediction** — from study hours, attendance, past marks, predicts a new student's score.
- **Medical diagnosis** — learns from past labelled scans, predicts for a new patient.

**Key idea:** ML is not about storing answers — it's about **finding the pattern** so the computer can **predict correctly for cases it has never seen.**

---

## 3. Relationship between AI and ML

**Core idea:** **ML is a subset of AI.** Every ML system is a type of AI, but not every AI is ML. (Like *cars* are a subset of *vehicles* — every car is a vehicle, but a vehicle could also be a bike or a plane.)

| | What it does |
|---|---|
| **AI** (outer circle) | Machines make **decisions and/or take actions** |
| **ML** (inner circle) | Machines make **predictions or learn patterns** |

ML is the *learning engine* that often *powers* the larger AI system: ML figures out the prediction, and AI uses it to make a decision or take an action.

**Connected example (self-driving car):**

- **ML** predicts: "That object ahead is a pedestrian." (learn patterns / prediction)
- **AI** decides: "Apply the brakes now." (decision / action)

**The three mathematical foundations (the arrows):**

- **Probability & Statistics** — handles uncertainty and noise; estimates likelihoods (e.g., "90% chance this is spam"). *Helps understand the data.*
- **Linear Algebra** — data is represented as **vectors and matrices**; nearly all ML computation is matrix operations. *Helps represent and compute the data.*
- **Multivariate Optimization** — "learning" means **minimizing error** by tuning model parameters (e.g., gradient descent). *Helps the model actually learn.*

**Key ideas:**

1. **ML ⊂ AI** — ML lives inside AI as a subset.
2. **AI = decisions/actions; ML = predictions/learning patterns.**
3. ML/AI stands on three pillars: **Probability & Statistics, Linear Algebra, Multivariate Optimization.**

---

## 4. Advantages of ML

**1. Mitigates the lack of closed-form expressions or theories.**
A *closed-form expression* is a neat exact formula (e.g., area of a circle `A = πr²`). Many real problems have **no such formula** — there's no tidy equation for a house's sale price or whether a photo has a cat. ML **learns the relationship directly from data**, so it works even when scientists have no theory or equation.
*Example:* No exact equation predicts a handwritten digit or a stock move, but ML handles both by learning from data.

**2. Helps demystify datasets with high dimensionality.**
*Dimensionality* = number of input variables. With **many variables** (hundreds/thousands), humans can't tell how each one affects the outcome. ML sifts through them **all at once**, finds which matter, and uncovers hidden patterns.
*Example:* Predicting disease from age, weight, blood pressure, and hundreds of gene markers — too many for a human, routine for ML.

**3. Once trained, it reduces the computational cost of predictions by several orders of magnitude.**
Training happens **once** (slow, expensive). Afterward, making predictions is **extremely fast and cheap** — 100×, 1000×, or more.
*Example:* A full physics simulation of airflow over a wing may take hours; a trained ML "surrogate" predicts it almost instantly.

**Key ideas:**

1. Works **even with no known formula** — learns from data.
2. Handles **many variables at once** that humans can't untangle.
3. Train once → **predictions are then very fast and cheap.**

---

## 5. Disadvantages of ML

**1. Models typically require a large amount of data to train — which may not be available or may be poor quality.**
ML learns from examples, so it needs **lots of good data**. Problems: data may be **scarce** (rare diseases, new materials) or **poor quality** (noisy, incomplete, mislabelled). Remember: **"Garbage in, garbage out."**
*→ Modern fix:* **Foundation models + fine-tuning.** A foundation model is pre-trained on enormous general data; you *fine-tune* it on your small dataset and still get great results.

**2. May obscure the physics / chemistry / biology of the problem, since all focus is on data.**
A typical ML model is a **"black box"** — accurate answers, but it can't explain *why*. Focusing only on data patterns, it can ignore the real scientific mechanism, giving predictions but **little scientific insight** (and sometimes physically impossible outputs).
*→ Modern fix:* **Physics-informed / physics-inspired ML** builds known scientific laws (e.g., conservation of energy) directly into the model — accurate *and* scientifically consistent.

**Key ideas:**

1. ML is **data-hungry** — needs lots of good data. *(Fix: foundation models + fine-tuning.)*
2. ML is often a **"black box"** hiding the real science. *(Fix: physics-informed ML.)*

---

*End of notes — Introduction to AI & ML.*
