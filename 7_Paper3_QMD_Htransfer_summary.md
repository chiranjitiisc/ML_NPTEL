# Lecture 07 --- Numerical Examples and Detailed Solutions

## Topics Covered

This file contains numerical examples and detailed solutions for:

1.  Population Mean / Expectation
2.  Variance and Standard Deviation
3.  Covariance
4.  Independent Random Variables
5.  Correlation Coefficient
6.  Perfect Linear Relationships

------------------------------------------------------------------------

# Formula Sheet

## 1. Expectation

\[ E(X)=`\sum `{=tex}xP(X=x) \]

## 2. Second Moment

\[ E(X^2)=`\sum `{=tex}x^2P(X=x) \]

## 3. Variance

\[ `\operatorname{Var}`{=tex}(X)=E(X^2)-\[E(X)\]^2 \]

## 4. Standard Deviation

\[ `\sigma`{=tex}=`\sqrt{\operatorname{Var}(X)}`{=tex} \]

## 5. Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)=E(XY)-E(X)E(Y) \]

## 6. Correlation Coefficient

\[ `\rho`{=tex}\_{XY} = `\frac{\operatorname{Cov}(X,Y)}`{=tex}
{`\sigma`{=tex}\_X`\sigma`{=tex}\_Y} =
`\frac{\operatorname{Cov}(X,Y)}`{=tex}
{`\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}`{=tex}} \]

## 7. Independent Random Variables

If (X) and (Y) are independent:

\[ E(XY)=E(X)E(Y) \]

Therefore:

\[ `\operatorname{Cov}`{=tex}(X,Y)=0 \]

## 8. Perfect Linear Relationship

If:

\[ Y=mX+c \]

then:

\[ `\operatorname{Var}`{=tex}(Y)=m\^2`\operatorname{Var}`{=tex}(X) \]

and:

\[ `\operatorname{Cov}`{=tex}(X,Y)=m`\operatorname{Var}`{=tex}(X) \]

------------------------------------------------------------------------

# Part 1 --- Expectation

## Question 1

A discrete random variable (X) has the following distribution:

  \(X\)        1     2     3     4
  -------- ----- ----- ----- -----
  (P(X))     0.1   0.3   0.4   0.2

Find:

\[ E(X) \]

### Solution

Use:

\[ E(X)=`\sum `{=tex}xP(X=x) \]

Therefore:

\[ E(X)=1(0.1)+2(0.3)+3(0.4)+4(0.2) \]

\[ =0.1+0.6+1.2+0.8 \]

\[ `\boxed{E(X)=2.7}`{=tex} \]

------------------------------------------------------------------------

## Question 2

A random variable (X) has the following distribution:

  \(X\)         0      2      4      6
  -------- ------ ------ ------ ------
  (P(X))     0.25   0.15   0.35   0.25

Find:

\[ E(X) \]

### Solution

\[ E(X)=0(0.25)+2(0.15)+4(0.35)+6(0.25) \]

\[ =0+0.3+1.4+1.5 \]

\[ `\boxed{E(X)=3.2}`{=tex} \]

------------------------------------------------------------------------

# Part 2 --- Variance and Standard Deviation

## Question 3

  \(X\)        1     2     3
  -------- ----- ----- -----
  (P(X))     0.2   0.5   0.3

Find:

1.  (E(X))
2.  (E(X\^2))
3.  (`\operatorname{Var}`{=tex}(X))
4.  (`\sigma`{=tex})

### Solution

#### Step 1: Find (E(X))

\[ E(X)=1(0.2)+2(0.5)+3(0.3) \]

\[ =0.2+1+0.9 \]

\[ `\boxed{E(X)=2.1}`{=tex} \]

#### Step 2: Find (E(X\^2))

Square (X) first:

\[ E(X^2)=1^2(0.2)+2^2(0.5)+3^2(0.3) \]

\[ =1(0.2)+4(0.5)+9(0.3) \]

\[ =0.2+2+2.7 \]

\[ `\boxed{E(X^2)=4.9}`{=tex} \]

#### Step 3: Find Variance

\[ `\operatorname{Var}`{=tex}(X)=E(X^2)-\[E(X)\]^2 \]

\[ =4.9-(2.1)\^2 \]

\[ =4.9-4.41 \]

\[ `\boxed{\operatorname{Var}(X)=0.49}`{=tex} \]

#### Step 4: Find Standard Deviation

\[ `\sigma`{=tex}=`\sqrt{0.49}`{=tex} \]

\[ `\boxed{\sigma=0.7}`{=tex} \]

### Final Answer

\[
`\boxed{E(X)=2.1,\quad E(X^2)=4.9,\quad \operatorname{Var}(X)=0.49,\quad \sigma=0.7}`{=tex}
\]

------------------------------------------------------------------------

## Question 4

  \(X\)        0     1     2     3
  -------- ----- ----- ----- -----
  (P(X))     0.1   0.3   0.4   0.2

Find:

1.  (E(X))
2.  (E(X\^2))
3.  (`\operatorname{Var}`{=tex}(X))
4.  (`\sigma`{=tex})

### Solution

#### Step 1: Find (E(X))

\[ E(X)=0(0.1)+1(0.3)+2(0.4)+3(0.2) \]

\[ =0+0.3+0.8+0.6 \]

\[ `\boxed{E(X)=1.7}`{=tex} \]

#### Step 2: Find (E(X\^2))

\[ E(X^2)=0^2(0.1)+1^2(0.3)+2^2(0.4)+3\^2(0.2) \]

\[ =0+0.3+1.6+1.8 \]

\[ `\boxed{E(X^2)=3.7}`{=tex} \]

#### Step 3: Find Variance

\[ `\operatorname{Var}`{=tex}(X)=3.7-(1.7)\^2 \]

\[ =3.7-2.89 \]

\[ `\boxed{\operatorname{Var}(X)=0.81}`{=tex} \]

#### Step 4: Find Standard Deviation

\[ `\sigma`{=tex}=`\sqrt{0.81}`{=tex} \]

\[ `\boxed{\sigma=0.9}`{=tex} \]

------------------------------------------------------------------------

# Part 3 --- Covariance

## Question 5

The joint distribution of (X) and (Y) is:

    \(X\)   \(Y\)   (P(X,Y))
  ------- ------- ----------
        1       2        0.2
        2       4        0.3
        3       6        0.5

Find:

1.  (E(X))
2.  (E(Y))
3.  (E(XY))
4.  (`\operatorname{Cov}`{=tex}(X,Y))

### Solution

#### Step 1: Find (E(X))

\[ E(X)=1(0.2)+2(0.3)+3(0.5) \]

\[ =0.2+0.6+1.5 \]

\[ `\boxed{E(X)=2.3}`{=tex} \]

#### Step 2: Find (E(Y))

\[ E(Y)=2(0.2)+4(0.3)+6(0.5) \]

\[ =0.4+1.2+3 \]

\[ `\boxed{E(Y)=4.6}`{=tex} \]

#### Step 3: Find (E(XY))

    \(X\)   \(Y\)   (XY)   (P(X,Y))
  ------- ------- ------ ----------
        1       2      2        0.2
        2       4      8        0.3
        3       6     18        0.5

\[ E(XY)=2(0.2)+8(0.3)+18(0.5) \]

\[ =0.4+2.4+9 \]

\[ `\boxed{E(XY)=11.8}`{=tex} \]

#### Step 4: Find Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)=E(XY)-E(X)E(Y) \]

\[ =11.8-(2.3)(4.6) \]

\[ =11.8-10.58 \]

\[ `\boxed{\operatorname{Cov}(X,Y)=1.22}`{=tex} \]

Since covariance is positive, (X) and (Y) tend to increase together.

------------------------------------------------------------------------

## Question 6

The joint distribution is:

    \(X\)   \(Y\)   (P(X,Y))
  ------- ------- ----------
        1       6       0.25
        2       4       0.50
        3       2       0.25

Find:

1.  (E(X))
2.  (E(Y))
3.  (E(XY))
4.  (`\operatorname{Cov}`{=tex}(X,Y))

### Solution

#### Step 1: Find (E(X))

\[ E(X)=1(0.25)+2(0.50)+3(0.25) \]

\[ =0.25+1+0.75 \]

\[ `\boxed{E(X)=2}`{=tex} \]

#### Step 2: Find (E(Y))

\[ E(Y)=6(0.25)+4(0.50)+2(0.25) \]

\[ =1.5+2+0.5 \]

\[ `\boxed{E(Y)=4}`{=tex} \]

#### Step 3: Find (E(XY))

    \(X\)   \(Y\)   (XY)   (P(X,Y))
  ------- ------- ------ ----------
        1       6      6       0.25
        2       4      8       0.50
        3       2      6       0.25

\[ E(XY)=6(0.25)+8(0.50)+6(0.25) \]

\[ =1.5+4+1.5 \]

\[ `\boxed{E(XY)=7}`{=tex} \]

#### Step 4: Find Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)=E(XY)-E(X)E(Y) \]

\[ =7-(2)(4) \]

\[ =7-8 \]

\[ `\boxed{\operatorname{Cov}(X,Y)=-1}`{=tex} \]

The negative covariance means that as (X) increases, (Y) tends to
decrease.

------------------------------------------------------------------------

# Part 4 --- Independent Random Variables

## Question 7

Suppose:

\[ E(X)=5,`\qquad `{=tex}E(Y)=10,`\qquad `{=tex}E(XY)=50 \]

Find:

\[ `\operatorname{Cov}`{=tex}(X,Y) \]

Is this consistent with independence?

### Solution

\[ `\operatorname{Cov}`{=tex}(X,Y)=E(XY)-E(X)E(Y) \]

\[ =50-(5)(10) \]

\[ =50-50 \]

\[ `\boxed{\operatorname{Cov}(X,Y)=0}`{=tex} \]

Also:

\[ E(X)E(Y)=5`\times10`{=tex}=50=E(XY) \]

Therefore, this is consistent with independence.

------------------------------------------------------------------------

## Question 8

Two independent random variables have:

\[ E(X)=3,`\qquad `{=tex}E(Y)=4 \]

Calculate:

1.  (E(XY))
2.  (`\operatorname{Cov}`{=tex}(X,Y))

### Solution

Since (X) and (Y) are independent:

\[ E(XY)=E(X)E(Y) \]

Therefore:

\[ E(XY)=(3)(4) \]

\[ `\boxed{E(XY)=12}`{=tex} \]

Now:

\[ `\operatorname{Cov}`{=tex}(X,Y)=E(XY)-E(X)E(Y) \]

\[ =12-(3)(4) \]

\[ `\boxed{\operatorname{Cov}(X,Y)=0}`{=tex} \]

------------------------------------------------------------------------

# Part 5 --- Correlation Coefficient

## Question 9

Given:

\[ `\operatorname{Cov}`{=tex}(X,Y)=12 \]

\[ `\operatorname{Var}`{=tex}(X)=4 \]

\[ `\operatorname{Var}`{=tex}(Y)=36 \]

Find:

\[ `\rho`{=tex}\_{XY} \]

### Solution

\[ `\rho`{=tex}\_{XY} = `\frac{\operatorname{Cov}(X,Y)}`{=tex}
{`\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}`{=tex}} \]

Substitute:

\[ `\rho`{=tex}\_{XY} = `\frac{12}{\sqrt{4\times36}}`{=tex} \]

# \[

```{=tex}
\frac{12}{\sqrt{144}}
```
\]

# \[

```{=tex}
\frac{12}{12}
```
\]

\[ `\boxed{\rho_{XY}=1}`{=tex} \]

This indicates a perfect positive linear relationship.

------------------------------------------------------------------------

## Question 10

Given:

\[ `\operatorname{Cov}`{=tex}(X,Y)=-15 \]

\[ `\sigma`{=tex}\_X=3,`\qquad `{=tex}`\sigma`{=tex}\_Y=5 \]

Find:

\[ `\rho`{=tex}\_{XY} \]

### Solution

\[ `\rho`{=tex}\_{XY} = `\frac{\operatorname{Cov}(X,Y)}`{=tex}
{`\sigma`{=tex}\_X`\sigma`{=tex}\_Y} \]

Substitute:

\[ `\rho`{=tex}\_{XY} = `\frac{-15}{(3)(5)}`{=tex} \]

# \[

```{=tex}
\frac{-15}{15}
```
\]

\[ `\boxed{\rho_{XY}=-1}`{=tex} \]

This indicates a perfect negative linear relationship.

------------------------------------------------------------------------

# Part 6 --- Perfect Linear Relationships

## Question 11

Suppose:

\[ Y=2X+5 \]

and:

\[ `\operatorname{Var}`{=tex}(X)=9 \]

Find:

1.  (`\operatorname{Var}`{=tex}(Y))
2.  (`\operatorname{Cov}`{=tex}(X,Y))
3.  (`\rho`{=tex}\_{XY})

### Solution

Here:

\[ m=2,`\qquad `{=tex}c=5 \]

#### Step 1: Find (`\operatorname{Var}`{=tex}(Y))

\[ `\operatorname{Var}`{=tex}(Y)=m\^2`\operatorname{Var}`{=tex}(X) \]

\[ =(2)\^2(9) \]

\[ `\boxed{\operatorname{Var}(Y)=36}`{=tex} \]

The constant (+5) does not affect variance.

#### Step 2: Find Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)=m`\operatorname{Var}`{=tex}(X) \]

\[ =2(9) \]

\[ `\boxed{\operatorname{Cov}(X,Y)=18}`{=tex} \]

#### Step 3: Find Correlation

\[ `\sigma`{=tex}\_X=`\sqrt{9}`{=tex}=3 \]

\[ `\sigma`{=tex}\_Y=`\sqrt{36}`{=tex}=6 \]

Therefore:

\[ `\rho`{=tex}\_{XY} = `\frac{18}{(3)(6)}`{=tex} \]

# \[

```{=tex}
\frac{18}{18}
```
\]

\[ `\boxed{\rho_{XY}=1}`{=tex} \]

------------------------------------------------------------------------

## Question 12

Suppose:

\[ Y=3X+10 \]

and:

\[ `\operatorname{Var}`{=tex}(X)=16 \]

Find:

1.  (`\operatorname{Var}`{=tex}(Y))
2.  (`\operatorname{Cov}`{=tex}(X,Y))
3.  (`\rho`{=tex}\_{XY})

### Solution

Here:

\[ m=3,`\qquad `{=tex}c=10 \]

#### Step 1: Find (`\operatorname{Var}`{=tex}(Y))

\[ `\operatorname{Var}`{=tex}(Y)=m\^2`\operatorname{Var}`{=tex}(X) \]

\[ =(3)\^2(16) \]

\[ =9(16) \]

\[ `\boxed{\operatorname{Var}(Y)=144}`{=tex} \]

#### Step 2: Find Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)=m`\operatorname{Var}`{=tex}(X) \]

\[ =3(16) \]

\[ `\boxed{\operatorname{Cov}(X,Y)=48}`{=tex} \]

#### Step 3: Find Correlation

\[ `\sigma`{=tex}\_X=`\sqrt{16}`{=tex}=4 \]

\[ `\sigma`{=tex}\_Y=`\sqrt{144}`{=tex}=12 \]

Therefore:

\[ `\rho`{=tex}\_{XY} = `\frac{48}{(4)(12)}`{=tex} \]

# \[

```{=tex}
\frac{48}{48}
```
\]

\[ `\boxed{\rho_{XY}=1}`{=tex} \]

------------------------------------------------------------------------

# Final Summary Table

  -----------------------------------------------------------------------------------------------------------------------------------
  Question                            Final Answer
  ----------------------------------- -----------------------------------------------------------------------------------------------
  Q1                                  (E(X)=2.7)

  Q2                                  (E(X)=3.2)

  Q3                                  (E(X)=2.1, E(X\^2)=4.9, `\operatorname{Var}`{=tex}(X)=0.49, `\sigma=0.7`{=tex})

  Q4                                  (E(X)=1.7, E(X\^2)=3.7, `\operatorname{Var}`{=tex}(X)=0.81, `\sigma=0.9`{=tex})

  Q5                                  (`\operatorname{Cov}`{=tex}(X,Y)=1.22)

  Q6                                  (`\operatorname{Cov}`{=tex}(X,Y)=-1)

  Q7                                  (`\operatorname{Cov}`{=tex}(X,Y)=0)

  Q8                                  (E(XY)=12, `\operatorname{Cov}`{=tex}(X,Y)=0)

  Q9                                  (`\rho`{=tex}\_{XY}=1)

  Q10                                 (`\rho`{=tex}\_{XY}=-1)

  Q11                                 (`\operatorname{Var}`{=tex}(Y)=36, `\operatorname{Cov}`{=tex}(X,Y)=18, `\rho`{=tex}\_{XY}=1)

  Q12                                 (`\operatorname{Var}`{=tex}(Y)=144, `\operatorname{Cov}`{=tex}(X,Y)=48, `\rho`{=tex}\_{XY}=1)
  -----------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Important Concepts to Remember

## 1. Do not confuse (E(X\^2)) and (\[E(X)\]\^2)

They are generally different:

\[ `\boxed{E(X^2)\neq[E(X)]^2}`{=tex} \]

For example, in Question 3:

\[ E(X\^2)=4.9 \]

but:

\[ \[E(X)\]^2=(2.1)^2=4.41 \]

Therefore:

\[ `\operatorname{Var}`{=tex}(X)=4.9-4.41=0.49 \]

## 2. Covariance

\[ `\operatorname{Cov}`{=tex}(X,Y)\>0 \]

means the variables tend to move in the same direction.

\[ `\operatorname{Cov}`{=tex}(X,Y)\<0 \]

means they tend to move in opposite directions.

## 3. Independent Variables

\[ `\text{Independent}`{=tex}
`\Rightarrow `{=tex}`\operatorname{Cov}`{=tex}(X,Y)=0 \]

## 4. Correlation

\[ -1`\leq`{=tex}`\rho`{=tex}\_{XY}`\leq1`{=tex} \]

-   (`\rho=1`{=tex}): perfect positive linear relationship
-   (`\rho=-1`{=tex}): perfect negative linear relationship
-   (`\rho`{=tex}`\approx0`{=tex}): no linear relationship

## 5. Perfect Linear Relationship

For:

\[ Y=mX+c \]

the constant (c) shifts the values but does not change their spread.

The key formulas are:

\[ `\boxed{\operatorname{Var}(Y)=m^2\operatorname{Var}(X)}`{=tex} \]

\[ `\boxed{\operatorname{Cov}(X,Y)=m\operatorname{Var}(X)}`{=tex} \]
