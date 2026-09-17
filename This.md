# DEEP LEARNING — LAST-MINUTE REVISION

## MODULE 0 — Regression, Cost Functions & Gradient Descent

### 1. Core Concepts

**Linear Regression**

* Predicts a **continuous, unbounded** value.
* One feature:

  $$
  h_\theta(x)=\theta_0+\theta_1x
  $$
* Multiple features:

  $$
  \hat y=w_0+\sum_{m=1}^{M}w_mx_m
  $$
* Matrix form:

  $$
  \hat Y=Xw
  $$

**MSE Cost**

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

* Linear regression cost surface is **convex**.
* Therefore: **one global minimum**.

**Gradient Descent**

$$
\theta_j:=\theta_j-\alpha\frac{\partial J}{\partial\theta_j}
$$

* Moves **opposite to the gradient**.
* \(\alpha\) = learning rate.
* **All gradients must be computed using the old parameter values, then all parameters updated simultaneously.**

### Learning Rate

| Learning rate | Effect                                |
| ------------- | ------------------------------------- |
| Too small     | Very slow convergence                 |
| Appropriate   | Stable convergence                    |
| Too large     | Overshooting, oscillation, divergence |

**Important:** Near the minimum, the gradient naturally becomes smaller. This is **not the same thing as deliberate learning-rate decay**.

---

### Logistic Regression

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

$$
z=\theta_0+\sum_m\theta_mx_m
$$

$$
\hat y=\sigma(z)=P(y=1\mid x;\theta)
$$

**Decision rule**

* \(\hat y>0.5\Rightarrow y=1\)
* Otherwise \(y=0\)
* Since \(\sigma(z)=0.5\) at \(z=0\):

  $$
  \boxed{z=0}
  $$

  is the decision boundary.

Thus plain logistic regression produces a **linear decision boundary**.

### BCE

$$
J(\theta)=
-\frac1m\sum_i
[y^{(i)}\log(h_\theta(x^{(i)}))
+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]
$$

**Why BCE, not MSE?**

* Sigmoid + MSE → **non-convex**
* Sigmoid + BCE → **convex**
* Therefore BCE is the appropriate logistic-regression loss.

### High-Yield Traps

* **Convex logistic-regression loss = BCE, not MSE.**
* Gradient-descent update can look algebraically similar for linear and logistic regression, but the **hypothesis and cost are different**.
* Convexity does **not** mean GD cannot operate on non-convex functions. It means a local minimum is guaranteed to be global.

---

# MODULE 1 — Neural Network Architecture & Parameters

## Core Equations

For one neuron:

$$
z=\sum_iw_ix_i+b
$$

$$
a=g(z)
$$

For layer \(l\):

$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
$$

$$
a^{[l]}=g(z^{[l]})
$$

**Layer \(l\) receives \(a^{[l-1]}\), NOT raw \(x\).**

---

## Dimensions — VERY HIGH YIELD

Let:

* \(n^{[l]}\) = neurons in current layer
* \(n^{[l-1]}\) = neurons in previous layer

Then:

$$
\boxed{W^{[l]}:(n^{[l]},n^{[l-1]})}
$$

$$
\boxed{b^{[l]}:(n^{[l]},1)}
$$

For \(m\) examples:

$$
a^{[l]},z^{[l]}:(n^{[l]},m)
$$

### Remember

> **Weight matrix = current layer × previous layer**

or

> **W = output size × input size**

---

## Number of Parameters

For layer \(l\):

$$
\boxed{\text{parameters}=n^{[l]}n^{[l-1]}+n^{[l]}}
$$

Total:

$$
\boxed{\sum_l(n^{[l]}n^{[l-1]}+n^{[l]})}
$$

**Input layer has no learnable parameters.**

### Example

Network:

$$
5\rightarrow4\rightarrow2
$$

Layer 1:

$$
4(5)+4=24
$$

Layer 2:

$$
2(4)+2=10
$$

Total:

$$
\boxed{34}
$$

---

## One-Hot Encoding

For \(C\) mutually exclusive classes:

* Output layer → **\(C\) neurons**
* Target → length-\(C\) one-hot vector
* Exactly **one 1**, rest 0.

Reason:

* Class labels are categorical, not ordinal.
* Matches softmax probability-vector output.

---

## Train / Validation / Test

| Dataset              | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| **Training**         | Learn weights/biases                                |
| **Validation / Dev** | Tune hyperparameters and make development decisions |
| **Test**             | Final unbiased evaluation                           |

**Test set should be touched only at the end.**

### Critical rule

* Dev and test should come from the **same distribution**.
* Tuning using test data causes **data leakage / optimistic evaluation**.
* No separate test set can be acceptable in constrained settings, but **never train and finally evaluate on the same data**.

### Traps

* \(W\) shape is **not** \((n^{[l-1]},n^{[l]})\).
* Do not forget biases.
* Do not count input-layer neurons as learnable parameters.
* Do not use the test set for hyperparameter tuning.

---

# MODULE 2 — Activation Functions & Gradient Dynamics

## Why Activation Functions?

Without nonlinear activations:

> **A deep stack of linear layers collapses into one linear transformation.**

Therefore activation functions provide the **non-linearity** required for complex decision boundaries.

---

## Activation Comparison

| Function       | Formula                           | Range                | Zero-centered? | Main issue                      |
| -------------- | --------------------------------- | -------------------- | -------------- | ------------------------------- |
| **Sigmoid**    | \(\frac1{1+e^{-z}}\)              | \((0,1)\)            | ❌              | Saturation / vanishing gradient |
| **tanh**       | \(\frac{e^z-e^{-z}}{e^z+e^{-z}}\) | \((-1,1)\)           | ✅              | Saturation / vanishing gradient |
| **ReLU**       | \(\max(0,z)\)                     | \([0,\infty)\)       | ❌              | Dying ReLU                      |
| **Leaky ReLU** | \(\max(\alpha z,z)\)              | \((-\infty,\infty)\) | ❌              | Reduced dying-ReLU problem      |

---

## Sigmoid

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

### Must memorize

$$
\boxed{\max \sigma'(z)=0.25}
$$

Occurs at:

$$
z=0,\quad \sigma(0)=0.5
$$

$$
0.5(1-0.5)=0.25
$$

As \(|z|\) becomes large:

$$
\sigma'(z)\rightarrow0
$$

→ **saturation → vanishing gradients**

---

## tanh

$$
\tanh(z)\in(-1,1)
$$

* **Zero-centered**
* Better optimization behavior than sigmoid in hidden layers.
* Still saturates.
* Therefore **tanh does NOT solve vanishing gradients completely.**

Relationship:

$$
\boxed{\tanh(z)=2\sigma(2z)-1}
$$

---

## ReLU

$$
\text{ReLU}(z)=\max(0,z)
$$

Derivative:

$$
\text{ReLU}'(z)=
\begin{cases}
1,&z>0\\
0,&z<0
\end{cases}
$$

* Positive side does **not saturate**.
* Cheap computationally.
* Usually preferred for hidden layers.

### Dying ReLU

If neuron remains at \(z<0\):

* Gradient = 0
* Weight cannot recover
* Neuron becomes effectively **dead**

### Fixes

* **Leaky ReLU**
* **ELU**

---

## Vanishing Gradient

During backpropagation, gradients are products of many derivatives.

If many factors are \(<1\):

$$
\text{gradient}\rightarrow0
$$

As depth increases, this can become exponentially small.

**Common causes:**

* Sigmoid
* tanh
* Poor initialization

**Opposite problem:** large gradients → **exploding gradients**.

---

## High-Yield Traps

* Sigmoid derivative maximum = **0.25**.
* ReLU derivative for \(z<0\) = **0**, not \(-1\).
* ReLU at \(z=0\) is technically undefined/subgradient.
* **Zero-centered ≠ non-saturating.**
* tanh is zero-centered but still suffers saturation.
* ReLU avoids saturation only on its **positive side**.
* Leaky ReLU primarily addresses **dying ReLU**.

---

# MODULE 3 — Optimization Algorithms

## Batch-Size Spectrum

| Method            | Batch size | Main property                 |
| ----------------- | ---------: | ----------------------------- |
| **Batch GD**      |      \(m\) | Accurate/stable but expensive |
| **SGD**           |      \(1\) | Fast/noisy                    |
| **Mini-batch GD** |  \(1<b<m\) | Practical default             |

One complete pass through all mini-batches:

$$
\boxed{1\text{ epoch}}
$$

Typical practical batch sizes:

$$
64,128,256,512,\ldots
$$

Usually powers of 2 for hardware efficiency, **not a mathematical requirement**.

---

# Exponentially Weighted Average (EWA)

$$
\boxed{v_t=\beta v_{t-1}+(1-\beta)\theta_t}
$$

Approximate memory length:

$$
\boxed{\frac1{1-\beta}}
$$

Examples:

* \(\beta=0.9\) → ~10 values
* \(\beta=0.98\) → ~50
* \(\beta=0.5\) → ~2

### Bias Correction

$$
\boxed{v_t^{corrected}=\frac{v_t}{1-\beta^t}}
$$

Important mainly during the **early iterations**.

---

# Momentum

$$
v_{dW}=\beta v_{dW}+(1-\beta)dW
$$

$$
W:=W-\alpha v_{dW}
$$

Usually:

$$
\boxed{\beta=0.9}
$$

### Purpose

* Smooth noisy gradients
* Reduce oscillations
* Accelerate movement in consistent directions

**Think:** momentum = **average past gradients**.

---

# RMSprop

$$
S_{dW}=\beta S_{dW}+(1-\beta)(dW)^2
$$

$$
\boxed{
W:=W-\alpha\frac{dW}{\sqrt{S_{dW}}+\epsilon}
}
$$

Purpose:

* Adapt step size **per parameter**
* Large historical gradients → smaller effective step
* Small historical gradients → larger effective step

$$
\epsilon\approx10^{-8}
$$

**Think:** RMSprop = **square gradients + divide by their scale**.

---

# Adam

> **Adam = Momentum + RMSprop**

First moment:

$$
V_{dW}=\beta_1V_{dW}+(1-\beta_1)dW
$$

Second moment:

$$
S_{dW}=\beta_2S_{dW}+(1-\beta_2)(dW)^2
$$

Bias correction:

$$
V^{corr}=\frac{V}{1-\beta_1^t}
$$

$$
S^{corr}=\frac{S}{1-\beta_2^t}
$$

Update:

$$
\boxed{
W:=W-\alpha
\frac{V^{corr}}
{\sqrt{S^{corr}}+\epsilon}
}
$$

### MEMORIZE

$$
\boxed{\beta_1=0.9}
$$

$$
\boxed{\beta_2=0.999}
$$

$$
\boxed{\epsilon=10^{-8}}
$$

* \(\beta_1\) → momentum / first moment
* \(\beta_2\) → squared-gradient / second moment

**Adam does NOT remove the need to tune \(\alpha\).**

---

# Learning Rate Decay

Main forms:

$$
\boxed{
\alpha=
\frac{\alpha_0}{1+\text{decay\_rate}\times\text{epoch}}
}
$$

$$
\boxed{
\alpha=0.95^{epoch}\alpha_0
}
$$

$$
\boxed{
\alpha=\frac{k}{\sqrt{epoch}}\alpha_0
}
$$

Can also use discrete drops after selected epochs.

---

# Weight Initialization

For \(n\) inputs:

$$
\boxed{
\operatorname{Var}(a)=n\operatorname{Var}(w)\operatorname{Var}(x)
}
$$

Stable condition:

$$
n\operatorname{Var}(w)=1
$$

Therefore:

$$
\boxed{\operatorname{Var}(w)=\frac1n}
$$

Basic scaling:

$$
W\sim N(0,1)\times\sqrt{\frac1{n^{[l-1]}}}
$$

Xavier/Bengio variant:

$$
\boxed{
W\sim N(0,1)
\times
\sqrt{
\frac{2}{n^{[l-1]}+n^{[l]}}
}
}
$$

**\(n^{[l-1]}\) = fan-in.**

---

# Gradient Clipping

* Cap gradient magnitude before updating.
* Directly prevents **exploding gradients**.

---

# Forward vs Backward vs Gradient Descent

### Sequence

1. **Forward propagation**

   * Uses current \(W,b\)
   * Computes prediction

2. **Cost**

   * Measures error

3. **Backward propagation**

   * Computes \(dW,db\)
   * Uses chain rule

4. **Gradient descent / optimizer**

   * Actually updates \(W,b\)

### One-line memory trick

> **Forward reads → Backward computes → Optimizer updates.**

---

# MODULE 4 — Regularization & Model Tuning

## Bias vs Variance

| Train error |    Dev error | Diagnosis                       |
| ----------: | -----------: | ------------------------------- |
|         Low |         High | **High variance / overfitting** |
|        High | Similar high | **High bias / underfitting**    |
|        High |  Much higher | **High bias + high variance**   |
|         Low |          Low | Good generalization             |

### Core idea

* **Bias** → model cannot fit training data well.
* **Variance** → model fits training data well but fails on unseen data.
* Train error approximates **bias**.
* Train–dev gap indicates **variance**.

---

## Remedies

### High Bias

Increase capacity:

* Bigger network
* More training
* Better architecture

### High Variance

Constrain capacity / increase effective data:

* More data
* Regularization
* Dropout
* Better architecture

**Trap:** Do not heavily regularize a model already suffering from high bias.

---

# Model Capacity

More parameters:

$$
\Rightarrow \text{higher capacity}
$$

Too little capacity:
→ underfitting

Too much:
→ overfitting

Ideal:
→ low train error + low dev error + small gap

---

# Early Stopping

During training:

* Training loss usually keeps decreasing.
* Validation loss decreases initially.
* Validation loss eventually starts increasing.

**Stop at:**

$$
\boxed{\text{minimum validation error}}
$$

Keep that model checkpoint.

**Not minimum training error.**

---

# L2 Regularization / Weight Decay

Cost:

$$
\boxed{
J=
\frac1m\sum_iLoss_i
+
\frac{\lambda}{2m}\sum_l\|W^{[l]}\|_F^2
}
$$

For a matrix:

$$
\boxed{
\|W\|_F^2=\sum_i\sum_jW_{ij}^2
}
$$

Gradient contribution:

$$
\boxed{
\frac{\lambda}{m}W
}
$$

Update:

$$
W:=
W-\alpha
\left[
\text{backprop term}
+
\frac{\lambda}{m}W
\right]
$$

Equivalent:

$$
\boxed{
W=
\left(1-\frac{\lambda\alpha}{m}\right)W
-\alpha(\text{backprop term})
}
$$

Hence **weight decay**.

### Why it reduces complexity

$$
\lambda\uparrow
\Rightarrow W\downarrow
\Rightarrow z=Wx+b\downarrow
$$

With activations such as tanh, smaller \(z\) keeps neurons closer to the near-linear region, reducing effective model complexity.

---

# L1 vs L2

| L1                                | L2                                          |   |                |
| --------------------------------- | ------------------------------------------- | - | -------------- |
| (\sum                             | w_j                                         | ) | \(\sum w_j^2\) |
| Encourages **sparsity**           | Encourages small weights                    |   |                |
| Many weights can become exactly 0 | Usually weights become small, not exactly 0 |   |                |
| Feature-selection effect          | Commonly called **weight decay**            |   |                |

**MEMORIZE:**

$$
\boxed{\text{L1}\rightarrow\text{sparse}}
$$

$$
\boxed{\text{L2}\rightarrow\text{weight decay}}
$$

---

# Dropout

During **training**:

* Randomly deactivate neurons.

If keep probability = \(k\):

$$
a:=\frac{a\odot d}{k}
$$

This is **inverted dropout**.

At test time:

* Use the **full network**.
* No random dropping.

### Purpose

Prevents excessive co-adaptation and reduces overfitting.

**Trap:** Dropout is NOT applied at inference.

---

# Data Augmentation / Noise

* Inject noise or transform data.
* Makes model more robust.
* Examples: image rotations, flips, crops.
* Can be viewed as a form of regularization.

### Major traps

* Regularization generally **increases training error slightly**.
* It aims to improve **generalization**, not make training error smaller.
* Too large \(\lambda\) → underfitting.
* Too small \(\lambda\) → little effect.

---

# MODULE 5 — Normalization & Initialization

## Input Normalization

Per feature:

### Mean

$$
\mu=\frac1m\sum_iX^{(i)}
$$

Center:

$$
X:=X-\mu
$$

### Standardization

The intended z-score result is:

$$
\boxed{\text{mean}=0,\quad\text{standard deviation}=1}
$$

The syllabus emphasizes using the **training-set statistics** for both training and test data.

### Critical rule

> Compute normalization statistics from the **training set only** and reuse those same statistics on test data.

Do **not** recompute them from the test set.

### Why normalization helps

Unnormalized features → elongated cost surface → zig-zagging / slow GD.

Normalized features → more symmetric cost surface → faster convergence.

---

# Batch Normalization

Applied **inside the network**.

For mini-batch \(Z\):

### Step 1 — Mean

$$
\mu=\frac1m\sum_i z^{(i)}
$$

### Step 2 — Variance

$$
\sigma^2=
\frac1m
\sum_i(z^{(i)}-\mu)^2
$$

### Step 3 — Normalize

$$
\boxed{
z_{norm}^{(i)}
=
\frac{z^{(i)}-\mu}
{\sqrt{\sigma^2+\epsilon}}
}
$$

### Step 4 — Learnable scale and shift

$$
\boxed{
\tilde z^{(i)}
=
\gamma z_{norm}^{(i)}+\beta
}
$$

* \(\gamma\) = learnable scale
* \(\beta\) = learnable shift

Both are learned through backpropagation.

---

## Batch Norm & Bias

Without BN:

$$
Z=WA+b
$$

Because batch normalization subtracts the batch mean and \(b\) is constant across examples:

$$
b\rightarrow \text{cancelled by mean subtraction}
$$

Therefore:

$$
\boxed{\text{Batch Norm makes }b\text{ redundant}}
$$

The shift is handled by \(\beta\).

### Learnable parameters with BN

$$
\boxed{W,\gamma,\beta}
$$

instead of:

$$
W,b
$$

---

## Why Batch Norm?

Three points to remember:

1. Makes optimization easier by normalizing internal layer values.
2. Stabilizes changing activation distributions during training.
3. Has a **mild regularization side effect** because mini-batch statistics are noisy.

**Important:** Regularization is a **side effect**, not the main purpose.

### Batch size trap

Larger mini-batch:

$$
\Rightarrow \text{less statistical noise}
\Rightarrow \text{weaker BN regularization side effect}
$$

---

## Batch Norm at Test Time

You normally cannot calculate meaningful batch statistics from one test example.

Therefore use:

$$
\boxed{\text{running / exponentially weighted estimates}}
$$

of mean and variance collected during training.

---

# Softmax

For \(C\) classes:

$$
\boxed{
A_j=
\frac{e^{z_j}}
{\sum_{k=1}^{C}e^{z_k}}
}
$$

Properties:

* Each output \(\ge0\)
* Outputs sum to:

$$
\boxed{1}
$$

* Outputs are **coupled**.
* Increasing one class probability decreases others.

This makes softmax appropriate for **mutually exclusive classes**.

---

# MODULE 6 — Loss Functions & Classification

# THE MOST IMPORTANT TABLE

| Problem                             | Output activation | Loss                          |
| ----------------------------------- | ----------------- | ----------------------------- |
| **Regression**                      | Linear            | **MSE**                       |
| **Binary classification**           | Sigmoid           | **BCE**                       |
| **Multi-class, mutually exclusive** | Softmax           | **Categorical Cross-Entropy** |
| **Multilabel, independent**         | Multiple Sigmoids | **BCE**                       |

---

## MSE

$$
\boxed{
J=\frac1m\sum_i(\hat y^{(i)}-y^{(i)})^2
}
$$

Use for:

* Continuous
* Unbounded targets

Output:

$$
\boxed{\text{Linear}}
$$

---

# Binary Cross-Entropy

$$
\boxed{
L(\hat y,y)=
-[y\log\hat y+(1-y)\log(1-\hat y)]
}
$$

Use:

* Binary classification
* Multilabel classification

Output:

$$
\boxed{\text{Sigmoid}}
$$

---

# Categorical Cross-Entropy

$$
\boxed{
L(y,\hat y)
=
-\sum_{j=0}^{C-1}y_j\log\hat y_j
}
$$

For one-hot targets:

$$
\boxed{
L=-\log(\hat y_{\text{true class}})
}
$$

Use:

* Multi-class
* Classes are **mutually exclusive**
* Softmax output

If the true class gets probability near 1:

$$
L\rightarrow0
$$

If probability approaches 0:

$$
L\rightarrow\infty
$$

---

# Sigmoid vs Softmax

| Sigmoid                         | Softmax                                 |
| ------------------------------- | --------------------------------------- |
| Outputs independent             | Outputs compete                         |
| Each output in \((0,1)\)        | Outputs sum to 1                        |
| Good for binary / multilabel    | Good for mutually exclusive multi-class |
| Multiple labels can all be high | Raising one lowers others               |

### The conceptual question to ask

**Can multiple labels be true simultaneously?**

* **No** → Softmax + CCE
* **Yes** → Multiple Sigmoids + BCE

---

# EXAM TRAPS — MASTER LIST

### Regression vs Classification

* Regression → **Linear + MSE**
* Binary → **Sigmoid + BCE**
* Multi-class → **Softmax + CCE**
* Multilabel → **Sigmoid + BCE**

### Dimensions

* \(W^{[l]}\) = **current × previous**
* \(b^{[l]}\) = **current × 1**
* Parameters:

  $$
  n^{[l]}n^{[l-1]}+n^{[l]}
  $$

### Dataset Split

* Training → parameters
* Validation → hyperparameters
* Test → final evaluation

### Activations

* Sigmoid max derivative → **0.25**
* tanh → **zero-centered but saturates**
* ReLU → **derivative 1 for positive \(z\)**
* ReLU negative region → **0 gradient**
* Leaky ReLU → fixes **dying ReLU**

### Optimization

* Batch GD → batch size \(m\)
* SGD → batch size 1
* Mini-batch → practical default
* Momentum → average **gradients**
* RMSprop → average **squared gradients**
* Adam → **both**
* Adam:

  $$
  \beta_1=0.9,\quad\beta_2=0.999,\quad\epsilon=10^{-8}
  $$

### Regularization

* High bias → **increase capacity**
* High variance → **regularize / more data**
* L1 → **sparse**
* L2 → **weight decay**
* Dropout → **training only**
* Early stopping → minimum **validation** error

### Normalization

* Training statistics normalize both train and test.
* BN:

  $$
  z\rightarrow normalize\rightarrow\gamma,\beta
  $$
* BN makes bias \(b\) redundant.
* BN test time → running mean/variance.

---

# FORMULA SHEET — MEMORIZE THIS

### Linear Regression

$$
\boxed{\hat y=w_0+\sum_mw_mx_m}
$$

$$
\boxed{
J=\frac{1}{2m}\sum_i(\hat y_i-y_i)^2
}
$$

### Gradient Descent

$$
\boxed{
\theta:=\theta-\alpha\nabla J
}
$$

### Sigmoid

$$
\boxed{
\sigma(z)=\frac1{1+e^{-z}}
}
$$

$$
\boxed{
\sigma'(z)=\sigma(z)(1-\sigma(z))
}
$$

$$
\boxed{\max\sigma'(z)=0.25}
$$

### Neuron

$$
\boxed{z=Wx+b}
$$

$$
\boxed{a=g(z)}
$$

### Layer

$$
\boxed{
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}
}
$$

### Parameters

$$
\boxed{
n^{[l]}n^{[l-1]}+n^{[l]}
}
$$

### EWA

$$
\boxed{
v_t=\beta v_{t-1}+(1-\beta)x_t
}
$$

### Bias Correction

$$
\boxed{
v_t^{corr}=\frac{v_t}{1-\beta^t}
}
$$

### Momentum

$$
\boxed{
v=\beta v+(1-\beta)dW
}
$$

$$
\boxed{W:=W-\alpha v}
$$

### RMSprop

$$
\boxed{
S=\beta S+(1-\beta)(dW)^2
}
$$

$$
\boxed{
W:=W-\alpha\frac{dW}{\sqrt S+\epsilon}
}
$$

### Adam

$$
\boxed{
V=\beta_1V+(1-\beta_1)dW
}
$$

$$
\boxed{
S=\beta_2S+(1-\beta_2)(dW)^2
}
$$

$$
\boxed{
W:=W-\alpha
\frac{V^{corr}}{\sqrt{S^{corr}}+\epsilon}
}
$$

$$
\boxed{
\beta_1=0.9,\;\beta_2=0.999,\;\epsilon=10^{-8}
}
$$

### L2

$$
\boxed{
J=Loss+\frac{\lambda}{2m}\sum_l\|W^{[l]}\|_F^2
}
$$

$$
\boxed{
W=
\left(1-\frac{\lambda\alpha}{m}\right)W
-\alpha(\text{gradient})
}
$$

### BN

$$
\boxed{
z_{norm}=\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}
}
$$

$$
\boxed{
\tilde z=\gamma z_{norm}+\beta
}
$$

### Softmax

$$
\boxed{
A_j=\frac{e^{z_j}}{\sum_ke^{z_k}}
}
$$

### BCE

$$
\boxed{
L=-[y\log\hat y+(1-y)\log(1-\hat y)]
}
$$

### Categorical CE

$$
\boxed{
L=-\sum_jy_j\log\hat y_j
}
$$

One-hot:

$$
\boxed{
L=-\log(\hat y_{\text{true}})
}
$$

---

# 10-MINUTE PRE-EXAM CHEAT SHEET

## 🔥 Absolute Must-Know

* **Linear regression** → Linear output + **MSE**

* **Binary classification** → Sigmoid + **BCE**

* **Multi-class** → Softmax + **Categorical CE**

* **Multilabel** → Multiple Sigmoid + **BCE**

* \(W^{[l]}\) shape:

  $$
  \boxed{(n^{[l]},n^{[l-1]})}
  $$

* Parameters/layer:

  $$
  \boxed{n^{[l]}n^{[l-1]}+n^{[l]}}
  $$

* Gradient descent:

  $$
  \boxed{\theta=\theta-\alpha\nabla J}
  $$

* \(\alpha\) too small → **slow**

* \(\alpha\) too large → **oscillate/diverge**

* Sigmoid:

  $$
  \boxed{\sigma'_{\max}=0.25}
  $$

* Sigmoid/tanh → **saturation + vanishing gradient**

* tanh → **zero-centered**

* ReLU → derivative **1 when \(z>0\)**

* ReLU → **dying neurons** for persistent \(z<0\)

* Leaky ReLU → fixes dying-ReLU issue

* **Batch GD** = batch \(m\)

* **SGD** = batch 1

* **Mini-batch** = practical default

* Momentum → smooths **gradient**

* RMSprop → smooths **squared gradient**

* Adam → **Momentum + RMSprop**

* Adam:

  $$
  \boxed{\beta_1=.9,\;\beta_2=.999,\;\epsilon=10^{-8}}
  $$

* EWA:

  $$
  \boxed{1/(1-\beta)}
  $$

* Bias correction:

  $$
  \boxed{v_t/(1-\beta^t)}
  $$

* High bias → **bigger network / train longer**

* High variance → **more data / regularization**

* **L1 → sparse weights**

* **L2 → weight decay**

* Dropout → **training only**

* Early stopping → **minimum validation loss**

* Normalize using **training-set statistics** for both train and test.

* Batch Norm:

  $$
  \boxed{z\rightarrow\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}}
  \rightarrow\gamma z+\beta}
  $$

* BN makes \(b\) **redundant**.

* Softmax outputs:

  $$
  \boxed{\text{sum}=1}
  $$

* Softmax = **competition**

* Sigmoid outputs = **independent**

---

## ⚠️ 15 Classic MCQ/Short-Answer Traps

1. **Test set tunes hyperparameters?** → NO.
2. **Input layer has learnable parameters?** → NO.
3. \(W\) = previous × current? → **NO; current × previous.**
4. Sigmoid max derivative? → **0.25**.
5. tanh zero-centered? → **YES**.
6. tanh solves vanishing gradients? → **NO**.
7. ReLU negative derivative? → **0**.
8. Dropout at test time? → **NO**.
9. L1 creates sparse solutions? → **YES**.
10. L2 = weight decay? → **YES**.
11. Adam eliminates LR tuning? → **NO**.
12. BN's main purpose = regularization? → **NO**.
13. Bigger BN batch → stronger regularization? → **NO; weaker side effect**.
14. Multilabel → Softmax? → **NO; sigmoid per label**.
15. Multi-class → independent sigmoids? → **No; softmax + CCE**.

---

# ONE-LINE MEMORY MAP

**Regression:**
`Linear → MSE`

**Binary:**
`Sigmoid → BCE`

**Multi-class:**
`Softmax → CCE`

**Multilabel:**
`Sigmoid × K → BCE`

**Activation:**
`Sigmoid/tanh → vanishing`
`ReLU → dying`
`Leaky ReLU → fixes dying`

**Optimizer:**
`GD → current gradient`
`Momentum → averaged gradient`
`RMSprop → averaged squared gradient`
`Adam → both`

**Generalization:**
`High bias → add capacity`
`High variance → constrain / add data`

**Regularization:**
`L1 → sparse`
`L2 → decay`
`Dropout → random neurons off during training`

**Data:**
`Train → learn`
`Dev → tune`
`Test → final`

**BN:**
`normalize → gamma/beta → running stats at test`


---


# Mid-Semester Question Bank — Full Answer Key

*Answers to every question in the Sample Question Bank (Sections A–D), drawn from Modules 0–6.*

---

## Section A: Short Answer Questions

**1. What are the parameters of a neural network? Explain with examples.**
> Parameters are the variables the network learns during training via backpropagation and gradient descent — they directly determine the model's predictions. Examples: the weight matrices $W^{[l]}$ and bias vectors $b^{[l]}$ at each layer $l$. If Batch Normalization is used, $\gamma^{[l]}$ and $\beta^{[l]}$ are also learnable parameters. Parameters are distinguished from hyperparameters (e.g., learning rate, number of layers), which are set by the user before training and are not learned from data.

**2. What is the purpose of using an activation function in a neural network?**
> Activation functions introduce non-linearity into the network. Without them, any stack of layers — no matter how deep — mathematically collapses into a single linear transformation, since a composition of linear functions is itself linear. Non-linear activations allow the network to approximate complex, non-linear decision boundaries and functions, essential for problems that aren't linearly separable.

**3. Define vanishing gradients. Why is it a problem in deep networks?**
> Vanishing gradients occur when the gradient of the loss with respect to early-layer weights becomes extremely small (approaching zero) as it is propagated backward through many layers via the chain rule. This happens because activation functions like sigmoid and tanh have derivatives that are always less than 1 (sigmoid's maximum derivative is 0.25) and shrink further toward 0 under saturation (large $|z|$). Multiplying many such small factors together across layers causes the gradient to shrink exponentially with depth. It is a problem because early layers effectively stop receiving any meaningful gradient signal, so their weights stop updating and the network fails to learn effectively — especially damaging in deep networks with many layers for the gradient to pass through.

**4. What are the dimensions of weight and bias matrices at layer 'L'?**
> If layer $L$ has $n^{[L]}$ neurons and receives input from layer $L-1$ with $n^{[L-1]}$ neurons: $W^{[L]}$ has shape $(n^{[L]}, n^{[L-1]})$, and $b^{[L]}$ has shape $(n^{[L]}, 1)$. This ensures $z^{[L]} = W^{[L]}a^{[L-1]} + b^{[L]}$ is dimensionally valid.

**5. What is the effect of applying dropout during training?**
> Dropout randomly deactivates a fraction $p$ of neurons in a layer during each training iteration/epoch (selection is random each time), forcing the network to not rely too heavily on any single neuron and effectively training a different, thinned sub-network each pass. This reduces overfitting by preventing complex co-adaptations between neurons. At test time, the full network is used, with a scaling correction (multiply weights by $(1-p)$, or use "inverted dropout" which divides training activations by keep_prob so no test-time adjustment is needed).

**6. Compare L1 and L2 regularization.**
> Both add a penalty term to the cost function based on weight magnitude, constraining model capacity to reduce overfitting. L2 (weight decay) uses $\lambda\sum w_j^2$; its gradient is proportional to $w$, producing smooth multiplicative shrinkage toward — but rarely exactly — zero: $W:=(1-\frac{\lambda\alpha}{m})W - \alpha(\text{backprop term})$. L1 uses $\lambda\sum|w_j|$; its gradient is a constant $\pm\lambda$ regardless of $w$'s magnitude, which tends to push many weights to *exactly* zero, producing sparse models (useful for implicit feature selection). L2/weight decay is more common in neural networks in practice.

**7. What are the different methods to solve the high bias and high variance problem?**
> Diagnosis: compare training error (indicates bias) to the training–dev error gap (indicates variance). **For high bias:** use a bigger network, train longer, or search for a better network architecture. **For high variance:** train with more data, apply regularization (L2, dropout, etc.), or search for a better architecture. Fixes should be applied in order — resolve bias first, then variance — since applying a variance remedy to a bias problem (or vice versa) worsens the mismatched issue.

**8. Why do we normalize input data before training a neural network?**
> Normalizing input data (subtracting the mean, then scaling by variance) reshapes the cost surface from an elongated, skewed bowl into a more symmetric, round one. Differently-scaled raw features distort the cost surface unevenly, causing gradient descent to oscillate and converge slowly. Normalization lets gradient descent head more directly toward the minimum, speeding up training.

**9. What is weight initialization, and why is it important?**
> Weight initialization is the one-time assignment of starting values to a network's weight matrices before training begins. It matters because $\text{Var}(a) = n\cdot\text{Var}(w)\cdot\text{Var}(x)$ for a neuron with $n$ inputs — if $n\cdot\text{Var}(w) > 1$, variance (and later, gradients) grow exponentially with depth (exploding gradients); if $n\cdot\text{Var}(w) < 1$, variance shrinks to zero with depth (vanishing gradients). Proper initialization sets $\text{Var}(w) = 1/n$ to keep signal stable across layers — the basis of He/Xavier initialization: $W^{[l]} = \text{randn(shape)}\times\sqrt{1/n^{[l-1]}}$ (or the Bengio variant, $\sqrt{2/(n^{[l-1]}+n^{[l]})}$).

**10. Describe the difference between Gradient descent and gradient descent with momentum.**
> Plain gradient descent updates weights using only the current gradient: $W := W - \alpha\,dW$. Momentum instead computes an exponentially weighted average of past gradients (a "velocity" $v_{dW} = \beta v_{dW} + (1-\beta)dW$) and updates using that smoothed velocity: $W := W - \alpha v_{dW}$. This dampens oscillations in directions where the gradient sign flips frequently, and accelerates progress in directions where the gradient is consistent — generally converging faster than plain gradient descent. Default $\beta = 0.9$.

**11. What does the softmax activation function do?**
> Softmax converts a vector of raw output scores into a probability distribution over $C$ mutually exclusive classes: $A^{[L]}_j = \frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}}$. All $C$ outputs are non-negative and sum to exactly 1, interpretable as class probabilities. Used in the output layer for multi-class (mutually exclusive) classification.

**12. Explain the concept of learning rate. What happens if it is too high or too low?**
> The learning rate α scales the step size of each gradient descent update: $\theta := \theta - \alpha\nabla J(\theta)$. Too small → convergence is correct but very slow (many iterations needed). Too large → updates can overshoot the minimum, causing oscillation or outright divergence (cost increases without bound). Techniques like learning rate decay and adaptive optimizers (RMSprop, Adam) help balance fast early progress against stability near convergence.

**13. What is batch normalization, and why is it applied?**
> Batch Normalization normalizes each hidden layer's pre-activation values ($Z^{[l]}$) within a mini-batch to zero mean and unit variance, then rescales/shifts using learnable parameters $\gamma$ and $\beta$: $\tilde z^{(i)} = \gamma\cdot z_{\text{norm}}^{(i)} + \beta$. It is applied to (1) speed up training by reshaping each layer's local cost landscape favorably (same reasoning as input normalization), and (2) reduce "internal covariate shift" — the constantly-shifting distribution of activations feeding later layers as earlier layers' weights update during training. It has a mild secondary regularization side-effect but should not be relied upon as a substitute for L2/dropout.

**14. What is the train, validation, and test set?**
> The **training set** is used to learn model parameters (weights/biases) via backpropagation. The **validation (dev) set** is used during development to tune hyperparameters and make model decisions (e.g., when to stop training), without directly training on it. The **test set** is used only once, after development is finalized, to report an unbiased estimate of real-world performance. Dev and test sets must come from the same distribution; using only a dev set (no separate test set) can be acceptable, but final evaluation must never reuse training data.

**15. What are one-hot encoded labels, and why are they used in classification problems?**
> A one-hot encoded label is a vector of length $C$ (number of classes) with a single 1 at the index of the true class and 0 elsewhere (e.g., for 4 classes, "cat" → $[0,0,1,0]$). They are used instead of plain integer class labels because integer labels falsely imply an ordinal relationship between classes (e.g., that class 3 is "more" than class 1), and because one-hot vectors match the shape of a softmax output layer's probability vector, allowing elementwise cross-entropy loss computation.

---

## Section B: Multiple Choice Questions

**1. Which activation function suffers from the vanishing gradient problem?**
> **(b) Sigmoid** — its derivative has a maximum of 0.25 and shrinks toward 0 under saturation at large $|z|$, causing gradients to vanish across deep networks. (Tanh shares this issue, but sigmoid is the flagged option here.)

**2. Which of the following is a hyperparameter?**
> **(c) Learning rate** — set by the user before training, not learned from data. Weights, bias, and activation outputs are either parameters or computed values, not hyperparameters.

**3. The ReLU function outputs:**
> **(b) Non-negative values** — $\text{ReLU}(z) = \max(0,z)$, so output is always $\geq 0$.

**4. Which optimizer combines momentum and adaptive learning rates?**
> **(b) Adam** — Adam = Momentum (first-moment estimate) + RMSprop (second-moment/adaptive-scaling estimate), combined with bias correction.

**5. In dropout with p = 0.5, during training:**
> **(b) 50% of neurons are dropped randomly** — a fresh random 50% subset is deactivated each iteration/epoch.

**6. Which loss function is used for multi-class classification?**
> **(c) Categorical Cross Entropy** — paired with softmax activation for mutually exclusive multi-class problems.

**7. Which of the following helps mitigate exploding gradients?**
> **(a) Gradient clipping** — caps gradient magnitude at a threshold before the weight update, directly preventing excessively large updates from exploding gradients.

**8. CNNs are mainly used for:**
> **(b) Image and video analysis** — convolutional architectures are designed to exploit spatial structure, making them well-suited to visual data. *(Note: CNNs themselves are not covered in your uploaded slide decks — this answer reflects general deep learning knowledge, not slide-specific content.)*

**9. Backpropagation calculates:**
> **(b) Error gradients** — backpropagation computes $\frac{\partial J}{\partial W^{[l]}}$ and $\frac{\partial J}{\partial b^{[l]}}$ via the chain rule; it does not itself compute forward outputs, activation functions, or the loss function value (those are separate steps).

**10. Which is NOT a reason for using normalization?**
> **(c) To add non-linearity** — normalization (input normalization or batch norm) reshapes the cost surface and stabilizes activation distributions; it does not introduce non-linearity (that is the role of activation functions).

**11. What is the derivative of ReLU for negative input values?**
> **(a) 0** — for $z<0$, $\text{ReLU}'(z) = 0$ exactly.

**12. Which technique is specifically for reducing overfitting?**
> **(a) Dropout** — a regularization technique specifically designed to reduce overfitting. (Weight initialization and gradient descent address training dynamics/optimization, not overfitting directly; learning rate decay aids convergence, not generalization.)

**13. Which function is mostly used in the output layer for binary classification?**
> **(a) Sigmoid** — outputs a single probability in $(0,1)$ interpretable as $P(y=1|x)$, paired with Binary Cross-Entropy.

---

## Section C: Fill in the Blanks

**1. The gradient of sigmoid activation becomes very small when input is very ___ or very ___.**
> **large (positive)** or **small (very negative)** — sigmoid saturates in either direction, driving its derivative toward 0.

**2. ReLU activation function outputs ___ for negative values.**
> **0** — $\text{ReLU}(z) = \max(0,z) = 0$ for $z<0$.

**3. The total number of learnable parameters in a fully connected layer is given by ___.**
> $n^{[l]} \times n^{[l-1]} + n^{[l]}$ — (weights: current layer size × previous layer size) + (biases: one per current-layer neuron).

**4. Dropout is used to reduce ___.**
> **overfitting** (equivalently, variance / generalization error).

**5. Z-score normalization makes mean = ___ and standard deviation = ___.**
> mean = **0**, standard deviation = **1**.

**6. The function used in the output layer for multi-class classification is ___.**
> **Softmax**.

**7. Cross entropy loss is suitable for ___ tasks.**
> **classification** tasks (binary cross-entropy for binary/multilabel; categorical cross-entropy for multi-class).

**8. The derivative of ReLU for positive inputs is ___.**
> **1**.

**9. Increasing batch size generally makes training ___ but less noisy.**
> **slower (per-iteration progress)** / **more stable** — larger batches give a more accurate but computationally heavier gradient estimate per step, reducing noise/oscillation compared to small mini-batches or SGD.

**10. Gradient descent updates weights in the direction of ___ of the loss function.**
> **the negative gradient (steepest descent)** — i.e., opposite to the gradient's direction, since the gradient points toward the steepest *increase*.

**11. Weight initialization helps avoid ___ gradients at the start of training.**
> **vanishing (and exploding)** gradients.

**12. Batch normalization is applied between ___ and ___.**
> between the **linear step** ($Z^{[l]} = W^{[l]}A^{[l-1]}$) and the **activation function** ($A^{[l]} = g(Z^{[l]})$) — i.e., $Z$ is normalized before the nonlinearity is applied.

**13. Vanishing gradients occur due to activation functions like ___ and ___.**
> **sigmoid** and **tanh**.

**14. RMSProp optimizer uses exponentially weighted ___ of gradients.**
> **averages of the squared gradients** (second moment) — $S_{dW} = \beta S_{dW} + (1-\beta)(dW)^2$.

---

## Section D: True/False

**1. Softmax is used in binary classification.**
> **False** (in the conventional sense). Sigmoid is the standard choice for binary classification. Softmax is designed for multi-class (mutually exclusive, $C>2$) classification, though it is mathematically usable in a 2-class setting — the expected/conventional answer here is False.

**2. Adding more layers always improves neural network performance.**
> **False**. Adding layers increases model capacity, which can help with high bias, but beyond a certain point leads to overfitting (high variance), vanishing/exploding gradient issues, and diminishing or negative returns. There is an "optimal capacity" beyond which generalization error increases.

**3. Batch normalization eliminates the need for bias terms.**
> **True**. Because Batch Norm's first step subtracts the batch mean from $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$, and $b^{[l]}$ is a constant added identically to every example, mean-subtraction cancels $b^{[l]}$'s effect entirely — its role is taken over by the learnable $\beta^{[l]}$ parameter instead.

**4. Overfitting happens when the model performs well on training but poorly on test data.**
> **True**. This is precisely the definition of high variance / overfitting — a large gap between low training error and higher test/dev error.

**5. Weight decay is another name for L2 regularization.**
> **True**. The L2 penalty term's gradient produces a multiplicative shrinkage of weights at each update step ($W := (1-\frac{\lambda\alpha}{m})W - \dots$), which is why L2 regularization is commonly called "weight decay."

**6. Dropout is only applied during inference.**
> **False**. Dropout is applied during **training only**; at inference/test time, the full network is used (with a scaling correction to account for the difference in expected activation magnitude).

**7. Learning rate decay is a strategy to gradually reduce learning rate during training.**
> **True**. This is exactly its definition — reducing α over epochs/iterations so that step sizes shrink as training approaches convergence, reducing oscillation near the minimum.

**8. ReLU activation avoids vanishing gradient problem for positive inputs.**
> **True**. For $z>0$, ReLU's derivative is a constant 1 (no saturation), so the gradient signal is not diminished by the activation function in this region.

**9. The derivative of sigmoid activation never exceeds 0.25.**
> **True**. $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which is maximized at $z=0$ where $\sigma(z)=0.5$, giving $\sigma'(0) = 0.5\times0.5 = 0.25$ — the global maximum of the derivative.

**10. Early stopping prevents overfitting by stopping training before validation error increases.**
> **True**. Early stopping monitors validation loss/error and halts training at the point where it starts rising again (even as training loss continues to fall), selecting the checkpoint with minimum validation error.

**11. Adam optimizer combines momentum and adaptive learning rates.**
> **True**. Adam maintains a momentum-style first-moment estimate ($V_{dW}$) and an RMSprop-style adaptive second-moment estimate ($S_{dW}$), combining both (with bias correction) in its update rule.

**12. Cross entropy loss is differentiable and suitable for classification tasks.**
> **True**. Both binary and categorical cross-entropy are differentiable (required for gradient-based optimization) and are the standard loss choices for classification, since they produce convex cost surfaces when paired with sigmoid/softmax respectively and directly penalize low predicted probability on the correct class.

**13. Data augmentation helps improve generalization.**
> **True**. By generating synthetic variations of training examples (rotations, crops, noise, etc.), data augmentation increases effective dataset size and diversity, making the model more robust and less prone to memorizing exact training examples — a form of regularization.

---

*End of answer key. Cross-reference: Sections A–D map to Modules 0–6 in the companion "Deep Learning Mid-Semester Exam Prep" document.*