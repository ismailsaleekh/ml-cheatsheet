# ML Cheatsheet Content - Sections 7-8: Supervised Learning

## 7. SUPERVISED LEARNING - REGRESSION

### 7.1 Linear Models

---

#### 7.1.1 Linear Regression

**ID:** `linear-regression`
**Parent:** `7.1`

**Full Explanation:**
Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. The model assumes: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε, where β are coefficients and ε is the error term. Parameters are typically estimated using Ordinary Least Squares (OLS), minimizing the sum of squared residuals. Assumes linearity, independence, homoscedasticity, and normally distributed errors.

**Simple Explanation:**
Draw the best straight line through your data points. The line predicts the output based on inputs. If you know square footage, you can predict house price by following the line. Simple, interpretable, and often surprisingly effective.

**Example:**
Predicting house price from square footage:
- Data: [(1000 sqft, $200K), (1500 sqft, $300K), (2000 sqft, $400K)]
- Model learns: Price = $100 × sqft + $100K
- New house 1800 sqft → Predicted price = $100 × 1800 + $100K = $280K

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

#### 7.1.2 Multiple Linear Regression

**ID:** `multiple-linear-regression`
**Parent:** `7.1`

**Full Explanation:**
Multiple Linear Regression extends simple linear regression to multiple predictor variables: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ. Each coefficient βᵢ represents the change in y for a one-unit change in xᵢ, holding all other variables constant. Enables modeling complex relationships while maintaining interpretability. Requires checking for multicollinearity between predictors.

**Simple Explanation:**
Use multiple inputs to make a prediction. Instead of just square footage, use square footage AND bedrooms AND location to predict price. Each factor contributes independently to the final prediction.

**Example:**
House price with multiple features:
- Price = $50×sqft + $10,000×bedrooms - $5,000×age + $20,000×(good_school_district)
- 1500 sqft, 3 bed, 10 years old, good schools:
- Price = $50×1500 + $10K×3 + (-$5K×10) + $20K = $75K + $30K - $50K + $20K = $125K

---

#### 7.1.3 Polynomial Regression

**ID:** `polynomial-regression`
**Parent:** `7.1`

**Full Explanation:**
Polynomial Regression fits a polynomial equation to the data: y = β₀ + β₁x + β₂x² + ... + βₙxⁿ. Despite the nonlinear relationship with x, it's still linear in parameters (coefficients), so linear regression techniques apply. Higher-degree polynomials can fit complex curves but risk overfitting. Degree selection is crucial—use cross-validation.

**Simple Explanation:**
When a straight line doesn't fit, use a curve. Add x², x³, etc. as features. A degree-2 polynomial can fit a parabola. Higher degrees fit more complex shapes but may overfit.

**Example:**
Data follows a parabola (throwing a ball):
- Height = -5t² + 20t + 1 (physics equation)
- Linear regression: Poor fit (straight line through curve)
- Polynomial degree 2: Perfect fit (captures the parabola)
- Polynomial degree 10: Overfits (wiggles through every point)

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # Creates [1, x, x²]
model.fit(X_poly, y)
```

---

#### 7.1.4 Ridge Regression

**ID:** `ridge-regression`
**Parent:** `7.1`

**Full Explanation:**
Ridge Regression (L2 regularization) adds a penalty term to the loss function: Loss = Σ(yᵢ - ŷᵢ)² + λΣβⱼ². The penalty shrinks coefficients toward zero, reducing variance at the cost of introducing bias. Particularly effective when predictors are correlated (multicollinearity) or when there are more features than observations. λ controls regularization strength.

**Simple Explanation:**
Linear regression that keeps weights small. Adds a penalty for large coefficients, preventing the model from relying too heavily on any single feature. Good when features are correlated or you have many features.

**Example:**
Predicting with correlated features (height in cm AND height in inches):
- Regular regression: Unstable coefficients, one huge positive, one huge negative
- Ridge regression: Both coefficients moderate, model more stable

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # alpha is λ
model.fit(X_train, y_train)
```

---

#### 7.1.5 Lasso Regression

**ID:** `lasso-regression`
**Parent:** `7.1`

**Full Explanation:**
Lasso Regression (Least Absolute Shrinkage and Selection Operator) uses L1 regularization: Loss = Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|. Unlike Ridge, Lasso can shrink coefficients exactly to zero, performing automatic feature selection. Useful when you suspect many features are irrelevant. The L1 penalty creates sparse models.

**Simple Explanation:**
Linear regression that eliminates useless features. The L1 penalty drives some coefficients to exactly zero, effectively removing those features. Great for feature selection—tells you which features actually matter.

**Example:**
100 features, but only 10 actually matter:
- Regular regression: All 100 have non-zero coefficients
- Lasso: Only 10-15 features have non-zero coefficients, rest are exactly 0
- Interpretation: "These 10 features drive the prediction"

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
# Check which features were selected
important_features = X.columns[model.coef_ != 0]
```

---

#### 7.1.6 Elastic Net

**ID:** `elastic-net`
**Parent:** `7.1`

**Full Explanation:**
Elastic Net combines L1 and L2 regularization: Loss = Σ(yᵢ - ŷᵢ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ². It inherits both feature selection from Lasso and coefficient shrinkage from Ridge. Particularly useful when features are correlated—Lasso might arbitrarily select one, while Elastic Net tends to keep correlated features together. Two hyperparameters: overall regularization and L1/L2 ratio.

**Simple Explanation:**
Best of both Ridge and Lasso. Gets feature selection (some zeros) AND keeps correlated features together. Use when you want sparsity but features might be related.

**Example:**
Gene expression data: 10,000 genes, many correlated in pathways
- Lasso: Picks one gene from each pathway arbitrarily
- Elastic Net: Keeps groups of related genes together, still sparse

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50% L1, 50% L2
model.fit(X_train, y_train)
```

---

### 7.2 Kernel Methods

---

#### 7.2.1 Support Vector Regression (SVR)

**ID:** `svr`
**Parent:** `7.2`

**Full Explanation:**
Support Vector Regression applies SVM principles to regression. It finds a function that deviates from actual targets by at most ε (epsilon) for all training data, while being as flat as possible. Points within the ε-tube don't contribute to the loss (ε-insensitive loss). Support vectors are points at or outside the tube boundaries. Kernels enable nonlinear regression.

**Simple Explanation:**
Regression using SVM concepts. Instead of minimizing all errors, it ignores small errors (within a tolerance tube) and focuses on keeping the function smooth. Points that define the tube edges are "support vectors."

**Example:**
Predicting stock prices with tolerance:
- ε = $0.50 (we don't care about errors smaller than 50 cents)
- SVR finds a smooth curve where most points are within $0.50
- Only points outside the tube affect the model
- Result: Robust to small noise, smooth predictions

```python
from sklearn.svm import SVR
model = SVR(kernel='rbf', epsilon=0.1, C=100)
model.fit(X_train, y_train)
```

---

#### 7.2.2 Kernel Trick

**ID:** `kernel-trick`
**Parent:** `7.2`

**Full Explanation:**
The kernel trick enables algorithms to operate in high-dimensional feature spaces without explicitly computing the transformation. Instead of mapping x → φ(x) and computing φ(x)ᵀφ(y), we compute k(x,y) = φ(x)ᵀφ(y) directly. This allows learning nonlinear patterns with linear algorithms. Common kernels: RBF (Gaussian), polynomial, sigmoid.

**Simple Explanation:**
A mathematical shortcut to work in higher dimensions without the cost. Imagine transforming 2D data to 1000D to find a pattern—expensive! The kernel trick gets the same result without actually doing the transformation.

**Example:**
XOR problem (not linearly separable in 2D):
- Original: [(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0]
- Cannot draw a straight line to separate

With polynomial kernel (implicitly maps to higher dimension):
- Kernel computes similarity as if data were in higher dimension
- Linear separator found in implicit high-dimensional space
- Nonlinear boundary in original 2D space

---

#### 7.2.3 RBF Kernel

**ID:** `rbf-kernel`
**Parent:** `7.2`

**Full Explanation:**
The Radial Basis Function (RBF) kernel, also called Gaussian kernel, measures similarity as: k(x,y) = exp(-γ||x-y||²). Points close together have similarity near 1; distant points have similarity near 0. The γ parameter controls the kernel's reach—higher γ means tighter, more local influence. RBF kernel can approximate any continuous function given enough data.

**Simple Explanation:**
Similarity based on distance—close points are similar, far points are different. Like a spotlight: each point illuminates its neighborhood. γ controls spotlight width—small γ = wide reach, large γ = focused locally.

**Example:**
Two points in 2D:
- x = (0, 0), y = (1, 1)
- Distance ||x-y||² = 2
- With γ = 0.5: k(x,y) = exp(-0.5 × 2) = 0.37
- With γ = 2.0: k(x,y) = exp(-2 × 2) = 0.02

Higher γ makes distant points less similar (tighter kernel).

---

### 7.3 Tree-Based Regression

---

#### 7.3.1 Decision Tree Regressor

**ID:** `decision-tree-regressor`
**Parent:** `7.3`

**Full Explanation:**
Decision Tree Regressor recursively partitions the feature space into regions and predicts the mean target value within each region. Splits are chosen to minimize variance (or MSE) in resulting regions. Trees are interpretable and handle nonlinear relationships naturally. Prone to overfitting without pruning or depth limits. Predictions are piecewise constant.

**Simple Explanation:**
A flowchart that makes predictions. "Is square footage > 1500? If yes, is it in a good neighborhood? If yes, predict $400K." Each leaf contains an average value from training data in that region.

**Example:**
Predicting house prices:
```
                    [sqft > 1500?]
                    /            \
                 No               Yes
                 /                  \
        [age > 20?]           [bedrooms > 3?]
         /      \               /         \
      $150K    $200K        $350K       $450K
```

New house: 1800 sqft, 4 bedrooms → Follow: Yes → Yes → Predict $450K

---

#### 7.3.2 Regression Tree Splitting

**ID:** `regression-tree-splitting`
**Parent:** `7.3`

**Full Explanation:**
Regression trees split nodes to minimize prediction error, typically measured by MSE or MAE. For each feature, the algorithm considers all possible split points and selects the one that minimizes the weighted average variance of resulting child nodes. This greedy approach finds locally optimal splits. Stopping criteria include minimum samples, maximum depth, or minimum impurity decrease.

**Simple Explanation:**
Try every possible way to split the data, pick the one that groups similar target values together. "Splitting at sqft=1500 creates groups with less price variation than splitting at sqft=1200."

**Example:**
Data: Houses with prices [$100K, $150K, $400K, $450K]
Features: sqft [1000, 1200, 1800, 2000]

Trying splits:
- Split at sqft=1100: Left [$100K], Right [$150K, $400K, $450K]
  - Variance: Left=0, Right=high → Not great
- Split at sqft=1500: Left [$100K, $150K], Right [$400K, $450K]
  - Variance: Left=low, Right=low → Good split!

Choose sqft=1500 because it creates more homogeneous groups.

---

### 7.4 Instance-Based Regression

---

#### 7.4.1 K-Nearest Neighbors Regression

**ID:** `knn-regression`
**Parent:** `7.4`

**Full Explanation:**
KNN Regression predicts the target value as the average (or weighted average) of the K nearest training examples. Distance is typically Euclidean, but other metrics work. No explicit training phase—all computation happens at prediction time. Sensitive to feature scaling and curse of dimensionality. K controls bias-variance: small K = low bias, high variance; large K = high bias, low variance.

**Simple Explanation:**
To predict, find the K most similar examples and average their values. Predicting a house price? Find the 5 most similar houses that sold, average their prices. Simple, intuitive, no training needed.

**Example:**
Predict price for 1600 sqft house, using K=3:

Training data:
- 1500 sqft → $300K (distance: 100)
- 1550 sqft → $310K (distance: 50)
- 1650 sqft → $330K (distance: 50)
- 2000 sqft → $450K (distance: 400)

3 nearest: 1550, 1650, 1500 sqft houses
Prediction: ($310K + $330K + $300K) / 3 = $313K

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
```

---

#### 7.4.2 Weighted KNN

**ID:** `weighted-knn`
**Parent:** `7.4`

**Full Explanation:**
Weighted KNN assigns higher influence to closer neighbors. Common weighting: inverse distance (w = 1/d) or inverse squared distance (w = 1/d²). Closer neighbors contribute more to the prediction than farther ones. Reduces the impact of K selection—even with large K, distant neighbors have minimal influence. Smooths predictions compared to uniform weighting.

**Simple Explanation:**
Closer neighbors count more. Instead of equal votes, a house 10 feet away has more influence than one 100 feet away. Closer = more similar = should matter more.

**Example:**
Predict price, K=3 with distance weighting:

Neighbors:
- 1550 sqft → $310K, distance=50, weight=1/50=0.02
- 1650 sqft → $330K, distance=50, weight=1/50=0.02
- 1500 sqft → $300K, distance=100, weight=1/100=0.01

Weighted average:
= (0.02×$310K + 0.02×$330K + 0.01×$300K) / (0.02+0.02+0.01)
= ($6.2K + $6.6K + $3K) / 0.05
= $316K

---

## 8. SUPERVISED LEARNING - CLASSIFICATION

### 8.1 Linear Classifiers

---

#### 8.1.1 Logistic Regression

**ID:** `logistic-regression`
**Parent:** `8.1`

**Full Explanation:**
Logistic Regression models the probability of a binary outcome using the logistic (sigmoid) function: P(y=1|x) = 1/(1+e^(-z)) where z = β₀ + β₁x₁ + ... + βₙxₙ. Output is a probability between 0 and 1. Decision boundary is linear in feature space. Trained by maximizing likelihood (minimizing log loss). Despite its name, it's a classification algorithm.

**Simple Explanation:**
Predict the probability of something being true. Instead of predicting a number, predict "80% chance this email is spam." The sigmoid function squashes any value into 0-1 range, giving you a probability.

**Example:**
Email spam classification:
- Features: word counts, sender reputation, etc.
- Model outputs: P(spam) = 0.85
- Threshold 0.5: Predict spam (0.85 > 0.5)

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)  # Get probabilities
predictions = model.predict(X_test)  # Get class labels
```

---

#### 8.1.2 Sigmoid Function

**ID:** `sigmoid-function`
**Parent:** `8.1`

**Full Explanation:**
The sigmoid function σ(z) = 1/(1+e^(-z)) maps any real number to the (0,1) interval. Properties: σ(0) = 0.5, σ(-∞) → 0, σ(+∞) → 1. Smooth and differentiable everywhere, enabling gradient-based optimization. Derivative: σ'(z) = σ(z)(1-σ(z)). Used in logistic regression and as activation function in neural networks.

**Simple Explanation:**
An S-shaped curve that squashes any number into 0-1. Big positive numbers become close to 1, big negative numbers become close to 0, and 0 maps to 0.5. Perfect for converting scores to probabilities.

**Example:**
```
Input z:  -5    -2     0    +2    +5
Output:  0.01  0.12  0.50  0.88  0.99

Graph:
1.0 |              ****
    |           ***
0.5 |        **
    |     ***
0.0 |*****
    ----------------------
       -5   0   +5
```

---

#### 8.1.3 Softmax Function

**ID:** `softmax-function`
**Parent:** `8.1`

**Full Explanation:**
Softmax generalizes sigmoid to multiple classes, converting a vector of real numbers into a probability distribution. For class i: P(i) = e^(zᵢ) / Σⱼe^(zⱼ). All outputs sum to 1 and are positive. Larger input values get higher probabilities. Used as the final layer in multiclass classification neural networks. Combined with cross-entropy loss for training.

**Simple Explanation:**
Turn a list of scores into probabilities that sum to 1. If a model outputs [2, 1, 0] for three classes, softmax converts to [0.67, 0.24, 0.09]—class 1 is most likely, but you still see relative confidence in others.

**Example:**
Model outputs (logits): [2.0, 1.0, 0.5]

Softmax calculation:
- e^2.0 = 7.39
- e^1.0 = 2.72
- e^0.5 = 1.65
- Sum = 11.76

Probabilities:
- P(class 0) = 7.39/11.76 = 0.63
- P(class 1) = 2.72/11.76 = 0.23
- P(class 2) = 1.65/11.76 = 0.14

Sum = 1.0 ✓

---

#### 8.1.4 Multiclass Classification

**ID:** `multiclass-classification`
**Parent:** `8.1`

**Full Explanation:**
Multiclass classification assigns instances to one of more than two classes. Approaches: (1) One-vs-Rest (OvR): Train K binary classifiers, each separating one class from all others. (2) One-vs-One (OvO): Train K(K-1)/2 classifiers for each pair of classes. (3) Native multiclass: Some algorithms naturally handle multiple classes (softmax regression, decision trees, naive Bayes).

**Simple Explanation:**
Predict one category from several options (not just yes/no). Examples: classifying images as cat/dog/bird, or emails as work/personal/spam. Different strategies exist to extend binary classifiers to multiple classes.

**Example:**
Classifying animals into 3 classes: Cat, Dog, Bird

One-vs-Rest approach:
- Classifier 1: Cat vs (Dog+Bird)
- Classifier 2: Dog vs (Cat+Bird)
- Classifier 3: Bird vs (Cat+Dog)

New image → Run all 3 classifiers → Pick highest confidence:
- "Cat vs Rest": 0.8 ← Winner
- "Dog vs Rest": 0.3
- "Bird vs Rest": 0.1
→ Predict: Cat

---

#### 8.1.5 Multilabel Classification

**ID:** `multilabel-classification`
**Parent:** `8.1`

**Full Explanation:**
Multilabel classification assigns multiple labels to each instance simultaneously. Unlike multiclass (exactly one label), an instance can have zero, one, or many labels. Approaches: binary relevance (independent classifier per label), classifier chains (classifiers conditioned on previous predictions), or neural networks with sigmoid outputs. Common in tagging, medical diagnosis, document categorization.

**Simple Explanation:**
An item can belong to multiple categories at once. A movie can be both "Comedy" AND "Romance" AND "Drama." Train a separate yes/no classifier for each label, or use methods that capture label dependencies.

**Example:**
Movie genre classification (multilabel):
- Movie: "The Proposal"
- Labels: [Comedy: Yes, Romance: Yes, Action: No, Drama: No, Thriller: No]

vs. Multiclass (only one allowed):
- Movie: "The Proposal"
- Label: Romance (must pick just one)

Multilabel allows: ["Comedy", "Romance"] simultaneously.

---

### 8.2 Support Vector Machines

---

#### 8.2.1 Support Vector Machine (SVM)

**ID:** `svm`
**Parent:** `8.2`

**Full Explanation:**
SVM finds the hyperplane that maximizes the margin between classes. The margin is the distance from the hyperplane to the nearest data points (support vectors). Maximizing margin improves generalization. For linearly inseparable data, soft-margin SVM allows some misclassifications (controlled by C parameter). Kernel trick enables nonlinear boundaries.

**Simple Explanation:**
Find the widest possible "street" separating two classes. The street's edges touch the closest points from each side (support vectors). Wider street = more confident separation = better generalization.

**Example:**
Separating spam from non-spam:
```
        Spam
    x   x
  x   x        <- Support vectors (on margin edge)
------------------ <- Decision boundary (middle of street)
  o   o        <- Support vectors (on margin edge)
    o   o
      Non-spam
```

Margin width determines confidence. Points far from boundary are clearly classified; points near are uncertain.

---

#### 8.2.2 Maximum Margin

**ID:** `maximum-margin`
**Parent:** `8.2`

**Full Explanation:**
Maximum margin is the SVM's optimization objective: find the hyperplane with the largest distance to the nearest training points. Mathematically, maximize 2/||w|| subject to yᵢ(w·xᵢ + b) ≥ 1. Larger margins correlate with better generalization (PAC learning bounds). Only support vectors affect the solution—other points could be removed without changing the boundary.

**Simple Explanation:**
Make the gap between classes as wide as possible. A tiny gap means small changes could cause misclassification. A wide gap means robust separation—new points falling in the gap are clearly uncertain rather than wrongly classified.

**Example:**
Two possible separating lines:
```
Option A (small margin):        Option B (large margin):
    x x x                           x x x
-----------  (margin=0.1)
    o o o                         =========  (margin=2.0)

                                    o o o
```

Option B is better—more room for error, more confidence in classification.

---

#### 8.2.3 Soft Margin SVM

**ID:** `soft-margin-svm`
**Parent:** `8.2`

**Full Explanation:**
Soft margin SVM allows some training points to violate the margin or be misclassified, introducing slack variables ξᵢ. Objective: minimize ||w||²/2 + C·Σξᵢ. The C parameter controls the tradeoff—large C penalizes violations heavily (harder margin), small C allows more violations (softer margin). Essential for real-world noisy data where perfect separation is impossible or would overfit.

**Simple Explanation:**
Allow some mistakes to get a better overall boundary. Real data has noise and outliers. Instead of contorting the boundary to classify every point, accept a few errors for a simpler, more generalizable model.

**Example:**
Data with one outlier:
```
Hard margin (C=∞):              Soft margin (C=1):
   x x x                           x x x
      x (outlier)                     x ← allowed to be wrong
 __________                      __________
  o o o o                         o o o o
```

Hard margin twists to include outlier → overfits
Soft margin ignores outlier → better boundary

---

#### 8.2.4 C Parameter

**ID:** `svm-c-parameter`
**Parent:** `8.2`

**Full Explanation:**
The C parameter in SVM controls regularization by setting the penalty for margin violations. High C: Strong penalty for violations, model tries to classify all points correctly (risk of overfitting). Low C: Weak penalty, allows more violations for a simpler model (risk of underfitting). C is typically tuned via cross-validation. It's the inverse of regularization strength.

**Simple Explanation:**
How much do we care about misclassifying training points? High C = "classify everything correctly, even if boundary is complex." Low C = "simple boundary is more important than perfect training accuracy."

**Example:**
Same dataset, different C values:
```
C = 0.01 (low):                 C = 1000 (high):
Simple boundary                 Complex boundary
Some training errors            All training points correct
Generalizes well                May overfit

Train acc: 92%                  Train acc: 100%
Test acc: 90%                   Test acc: 85%
```

---

### 8.3 Probabilistic Classifiers

---

#### 8.3.1 Naive Bayes

**ID:** `naive-bayes`
**Parent:** `8.3`

**Full Explanation:**
Naive Bayes applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class: P(y|x₁,...,xₙ) ∝ P(y)∏P(xᵢ|y). Despite the unrealistic independence assumption, it works surprisingly well in practice, especially for text classification. Fast training and prediction, handles high-dimensional data well, and provides probability estimates.

**Simple Explanation:**
Classify by multiplying probabilities, assuming features are independent. "Given this is spam, what's the probability of seeing 'free'? What about 'money'? What about 'offer'?" Multiply all together. Simple but effective, especially for text.

**Example:**
Email spam classification:
- P(spam) = 0.3, P(not spam) = 0.7
- P("free" | spam) = 0.8, P("free" | not spam) = 0.1
- P("money" | spam) = 0.7, P("money" | not spam) = 0.05

Email contains "free" and "money":
- P(spam | words) ∝ 0.3 × 0.8 × 0.7 = 0.168
- P(not spam | words) ∝ 0.7 × 0.1 × 0.05 = 0.0035

Normalized: P(spam) = 0.168/(0.168+0.0035) = 98%

---

#### 8.3.2 Gaussian Naive Bayes

**ID:** `gaussian-naive-bayes`
**Parent:** `8.3`

**Full Explanation:**
Gaussian Naive Bayes assumes continuous features follow Gaussian (normal) distributions within each class. For each class, it estimates mean μ and variance σ² per feature. Likelihood: P(xᵢ|y) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²)). Best for continuous data that's approximately normally distributed. Fast and works well with small training sets.

**Simple Explanation:**
Assume each feature follows a bell curve within each class. Measure the average height of spam emails vs non-spam emails. A new email's height is compared to both bell curves to see which it fits better.

**Example:**
Classifying flowers by petal length:
- Setosa: mean=1.5cm, std=0.2cm
- Versicolor: mean=4.0cm, std=0.5cm

New flower with petal length 1.6cm:
- P(1.6cm | Setosa) = Gaussian(1.6, μ=1.5, σ=0.2) = high
- P(1.6cm | Versicolor) = Gaussian(1.6, μ=4.0, σ=0.5) = very low

→ Predict Setosa

---

#### 8.3.3 Multinomial Naive Bayes

**ID:** `multinomial-naive-bayes`
**Parent:** `8.3`

**Full Explanation:**
Multinomial Naive Bayes models feature counts following multinomial distribution—ideal for text represented as word counts or TF-IDF. P(xᵢ|y) represents the probability of observing word i given class y. Handles variable document lengths naturally. Widely used for text classification, spam filtering, and sentiment analysis. Requires non-negative feature values.

**Simple Explanation:**
Perfect for counting things, especially words. "How often does 'sale' appear in spam vs legitimate emails?" The more times a word appears, the more evidence it provides. Standard choice for text classification.

**Example:**
Document classification:
- Training: Learn P(word|category) for each word
- Spam emails have more "buy", "free", "offer"
- Ham emails have more "meeting", "project", "schedule"

New email: "Free offer! Buy now!"
- Each word adds evidence toward spam
- Result: Classified as spam

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_tfidf, y_train)
```

---

### 8.4 Instance-Based Classification

---

#### 8.4.1 K-Nearest Neighbors Classification

**ID:** `knn-classification`
**Parent:** `8.4`

**Full Explanation:**
KNN Classification assigns the class most common among the K nearest training examples (majority voting). Ties can be broken randomly or by distance weighting. Non-parametric and makes no assumptions about data distribution. Decision boundaries can be complex and nonlinear. Sensitive to feature scaling, irrelevant features, and curse of dimensionality. K should be odd for binary classification to avoid ties.

**Simple Explanation:**
Look at the K most similar examples, take a vote. If 4 of 5 nearest neighbors are "cat," predict "cat." No training needed—just memorize all examples and compare at prediction time.

**Example:**
Classify a new point with K=5:
```
    New point: ?

    5 nearest neighbors:
    - Point 1: Cat (distance: 0.5)
    - Point 2: Cat (distance: 0.7)
    - Point 3: Dog (distance: 0.8)
    - Point 4: Cat (distance: 0.9)
    - Point 5: Dog (distance: 1.0)

    Vote: Cat=3, Dog=2
    → Predict: Cat
```

---

#### 8.4.2 Choosing K

**ID:** `choosing-k`
**Parent:** `8.4`

**Full Explanation:**
K selection is crucial in KNN. Small K (e.g., 1): High variance, sensitive to noise, complex decision boundaries. Large K: High bias, smoother boundaries, may include points from other regions. Common approach: test multiple K values using cross-validation, select K with best validation performance. Often K=√n is a starting point. Use odd K for binary classification.

**Simple Explanation:**
K=1: Just copy the closest neighbor—noisy but fits training data perfectly.
K=100: Average many neighbors—smooth but might miss local patterns.
Use cross-validation to find the sweet spot.

**Example:**
Testing different K values:
```
K=1:  Training acc=100%, Test acc=85% (overfitting)
K=5:  Training acc=95%,  Test acc=92% (good balance) ✓
K=15: Training acc=90%,  Test acc=91% (slightly oversmoothed)
K=50: Training acc=85%,  Test acc=88% (underfitting)
```

K=5 achieves best test accuracy, balancing bias and variance.

---

### 8.5 Decision Trees

---

#### 8.5.1 Decision Tree Classifier

**ID:** `decision-tree-classifier`
**Parent:** `8.5`

**Full Explanation:**
Decision Tree Classifier recursively partitions the feature space using binary splits that maximize class purity. Each internal node tests a feature, each branch represents an outcome, each leaf predicts a class. Splitting criteria: Gini impurity or information gain (entropy). Trees are interpretable, handle nonlinear relationships, and require no feature scaling. Prone to overfitting without pruning.

**Simple Explanation:**
A series of yes/no questions leading to a prediction. "Is age > 30? If yes, is income > $50K? If yes, predict: will buy." Easy to understand and explain to non-technical stakeholders.

**Example:**
Loan approval decision tree:
```
                [Income > $50K?]
                 /            \
              Yes              No
               /                \
    [Credit > 700?]        [Employed?]
       /       \              /      \
    Approve  Review       Review    Deny
```

Applicant: Income=$60K, Credit=720
→ Path: Yes → Yes → Decision: Approve

---

#### 8.5.2 Gini Impurity

**ID:** `gini-impurity`
**Parent:** `8.5`

**Full Explanation:**
Gini impurity measures class distribution impurity: Gini = 1 - Σpᵢ². Range: 0 (pure, all one class) to 0.5 for binary (equal split). Decision trees minimize weighted Gini impurity when splitting. Lower is better—pure nodes have Gini=0. Computationally simpler than entropy, often produces similar results. Used by default in scikit-learn.

**Simple Explanation:**
How mixed are the classes in a group? Gini=0 means all same class (pure). Gini=0.5 means 50-50 split (maximum impurity). Trees split to create purer groups.

**Example:**
Node A: 100 samples, 90 class A, 10 class B
Gini = 1 - (0.9² + 0.1²) = 1 - 0.82 = 0.18 (fairly pure)

Node B: 100 samples, 50 class A, 50 class B
Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.50 (maximum impurity)

Node C: 100 samples, 100 class A, 0 class B
Gini = 1 - (1.0² + 0²) = 0 (perfectly pure)

---

#### 8.5.3 Information Gain (Entropy)

**ID:** `information-gain`
**Parent:** `8.5`

**Full Explanation:**
Information gain measures entropy reduction from a split. Entropy: H = -Σpᵢ log₂(pᵢ), measuring class uncertainty (0 = pure, 1 = maximum for binary). Information gain = H(parent) - weighted_average(H(children)). Higher gain means better split. ID3 and C4.5 algorithms use this criterion. Slightly more computationally expensive than Gini but theoretically grounded.

**Simple Explanation:**
How much does a split reduce uncertainty? Before split: "could be cat or dog" (uncertain). After split: left branch is all cats, right is all dogs (certain). Information gain measures this uncertainty reduction.

**Example:**
Before split: 50 cats, 50 dogs
Entropy = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1.0 (maximum)

After split on "has whiskers":
- Left (whiskers=yes): 48 cats, 2 dogs → Entropy ≈ 0.18
- Right (whiskers=no): 2 cats, 48 dogs → Entropy ≈ 0.18

Weighted entropy after = 0.5×0.18 + 0.5×0.18 = 0.18
Information Gain = 1.0 - 0.18 = 0.82 (excellent split!)

---

#### 8.5.4 Pruning

**ID:** `pruning`
**Parent:** `8.5`

**Full Explanation:**
Pruning reduces tree complexity to prevent overfitting. Pre-pruning (early stopping): Limit depth, minimum samples per leaf, minimum impurity decrease during growth. Post-pruning: Grow full tree, then remove nodes that don't improve validation performance. Cost-complexity pruning (CCP) balances tree size against training error using a parameter α. Simpler trees generalize better.

**Simple Explanation:**
Cut back the tree to prevent memorization. A tree with a leaf for every training example is overfitting. Pruning removes unnecessary branches, keeping the tree simple and generalizable.

**Example:**
Unpruned tree:
```
           [root]
          /      \
       [A]        [B]
      / \         / \
    [C] [D]     [E] [F]
   / \   |     /|\   |
  1   2  3   4 5 6   7   <- 7 leaves, complex
```

After pruning:
```
           [root]
          /      \
       [A]        [B]
        |          |
      Class1    Class2   <- 2 leaves, simple
```

Pruned tree: Lower training accuracy but higher test accuracy.

---

#### 8.5.5 Feature Importance

**ID:** `feature-importance`
**Parent:** `8.5`

**Full Explanation:**
Feature importance in trees measures each feature's contribution to reducing impurity across all splits. Calculated as the total reduction in Gini/entropy weighted by the number of samples reaching each split. Features used higher in the tree (more samples affected) and causing larger purity gains rank higher. Useful for understanding which features drive predictions and for feature selection.

**Simple Explanation:**
Which features matter most? Features that appear early in the tree and create clean splits are important. Features that rarely appear or barely improve purity are less important. Gives insight into what drives predictions.

**Example:**
Decision tree for loan approval:
```
Feature Importances:
- Income: 0.45 (used at root, major split)
- Credit Score: 0.30 (second level, important)
- Age: 0.15 (lower level splits)
- Zip Code: 0.08 (rarely used)
- Eye Color: 0.02 (almost never helps)
```

Interpretation: Income and credit score drive most decisions.
Action: Could probably remove eye color feature entirely.
