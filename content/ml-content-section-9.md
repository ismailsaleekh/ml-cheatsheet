# ML Cheatsheet Content - Section 9: Ensemble Methods

## 9. ENSEMBLE METHODS

### 9.1 Core Concepts

---

#### 9.1.1 Ensemble Learning

**ID:** `ensemble-learning`
**Parent:** `9.1`

**Full Explanation:**
Ensemble learning combines multiple models (base learners) to produce a better predictor than any single model. Key insight: diverse models make different errors, and combining them cancels out individual mistakes. Ensembles reduce variance (bagging), bias (boosting), or both. Success requires base models that are accurate (better than random) and diverse (make different errors).

**Simple Explanation:**
Combine many models to make a better one. Like asking 100 experts instead of just one—they'll make different mistakes, but the majority is usually right. The "wisdom of crowds" for machine learning.

**Example:**
Weather prediction:
- Model 1 (temperature-based): "Rain" (70% confidence)
- Model 2 (humidity-based): "Rain" (65% confidence)
- Model 3 (pressure-based): "No rain" (55% confidence)

Ensemble vote: Rain (2-1)
Ensemble probability: avg([0.7, 0.65, 0.45]) = 0.6 = "Rain"

Individual accuracy: ~75%
Ensemble accuracy: ~85% (errors cancel out)

---

#### 9.1.2 Wisdom of Crowds

**ID:** `wisdom-of-crowds`
**Parent:** `9.1`

**Full Explanation:**
The wisdom of crowds principle states that aggregate judgments from diverse, independent individuals often outperform expert opinions. In ML: combining diverse models reduces variance without increasing bias. Mathematical basis: if individual errors are uncorrelated, averaging N models reduces variance by factor of 1/N. Key requirements: diversity (different approaches) and independence (not correlated errors).

**Simple Explanation:**
Many guesses averaged together beat one expert guess. If 100 people guess how many jellybeans are in a jar, the average is usually very close to the truth, even though individuals are way off. Same principle for ML models.

**Example:**
Estimating a value (true = 100):
- Expert 1: 120 (error: +20)
- Expert 2: 85 (error: -15)
- Expert 3: 105 (error: +5)
- Expert 4: 90 (error: -10)

Average: (120+85+105+90)/4 = 100 ← Perfect!

Individual errors canceled out. This is why ensembles work.

---

#### 9.1.3 Base Learner

**ID:** `base-learner`
**Parent:** `9.1`

**Full Explanation:**
A base learner (weak learner) is an individual model in an ensemble. It should be better than random but doesn't need to be highly accurate. Common choices: decision trees (especially stumps for boosting), linear models, neural networks. Trade-off: complex base learners = fewer needed but risk overfitting; simple base learners = need more models but better regularization.

**Simple Explanation:**
The individual models that get combined. They don't have to be great on their own—just slightly better than guessing. Decision trees are popular base learners because they're fast and diverse (small changes in data create different trees).

**Example:**
Boosting typically uses "weak learners":
- Decision stump (depth=1): ~55% accuracy
- Single rule: "If age > 30, predict yes" ~52% accuracy

These weak learners are combined:
- 100 decision stumps combined: ~95% accuracy

Each weak learner adds a little information. Together = strong model.

---

#### 9.1.4 Model Diversity

**ID:** `model-diversity`
**Parent:** `9.1`

**Full Explanation:**
Diversity ensures ensemble members make different errors. Sources of diversity: (1) Different training data (bagging samples with replacement), (2) Different features (random subspace), (3) Different algorithms (heterogeneous ensemble), (4) Different hyperparameters, (5) Different random initializations. Diversity is as important as individual accuracy—100 identical models = no improvement.

**Simple Explanation:**
Ensemble members should disagree on different examples. If all models make the same mistakes, combining them doesn't help. Create diversity through different data, different features, or different algorithms.

**Example:**
Two scenarios with 3 models:

Low diversity (all similar):
- Model 1 wrong on examples: {1, 2, 3}
- Model 2 wrong on examples: {1, 2, 4}
- Model 3 wrong on examples: {1, 2, 5}
- Ensemble wrong on: {1, 2} (still wrong where they agree)

High diversity:
- Model 1 wrong on examples: {1, 5, 9}
- Model 2 wrong on examples: {2, 6, 10}
- Model 3 wrong on examples: {3, 7, 11}
- Ensemble wrong on: {} (no overlap in errors!)

---

### 9.2 Bagging Methods

---

#### 9.2.1 Bagging

**ID:** `bagging`
**Parent:** `9.2`

**Full Explanation:**
Bagging (Bootstrap Aggregating) trains each base model on a bootstrap sample (random sample with replacement) of the training data. For classification, final prediction uses majority voting; for regression, averaging. Bagging reduces variance without substantially increasing bias. Most effective with unstable models (high variance) like decision trees. Each bootstrap sample contains ~63.2% unique examples (rest are duplicates).

**Simple Explanation:**
Create different training sets by randomly sampling with replacement. Train a model on each sample, then average their predictions. Each model sees a slightly different version of the data, creating diversity.

**Example:**
Original data: [A, B, C, D, E] (5 examples)

Bootstrap sample 1: [A, A, C, D, E] (A appears twice, B missing)
Bootstrap sample 2: [B, B, C, D, E] (B appears twice, A missing)
Bootstrap sample 3: [A, C, C, E, E] (C, E duplicated)

Train 3 models on these different samples.
Each model learns slightly different patterns.
Average predictions → reduced variance.

---

#### 9.2.2 Bootstrap Sampling

**ID:** `bootstrap-sampling`
**Parent:** `9.2`

**Full Explanation:**
Bootstrap sampling creates a new dataset of size N by randomly drawing N samples with replacement from the original data. Each sample has equal probability 1/N of being selected at each draw. On average, each bootstrap sample contains 63.2% unique examples (1 - 1/e). The ~36.8% of examples not selected form the "out-of-bag" (OOB) set, useful for validation.

**Simple Explanation:**
Pick random examples, allowing repeats. With N examples, draw N times—some examples appear multiple times, some don't appear at all. This randomness creates different training sets for each model.

**Example:**
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Bootstrap sample (draw 10 with replacement):
[3, 7, 7, 1, 5, 3, 9, 10, 2, 7]

Unique examples included: {1, 2, 3, 5, 7, 9, 10} = 7 out of 10 (70%)
Out-of-bag (OOB): {4, 6, 8}

On average, ~63% unique, ~37% OOB.

---

#### 9.2.3 Out-of-Bag Error

**ID:** `oob-error`
**Parent:** `9.2`

**Full Explanation:**
Out-of-bag (OOB) error provides validation without a separate holdout set. For each training example, only ~1/3 of bagged models didn't see it during training. The OOB prediction for that example uses only these models. OOB error approximates test error closely, eliminating the need for cross-validation in bagging methods. Free validation during training.

**Simple Explanation:**
Each example is "out of bag" for some models (wasn't in their training sample). Use those models to predict for that example—it's like a test set they never saw. Do this for all examples to estimate generalization error.

**Example:**
Example X appears in training for models [1, 3, 5] but not [2, 4, 6].
For OOB error, predict X using only models [2, 4, 6].

OOB predictions:
- Model 2: Cat
- Model 4: Cat
- Model 6: Dog

OOB prediction for X: Cat (majority vote)
If X is actually Cat → correct
If X is actually Dog → counts as OOB error

---

#### 9.2.4 Random Forest

**ID:** `random-forest`
**Parent:** `9.2`

**Full Explanation:**
Random Forest extends bagging by adding random feature selection at each split. Each tree uses a bootstrap sample AND only considers a random subset of features (typically √p for classification, p/3 for regression) when splitting. This additional randomness creates even more diverse trees. Random Forest is robust, accurate, handles high dimensions, provides feature importance, and rarely overfits. Default ensemble choice for tabular data.

**Simple Explanation:**
Bagging + random feature selection. Each tree sees different data AND can only use a random subset of features at each split. Double randomness = more diversity = better ensemble. Often the first algorithm to try for structured data.

**Example:**
Building a random forest with 100 trees:

Tree 1:
- Bootstrap sample 1
- At root split: can only consider features [1, 4, 7, 9] (random 4 of 20)
- Picks feature 4

Tree 2:
- Bootstrap sample 2
- At root split: can only consider features [2, 5, 8, 11] (different random 4)
- Picks feature 8

Different data + different feature subsets = very different trees.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf.fit(X_train, y_train)
```

---

#### 9.2.5 Random Subspace Method

**ID:** `random-subspace`
**Parent:** `9.2`

**Full Explanation:**
Random subspace method trains each model on a random subset of features rather than random subset of samples. Each model sees all training examples but only some features. Creates diversity through feature space partitioning. Especially useful when features are plentiful but samples are limited. Can be combined with bagging for additional diversity (as in Random Forest).

**Simple Explanation:**
Each model only sees some features. Instead of hiding examples, hide columns. Model 1 might see [age, income], Model 2 sees [education, zipcode]. Each model specializes in different aspects.

**Example:**
Dataset with 10 features: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

Model 1 uses: [f1, f3, f5, f7]
Model 2 uses: [f2, f4, f6, f8]
Model 3 uses: [f1, f2, f9, f10]

Each model learns patterns in different feature subspaces.
Combine for robust predictions across all features.

---

#### 9.2.6 Extra Trees

**ID:** `extra-trees`
**Parent:** `9.2`

**Full Explanation:**
Extra Trees (Extremely Randomized Trees) adds more randomness than Random Forest: splits are chosen randomly rather than optimized. For each considered feature, a random split point is selected instead of the best one. This extreme randomness increases diversity and speed (no need to find optimal splits) at the cost of slightly higher bias. Often matches or beats Random Forest performance.

**Simple Explanation:**
Random Forest picks the best split from random features. Extra Trees picks a random split point too—even more randomness. Faster training, more diverse trees, surprisingly effective. Trade precision for speed and diversity.

**Example:**
At a split, considering feature "age" (range 20-60):

Random Forest:
- Try all possible splits: 21, 22, 23, ..., 59
- Find best: split at 35 (lowest Gini)
- Use split at 35

Extra Trees:
- Pick random threshold: 42 (just random between 20-60)
- Use split at 42 (no optimization)

Faster + more diverse. The "suboptimal" splits average out in the ensemble.

```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)
```

---

### 9.3 Boosting Methods

---

#### 9.3.1 Boosting

**ID:** `boosting`
**Parent:** `9.3`

**Full Explanation:**
Boosting builds models sequentially, each focusing on examples the previous models got wrong. Early models capture easy patterns; later models handle difficult cases. Predictions are weighted combinations of all models. Boosting reduces bias (and variance) by progressively correcting errors. More powerful than bagging but more prone to overfitting. Key algorithms: AdaBoost, Gradient Boosting, XGBoost.

**Simple Explanation:**
Train models one at a time, each fixing the previous one's mistakes. First model does its best. Second model focuses on what the first got wrong. Third model focuses on remaining errors. Keep going until errors are minimized.

**Example:**
Sequential error correction:

Model 1: Gets 100 examples right, 20 wrong
Model 2: Focuses on those 20, gets 15 right, 5 wrong
Model 3: Focuses on those 5, gets 4 right, 1 wrong

Combined: 100 + 15 + 4 = 119 right, 1 wrong
Much better than any single model!

---

#### 9.3.2 AdaBoost

**ID:** `adaboost`
**Parent:** `9.3`

**Full Explanation:**
AdaBoost (Adaptive Boosting) adjusts example weights based on classification accuracy. Initially, all examples have equal weight. After each weak learner, weights of misclassified examples increase; correctly classified examples decrease. Next learner focuses on hard examples. Final prediction: weighted vote where better-performing models have higher influence. Sensitive to noisy data and outliers.

**Simple Explanation:**
Give each example a weight. Misclassified examples get heavier (more important). Next model pays more attention to heavy examples. After many rounds, hard examples have been repeatedly targeted. Models that perform better get more voting power.

**Example:**
Round 1: All 100 examples weight = 1.0
- Model 1 trained, 80% accuracy
- 20 wrong examples: weight increases to 1.5
- 80 right examples: weight decreases to 0.9

Round 2: Focus on high-weight examples
- Model 2 trained on weighted data
- Gets 15 of those 20 hard examples right
- Update weights again...

Final: Combine Model 1 (weight 0.6) + Model 2 (weight 0.4) + ...

```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
ada.fit(X_train, y_train)
```

---

#### 9.3.3 Gradient Boosting

**ID:** `gradient-boosting`
**Parent:** `9.3`

**Full Explanation:**
Gradient Boosting fits new models to the residual errors (gradients) of the current ensemble. Each iteration: compute pseudo-residuals (negative gradients of loss), fit a model to predict residuals, add to ensemble with learning rate shrinkage. Works with any differentiable loss function. More flexible than AdaBoost. Learning rate controls each tree's contribution—smaller rates need more trees but often generalize better.

**Simple Explanation:**
Each new model predicts the remaining error. Model 1 predicts the target. Model 2 predicts what Model 1 got wrong (the residual). Model 3 predicts what's still wrong. Add all predictions together for the final answer.

**Example:**
Predicting house price = $300K:

Model 1 predicts: $280K (residual: +$20K)
Model 2 predicts residual: +$15K (remaining: +$5K)
Model 3 predicts remaining: +$4K (remaining: +$1K)

Final prediction: $280K + $15K + $4K = $299K (close!)

Each model fixes the previous error.

```python
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)
```

---

#### 9.3.4 XGBoost

**ID:** `xgboost`
**Parent:** `9.3`

**Full Explanation:**
XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting implementation with regularization, parallel processing, and advanced features. Includes L1/L2 regularization on weights, tree pruning, handling of missing values, and column subsampling. Uses second-order gradient information for better splits. Highly efficient, scalable, and consistently wins ML competitions. Industry standard for tabular data.

**Simple Explanation:**
Gradient boosting on steroids. Faster, handles missing data, prevents overfitting with regularization, works at scale. The go-to algorithm for structured data. Won countless Kaggle competitions.

**Example:**
XGBoost advantages:
- Speed: Parallel tree building, cache optimization
- Regularization: Built-in L1/L2 to prevent overfitting
- Missing values: Learns optimal direction for missing data
- Flexibility: Custom objectives and evaluation metrics

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    reg_lambda=1.0,  # L2 regularization
    reg_alpha=0.0    # L1 regularization
)
model.fit(X_train, y_train)
```

---

#### 9.3.5 LightGBM

**ID:** `lightgbm`
**Parent:** `9.3`

**Full Explanation:**
LightGBM uses histogram-based learning and leaf-wise tree growth for efficiency. Instead of scanning all data for splits, it bins continuous features into discrete histograms. Leaf-wise growth (best-first) creates more accurate trees with fewer iterations than level-wise growth. GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive Feature Bundling) further improve speed. Excellent for large datasets.

**Simple Explanation:**
Even faster gradient boosting for big data. Groups feature values into bins (histograms) instead of checking every value. Grows trees leaf-by-leaf (picking best leaf to split) rather than level-by-level. Often faster than XGBoost with similar accuracy.

**Example:**
LightGBM speed tricks:
- Histogram binning: 256 bins instead of millions of values
- Leaf-wise growth: Splits the most promising leaf, not all leaves
- GOSS: Keeps examples with large gradients, samples rest
- EFB: Bundles mutually exclusive features together

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    boosting_type='gbdt'
)
model.fit(X_train, y_train)
```

---

#### 9.3.6 CatBoost

**ID:** `catboost`
**Parent:** `9.3`

**Full Explanation:**
CatBoost specializes in categorical feature handling without manual encoding. Uses ordered target encoding to avoid target leakage: for each example, encodes categories using only examples that appear before it in a random permutation. Implements ordered boosting to reduce overfitting. Symmetric trees (same feature/threshold at each level) enable fast inference. Excellent for datasets with many categorical features.

**Simple Explanation:**
Gradient boosting that handles categories natively. No need to one-hot encode or manually encode categorical features—CatBoost does it smartly while avoiding data leakage. Great when you have many categorical columns.

**Example:**
Dataset with categories: [City, Color, Brand]

Traditional approach:
1. Encode City (100 categories → 100 columns)
2. Encode Color (10 categories → 10 columns)
3. Encode Brand (500 categories → 500 columns)
4. Train XGBoost

CatBoost approach:
1. Pass categorical columns directly
2. CatBoost handles encoding internally
3. No feature explosion, no leakage

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    cat_features=['City', 'Color', 'Brand']
)
model.fit(X_train, y_train)
```

---

#### 9.3.7 Learning Rate in Boosting

**ID:** `boosting-learning-rate`
**Parent:** `9.3`

**Full Explanation:**
Learning rate (shrinkage) scales each tree's contribution to the ensemble. Prediction: F_m(x) = F_{m-1}(x) + η × h_m(x), where η is the learning rate. Smaller η means each tree contributes less, requiring more trees but often improving generalization. Trade-off: smaller learning rate + more trees = slower training but often better accuracy. Typical values: 0.01-0.3.

**Simple Explanation:**
How much each new tree contributes. Learning rate 1.0 = full contribution. Learning rate 0.1 = only 10% contribution. Smaller rates are more conservative—need more trees but often generalize better. Like taking small steps instead of big jumps.

**Example:**
Predicting with learning rate η=0.1:

Iteration 1: Tree predicts +20, add 0.1×20 = +2
Iteration 2: Tree predicts +15, add 0.1×15 = +1.5
Iteration 3: Tree predicts +10, add 0.1×10 = +1

Total after 3 trees: 2 + 1.5 + 1 = 4.5

vs. Learning rate η=1.0:
Total after 3 trees: 20 + 15 + 10 = 45 (overshoots!)

Lower learning rate = slower convergence but more stable.

---

### 9.4 Stacking & Blending

---

#### 9.4.1 Stacking

**ID:** `stacking`
**Parent:** `9.4`

**Full Explanation:**
Stacking (Stacked Generalization) uses a meta-learner to combine base model predictions. Level 0: Train diverse base models. Level 1: Use base model predictions as features to train a meta-learner. The meta-learner learns optimal weighting and interactions between base predictions. Cross-validation is essential to avoid overfitting: base predictions on the training set must be out-of-fold predictions.

**Simple Explanation:**
Train a model to combine other models. Instead of averaging, learn the best way to combine. Maybe Random Forest is good for some examples, while SVM is better for others. The meta-learner figures out when to trust each base model.

**Example:**
Base models (Level 0):
- Random Forest
- XGBoost
- Neural Network
- Logistic Regression

For each training example, get out-of-fold predictions:
[RF_pred, XGB_pred, NN_pred, LR_pred] → new features

Meta-learner (Level 1):
Train a Logistic Regression on [RF_pred, XGB_pred, NN_pred, LR_pred]
Meta-learner learns: "Trust XGBoost when RF and NN disagree..."

```python
from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier()),
    ('svc', SVC(probability=True))
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

---

#### 9.4.2 Blending

**ID:** `blending`
**Parent:** `9.4`

**Full Explanation:**
Blending is a simpler variant of stacking that uses a holdout set instead of cross-validation. Split training data into two parts: train base models on part 1, generate predictions on part 2 (blend set), train meta-learner on blend set predictions. Faster than stacking (no cross-validation) but uses less data for base model training. Risk: base models never see blend set during training.

**Simple Explanation:**
Like stacking but simpler. Hold out some training data. Train base models on the rest. Make predictions on the holdout. Train a combiner on those predictions. Faster than full cross-validation stacking.

**Example:**
Training data: 10,000 examples

Split:
- Train set (8,000): Train base models
- Blend set (2,000): Generate base model predictions

Base model predictions on blend set:
[RF: 0.8, XGB: 0.7, SVC: 0.9] for each of 2,000 examples

Meta-learner:
Train on 2,000 examples with 3 features (base predictions)
Learns: Final = 0.3×RF + 0.5×XGB + 0.2×SVC

---

#### 9.4.3 Voting Ensemble

**ID:** `voting-ensemble`
**Parent:** `9.4`

**Full Explanation:**
Voting ensemble combines predictions through voting (classification) or averaging (regression). Hard voting: each model gets one vote, majority wins. Soft voting: average predicted probabilities, highest average probability wins. Soft voting often performs better as it leverages confidence information. Simple to implement, no training of meta-learner required. Works best with diverse, accurate base models.

**Simple Explanation:**
Democracy for models. Hard voting: each model votes for a class, majority wins. Soft voting: average the probability predictions. Simple, no training needed, often surprisingly effective.

**Example:**
Three models classify an image:

Hard voting:
- Random Forest: Cat
- SVM: Dog
- Neural Net: Cat
- Result: Cat (2-1)

Soft voting:
- Random Forest: [Cat: 0.8, Dog: 0.2]
- SVM: [Cat: 0.3, Dog: 0.7]
- Neural Net: [Cat: 0.9, Dog: 0.1]
- Average: [Cat: 0.67, Dog: 0.33]
- Result: Cat (higher probability)

```python
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('nn', nn)],
    voting='soft'  # or 'hard'
)
voting.fit(X_train, y_train)
```

---

#### 9.4.4 Weighted Averaging

**ID:** `weighted-averaging`
**Parent:** `9.4`

**Full Explanation:**
Weighted averaging assigns different weights to base models based on their quality. Unlike simple averaging, better models have more influence. Weights can be determined by: validation performance, optimization (grid search), or learned (regression with non-negativity constraints). For regression: ŷ = Σwᵢŷᵢ where Σwᵢ = 1. For classification: average weighted probabilities.

**Simple Explanation:**
Better models get more say. If Random Forest has 90% accuracy and SVM has 70%, Random Forest should count more. Assign weights proportional to performance or optimize them.

**Example:**
Three regression models with validation errors:
- Model A: RMSE = 10 (best)
- Model B: RMSE = 15
- Model C: RMSE = 20 (worst)

Weights from inverse error:
- w_A = 1/10 = 0.100
- w_B = 1/15 = 0.067
- w_C = 1/20 = 0.050

Normalized: w_A=0.46, w_B=0.31, w_C=0.23

Weighted prediction: 0.46×pred_A + 0.31×pred_B + 0.23×pred_C
Best model contributes most.
