# ML Cheatsheet Content - Sections 20-22: Practical, MLOps & Ethics

## 20. PRACTICAL CONSIDERATIONS

### 20.1 Feature Engineering

---

#### 20.1.1 Feature Engineering

**ID:** `feature-engineering`
**Parent:** `20.1`

**Full Explanation:**
Feature engineering creates, transforms, and selects input features to improve model performance. Includes encoding categorical variables, scaling numerics, creating interactions, extracting from raw data (text, images, dates). Domain expertise crucial—knowing which features matter. Often more impactful than model choice. Deep learning reduces but doesn't eliminate need for feature engineering.

**Simple Explanation:**
Transform raw data into useful inputs for models. Create new features from existing ones. "Date of birth" becomes "age." "Address" becomes "distance to store." Good features make simple models work well.

**Example:**
```
Feature engineering for house price prediction:

Raw data:
- address, square_feet, bedrooms, listing_date

Engineered features:
- latitude, longitude (from address)
- distance_to_downtown
- price_per_sqft_neighborhood (aggregated)
- days_on_market (from listing_date)
- bedroom_to_sqft_ratio
- is_corner_lot (from property data)
- school_rating (external data)

Interactions:
- sqft × bedrooms
- location × age

These engineered features often matter more than raw data!
```

---

#### 20.1.2 Feature Scaling

**ID:** `feature-scaling`
**Parent:** `20.1`

**Full Explanation:**
Feature scaling normalizes feature ranges for algorithms sensitive to scale. Standardization (Z-score): (x - μ) / σ, resulting in mean=0, std=1. Min-Max: (x - min) / (max - min), scales to [0,1]. Robust scaling uses median and IQR (handles outliers). Required for gradient descent, distance-based methods, regularized models. Tree-based methods don't require scaling.

**Simple Explanation:**
Put all features on similar scales. Income (10K-1M) vs age (0-100) shouldn't have different importance just because of scale. Standardization makes all features have mean 0 and similar ranges.

**Example:**
```
Before scaling:
Age: [25, 30, 35, 40] (range 25-40)
Income: [50000, 80000, 120000, 200000] (range 50K-200K)

Standardization (Z-score):
Age: [-1.34, -0.45, 0.45, 1.34]
Income: [-0.98, -0.39, 0.20, 1.17]
Now comparable scales!

Min-Max:
Age: [0, 0.33, 0.67, 1.0]
Income: [0, 0.20, 0.47, 1.0]
All in [0,1] range.

Important: Fit scaler on training data only!
           Transform test with training parameters.
```

---

#### 20.1.3 Handling Missing Data

**ID:** `missing-data`
**Parent:** `20.1`

**Full Explanation:**
Missing data handling strategies depend on missingness mechanism. MCAR (Missing Completely at Random): safe to drop. MAR (Missing at Random): imputation appropriate. MNAR (Missing Not at Random): missingness is informative. Methods: deletion (listwise, pairwise), imputation (mean, median, mode, regression, KNN, MICE), indicator variables (missingness as feature). Missing data can itself be a signal.

**Simple Explanation:**
What to do when data has blanks? Delete rows (loses information), fill with average (simple), predict missing values (sophisticated), or add a "was_missing" feature. The best approach depends on why data is missing.

**Example:**
```
Handling missing data:

Original:
Age: [25, NaN, 35, NaN, 45]
Income: [50K, 80K, NaN, 120K, 150K]

Strategies:

1. Mean imputation:
   Age: [25, 35, 35, 35, 45] (mean=35)

2. Median imputation:
   More robust to outliers

3. Regression imputation:
   Predict missing Age from Income, other features

4. Missing indicator:
   Age: [25, 35, 35, 35, 45]
   Age_missing: [0, 1, 0, 1, 0]
   (Missingness might be informative!)

5. Multiple imputation (MICE):
   Generate several plausible values
   Account for imputation uncertainty
```

---

#### 20.1.4 Handling Imbalanced Data

**ID:** `imbalanced-data`
**Parent:** `20.1`

**Full Explanation:**
Imbalanced datasets have skewed class distributions (e.g., 99% negative, 1% positive). Models tend to predict majority class. Solutions: resampling (oversample minority: SMOTE, undersample majority), class weights (penalize majority class errors less), threshold adjustment, anomaly detection framing, appropriate metrics (F1, AUC-ROC, not accuracy). Common in fraud detection, medical diagnosis, manufacturing.

**Simple Explanation:**
When one class is rare (fraud, disease), models ignore it. Fixes: duplicate rare examples, remove common ones, or weight rare examples more. Use metrics that care about the rare class.

**Example:**
```
Imbalanced fraud detection:

Dataset:
99,000 legitimate transactions
1,000 fraud transactions (1%)

Problem:
Model predicts "not fraud" always → 99% accuracy!
But 0% fraud detection (useless)

Solutions:

1. Class weights:
   fraud weight = 99, normal weight = 1
   Misclassifying fraud costs 99× more

2. SMOTE (oversample):
   Generate synthetic fraud examples
   Balance to 50,000 / 50,000

3. Undersample majority:
   Random sample 1,000 normal transactions
   Now 1,000 / 1,000 balanced

4. Threshold adjustment:
   Lower threshold for fraud prediction
   P(fraud) > 0.1 instead of 0.5

5. Evaluation:
   Use Precision, Recall, F1, AUC-ROC
   NOT accuracy!
```

---

### 20.2 Hyperparameter Tuning

---

#### 20.2.1 Hyperparameter Tuning

**ID:** `hyperparameter-tuning`
**Parent:** `20.2`

**Full Explanation:**
Hyperparameters are settings configured before training (learning rate, regularization strength, tree depth). Tuning finds optimal values for validation performance. Methods: grid search (exhaustive), random search (often better), Bayesian optimization (intelligent exploration), Hyperband (early stopping). Use validation set or cross-validation for evaluation. Automated ML (AutoML) automates this process.

**Simple Explanation:**
Find the best settings for your model. Too small learning rate = slow. Too large = unstable. Try different values, see what works best on held-out data. Grid search tries all combinations; random search samples intelligently.

**Example:**
```
Hyperparameter tuning for Random Forest:

Hyperparameters:
- n_estimators: [50, 100, 200, 500]
- max_depth: [5, 10, 20, None]
- min_samples_split: [2, 5, 10]

Grid search: 4 × 4 × 3 = 48 combinations
Try all, pick best validation score

Random search: Sample 20 random combinations
Often finds good solution faster

Bayesian optimization:
1. Try some random configs
2. Build model of score vs hyperparameters
3. Choose next config that balances exploration/exploitation
4. Repeat, converge to optimum

Best found: n_estimators=200, max_depth=10, min_samples_split=5
Validation F1: 0.89
```

---

#### 20.2.2 Grid Search

**ID:** `grid-search`
**Parent:** `20.2`

**Full Explanation:**
Grid search exhaustively evaluates all combinations of specified hyperparameter values. Define grid of values for each hyperparameter. Evaluate each combination with cross-validation. Select best performing configuration. Pros: thorough, finds global optimum in grid. Cons: computationally expensive (exponential in hyperparameters), fixed resolution, doesn't adapt. Best for small search spaces.

**Simple Explanation:**
Try every combination. If you have 3 options for each of 3 hyperparameters, try all 27 combinations. Guaranteed to find the best in your grid. But expensive if grid is large.

**Example:**
```
Grid search example:

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': [0.1, 1, 10]
}

# 3 × 2 × 3 = 18 combinations
# × 5-fold CV = 90 model fits

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# {'C': 1, 'kernel': 'rbf', 'gamma': 0.1}

print(grid_search.best_score_)
# 0.92
```

---

#### 20.2.3 Random Search

**ID:** `random-search`
**Parent:** `20.2`

**Full Explanation:**
Random search samples hyperparameter configurations randomly from specified distributions. Given budget of N evaluations, samples N random configurations. More efficient than grid search: higher probability of finding good values in important dimensions (grid wastes effort on unimportant ones). Use continuous distributions for fine-grained search. Often matches grid search with fewer evaluations.

**Simple Explanation:**
Randomly try combinations instead of trying all. Surprisingly effective! If one hyperparameter matters most, random search explores it more thoroughly than grid search. Faster and often equally good.

**Example:**
```
Random search example:

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

param_distributions = {
    'C': loguniform(0.01, 100),        # Log-uniform: 0.01 to 100
    'gamma': loguniform(0.001, 10),    # Log-uniform: 0.001 to 10
    'kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=20,  # Only 20 tries (vs 18+ for grid)
    cv=5,
    scoring='f1'
)
random_search.fit(X_train, y_train)

# Often finds equally good or better solution!
print(random_search.best_params_)
# {'C': 2.35, 'gamma': 0.042, 'kernel': 'rbf'}
```

---

### 20.3 Model Debugging

---

#### 20.3.1 Error Analysis

**ID:** `error-analysis`
**Parent:** `20.3`

**Full Explanation:**
Error analysis systematically examines model mistakes to identify improvement opportunities. Categorize errors by type, find patterns, prioritize high-impact fixes. Confusion matrix shows class-wise performance. Look at individual misclassified examples. Slice performance by segments (demographics, categories). Informs feature engineering, data collection, model architecture decisions.

**Simple Explanation:**
Study your model's mistakes. Which examples does it get wrong? Are there patterns? Maybe it fails on long sentences, or blurry images. Understanding errors tells you how to improve.

**Example:**
```
Error analysis for sentiment classifier:

Overall accuracy: 85%

Confusion matrix:
              Predicted
              Pos   Neg
Actual Pos    900   100
       Neg    150   850

Examine false negatives (100):
- 40% contain sarcasm
- 30% have negation ("not bad" = positive)
- 20% are very short
- 10% other

Examine false positives (150):
- 50% have positive words in negative context
- 30% are complaints with polite language
- 20% other

Actions:
1. Add sarcasm detection features
2. Handle negation explicitly
3. Collect more short-text training data
```

---

#### 20.3.2 Learning Curves

**ID:** `learning-curves`
**Parent:** `20.3`

**Full Explanation:**
Learning curves plot performance vs training set size or training iterations. Training-validation gap indicates overfitting/underfitting. Both curves plateau high: good fit. Training high, validation low: overfitting (need regularization, more data, simpler model). Both plateau low: underfitting (need more features, complex model). Guides decisions on data collection and model complexity.

**Simple Explanation:**
Track performance as training progresses. Training score rising but validation flat? Overfitting. Both low? Model too simple. Learning curves tell you if you need more data, simpler model, or more complexity.

**Example:**
```
Learning curves interpretation:

1. Good fit:
   Training:    90% ───────────
   Validation:  88% ───────────
   Small gap, both high ✓

2. Overfitting:
   Training:    99% ───────────
   Validation:  75% ───────────
   Large gap! Regularize or get more data.

3. Underfitting:
   Training:    70% ───────────
   Validation:  68% ───────────
   Both low, small gap. Model too simple.

4. Need more data:
   Training:    95% ───────────
   Validation:  80% ────────────↗ (still rising)
   More data would help!

5. Data won't help:
   Training:    95% ───────────
   Validation:  85% ─────────── (flat)
   Validation plateaued. Need better model.
```

---

## 21. MLOPS & PRODUCTION

### 21.1 Model Deployment

---

#### 21.1.1 Model Deployment

**ID:** `model-deployment`
**Parent:** `21.1`

**Full Explanation:**
Model deployment makes trained models available for predictions. Deployment patterns: batch (scheduled predictions on datasets), real-time API (serve individual predictions), edge (deploy to devices). Considerations: latency requirements, throughput, model size, infrastructure costs. Formats: ONNX, TensorRT, SavedModel. Serving: TensorFlow Serving, TorchServe, Triton, FastAPI.

**Simple Explanation:**
Take your trained model and use it in the real world. Wrap it in an API, deploy to servers, respond to requests. Or run predictions on batches overnight. Or deploy to phones/sensors.

**Example:**
```
Deployment patterns:

1. Batch inference:
   Every night: Load model → Process all new data → Save results
   Use case: Email scoring, recommendation updates

2. Real-time API:
   User request → API → Model → Response (100ms)
   Use case: Fraud detection, chatbots

3. Edge deployment:
   Model on device (phone, IoT)
   Use case: Face unlock, voice assistant

FastAPI example:
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    prediction = model.predict([features])
    return {"prediction": prediction}
```
```

---

#### 21.1.2 Model Serving

**ID:** `model-serving`
**Parent:** `21.1`

**Full Explanation:**
Model serving infrastructure handles prediction requests at scale. Requirements: low latency, high throughput, model versioning, A/B testing, monitoring. Optimizations: batching requests, GPU utilization, model compression. Tools: TensorFlow Serving, TorchServe, Triton Inference Server, SageMaker. Containerization (Docker) enables consistent deployment. Load balancing distributes traffic.

**Simple Explanation:**
The system that runs your model in production. Handle many requests per second, keep latency low, update models safely. Usually involves specialized servers optimized for ML inference.

**Example:**
```
TensorFlow Serving:

# Save model
model.save('/models/my_model/1/')  # version 1

# Deploy with TF Serving (Docker)
docker run -p 8501:8501 \
  -v /models:/models \
  -e MODEL_NAME=my_model \
  tensorflow/serving

# Make prediction
curl -d '{"instances": [[1.0, 2.0, 3.0]]}' \
     -X POST http://localhost:8501/v1/models/my_model:predict

Features:
- Automatic batching
- GPU support
- Version management (/v1/, /v2/)
- Health checks
- Metrics (latency, throughput)
```

---

#### 21.1.3 Model Versioning

**ID:** `model-versioning`
**Parent:** `21.1`

**Full Explanation:**
Model versioning tracks model iterations, enabling rollback and comparison. Version: model code, hyperparameters, training data version, evaluation metrics, and artifacts. Tools: MLflow, DVC, Weights & Biases. Git for code, specialized tools for large files (models, data). Enables reproducibility, debugging, compliance. Blue-green deployment swaps versions safely.

**Simple Explanation:**
Keep track of every model version. Which data was used? What were the metrics? If the new model fails, roll back to the previous version. Like git but for ML models.

**Example:**
```
MLflow model versioning:

import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)

    # Train model
    model = train(...)

    # Log metrics
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("f1", 0.89)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# Track in registry
mlflow.register_model(
    "runs:/abc123/model",
    "production_model"
)

# Transition stages
mlflow.transition_model_version_stage(
    name="production_model",
    version=3,
    stage="Production"
)
```

---

### 21.2 Monitoring

---

#### 21.2.1 Model Monitoring

**ID:** `model-monitoring`
**Parent:** `21.2`

**Full Explanation:**
Model monitoring tracks production model performance over time. Monitor: prediction distributions, latency, error rates, feature distributions, ground truth (when available). Alerting on anomalies: prediction drift, data drift, performance degradation. Logging predictions for analysis. Dashboards for visualization. Enables proactive maintenance, triggering retraining when needed.

**Simple Explanation:**
Watch your model in production. Is it still accurate? Are inputs changing? Is it getting slower? Catch problems before they affect users. Like application monitoring but for ML.

**Example:**
```
Model monitoring dashboard:

Real-time metrics:
- Predictions/second: 150
- P99 latency: 45ms
- Error rate: 0.1%

Model performance (daily):
- Accuracy: 0.91 (baseline: 0.92) ⚠️
- False positive rate: 0.05 (up from 0.03) ⚠️

Data monitoring:
- Feature 'age' mean: 35.2 (baseline: 34.8) ✓
- Feature 'income' missing: 5% (baseline: 2%) ⚠️
- New category in 'region': "EU-West" detected ⚠️

Alerts:
[WARN] Accuracy dropped 1% - investigate
[WARN] Feature 'income' missing rate increased
[INFO] New category detected - may need retraining
```

---

#### 21.2.2 Data Drift

**ID:** `data-drift`
**Parent:** `21.2`

**Full Explanation:**
Data drift occurs when input data distribution changes from training distribution. Causes: seasonal changes, population shifts, external events, upstream data issues. Types: feature drift (input distribution change), label drift (target distribution change), concept drift (relationship between features and target changes). Detection: statistical tests (KS, chi-squared), distribution comparison. Response: retrain, update thresholds.

**Simple Explanation:**
The data your model sees changes over time. Trained on winter data, deployed in summer. Or user behavior shifts. If the model was built for different data, performance degrades. Detect drift early.

**Example:**
```
Data drift detection:

Training data (2023):
- Age distribution: mean=35, std=10
- Income: mean=$60K, std=$20K
- Category 'mobile': 60%

Production data (2024):
- Age distribution: mean=28, std=12 ⚠️ Drift!
- Income: mean=$58K, std=$22K ✓ OK
- Category 'mobile': 75% ⚠️ Drift!

KS test for age:
p-value = 0.001 (< 0.05) → Significant drift

Impact:
- Model trained on older users
- Now seeing younger users
- Model may underperform for new demographic

Actions:
1. Investigate: Why younger users?
2. Evaluate: Performance on new segment
3. Retrain: With recent data
```

---

#### 21.2.3 Model Retraining

**ID:** `model-retraining`
**Parent:** `21.2`

**Full Explanation:**
Model retraining updates models with new data to maintain performance. Triggers: scheduled (weekly, monthly), performance degradation, data drift detection, business requirements. Strategies: full retrain, incremental/online learning, fine-tuning. Validation before deployment: compare to baseline, A/B test. Automated pipelines (CI/CD for ML) enable frequent safe retraining.

**Simple Explanation:**
Models go stale—data changes, world changes. Retrain regularly or when performance drops. Use recent data, validate thoroughly, deploy carefully. Automate the pipeline for consistent updates.

**Example:**
```
Automated retraining pipeline:

Trigger: Weekly schedule OR drift alert OR metric drop

Pipeline:
1. Data preparation
   - Collect last 90 days of data
   - Join with labels
   - Feature engineering

2. Training
   - Train new model
   - Track with MLflow

3. Evaluation
   - Compare to current production model
   - Must beat baseline by >0.5%

4. Validation
   - Test on held-out recent data
   - Check for biases

5. Deployment
   - Shadow mode first (parallel predictions)
   - Gradual rollout (10% → 50% → 100%)
   - Automatic rollback if metrics drop

6. Monitoring
   - Watch new model performance
   - Compare to previous version
```

---

## 22. ML ETHICS & FAIRNESS

### 22.1 Bias & Fairness

---

#### 22.1.1 Algorithmic Bias

**ID:** `algorithmic-bias`
**Parent:** `22.1`

**Full Explanation:**
Algorithmic bias occurs when ML systems produce unfair outcomes for certain groups. Sources: biased training data (historical discrimination), biased labels (human prejudice), biased features (proxies for protected attributes), biased evaluation (not measuring on all groups). Types: disparate treatment (explicit use of protected attributes), disparate impact (neutral features causing unfair outcomes). Requires proactive mitigation.

**Simple Explanation:**
AI systems can be unfair, even accidentally. Trained on biased history, they repeat biases. Hiring algorithms that prefer men. Loan algorithms that reject minorities. Even without using "race" as a feature, other features can be proxies.

**Example:**
```
Bias sources:

Historical bias:
Training data: Past hiring decisions (90% male engineers hired)
Model learns: Prefer male candidates
Even without "gender" feature: prefers male-associated schools, activities

Measurement bias:
Training data: Arrest records (more policing in minority areas)
Model learns: Minority areas are "high crime"
Reality: More arrests ≠ more crime, just more policing

Representation bias:
Training data: Mostly light-skinned faces
Model: Worse performance on darker skin
Impact: Facial recognition fails for minorities

Aggregation bias:
One model for everyone
But relationship differs by group
Model good for majority, poor for minorities
```

---

#### 22.1.2 Fairness Metrics

**ID:** `fairness-metrics`
**Parent:** `22.1`

**Full Explanation:**
Fairness metrics quantify bias across groups. Demographic parity: equal positive prediction rates across groups. Equalized odds: equal TPR and FPR across groups. Predictive parity: equal precision across groups. Individual fairness: similar individuals treated similarly. These metrics can conflict—satisfying all simultaneously is often mathematically impossible. Choose metrics based on application context.

**Simple Explanation:**
How do you measure if a model is fair? Many definitions exist. Equal acceptance rates? Equal accuracy? Equal error rates? Different metrics capture different notions of fairness. Often can't satisfy all.

**Example:**
```
Fairness metrics for loan approval:

Groups: Group A (majority), Group B (minority)

Demographic Parity:
P(Approved | A) = P(Approved | B)
"Equal approval rates regardless of group"
A: 70% approved, B: 65% approved → Unfair

Equalized Odds:
P(Approved | Qualified, A) = P(Approved | Qualified, B)
P(Approved | Unqualified, A) = P(Approved | Unqualified, B)
"Equal TPR and FPR across groups"

Predictive Parity:
P(Repays | Approved, A) = P(Repays | Approved, B)
"Approval means same thing for both groups"

Trade-off:
Cannot have demographic parity AND predictive parity
if base rates differ (different qualification rates)
```

---

#### 22.1.3 Bias Mitigation

**ID:** `bias-mitigation`
**Parent:** `22.1`

**Full Explanation:**
Bias mitigation techniques reduce unfairness at different pipeline stages. Pre-processing: reweighting, resampling, transforming data. In-processing: constrained optimization, adversarial debiasing, fair representations. Post-processing: threshold adjustment per group, outcome calibration. Trade-offs: fairness vs accuracy, different fairness definitions. Requires iterative approach: measure, mitigate, evaluate.

**Simple Explanation:**
Fix bias at different stages. Before training: fix the data. During training: add fairness constraints. After training: adjust predictions. No perfect solution—requires careful trade-offs and ongoing monitoring.

**Example:**
```
Bias mitigation strategies:

Pre-processing:
- Reweighting: Increase minority sample weights
- Oversampling: Generate more minority examples
- Data augmentation: Balance representation

In-processing:
- Add fairness constraint to loss function:
  L = accuracy_loss + λ × fairness_penalty
- Adversarial debiasing:
  Learn features that don't predict protected attribute
- Fair representation learning:
  Encode inputs to be useful but fair

Post-processing:
- Separate thresholds per group:
  Group A: P > 0.6 → Approve
  Group B: P > 0.5 → Approve (different threshold)
- Calibrated to equalize positive rates

Evaluation:
- Check multiple fairness metrics
- Check accuracy for all groups
- Monitor in production
```

---

### 22.2 Responsible AI

---

#### 22.2.1 Explainability

**ID:** `explainability`
**Parent:** `22.2`

**Full Explanation:**
Explainability makes model predictions understandable. Global explanations: overall model behavior (feature importance, decision boundaries). Local explanations: individual prediction reasons (LIME, SHAP). Intrinsically interpretable models: linear regression, decision trees. Post-hoc explanations for complex models. Important for trust, debugging, compliance, and stakeholder communication.

**Simple Explanation:**
Why did the model make this prediction? Feature X was most important. This input pattern led to this output. Essential for trust, especially in high-stakes decisions like healthcare or lending.

**Example:**
```
SHAP explanation for loan denial:

Prediction: Denied (P=0.25 vs threshold 0.5)

Feature contributions:
Base rate: 0.50 (average approval probability)

Income: -0.15 (below average income)
Credit score: -0.08 (slightly low)
Employment: +0.05 (stable job)
Debt ratio: -0.12 (high debt)
Age: +0.05 (established credit history)

Final: 0.50 - 0.15 - 0.08 + 0.05 - 0.12 + 0.05 = 0.25

Explanation: "Denied primarily due to
below-average income (-0.15) and
high debt ratio (-0.12)"

Actionable: "Reduce debt or increase income to improve chances"
```

---

#### 22.2.2 Privacy in ML

**ID:** `privacy-ml`
**Parent:** `22.2`

**Full Explanation:**
Privacy-preserving ML protects sensitive data throughout the pipeline. Techniques: differential privacy (noise addition limiting individual exposure), federated learning (train on distributed data without centralizing), secure multi-party computation (compute on encrypted data), model privacy (prevent extraction of training data from models). Regulations: GDPR, CCPA. Trade-offs: privacy vs utility.

**Simple Explanation:**
Protect sensitive data used in ML. Don't memorize individual examples. Don't leak information through predictions. Train without seeing raw data. Important for healthcare, finance, and user data.

**Example:**
```
Privacy techniques:

1. Differential Privacy:
   Add noise to training or outputs
   Guarantee: Any individual's data has limited impact
   ε (epsilon): Privacy budget (lower = more private)

2. Federated Learning:
   Data stays on devices (phones)
   Only model updates sent to server
   Server never sees raw data

   Phone 1: Train local model → Send gradients
   Phone 2: Train local model → Send gradients
   Server: Aggregate gradients → Update global model

3. Membership Inference Protection:
   Attack: "Was this person in training data?"
   Defense: Regularization, differential privacy

4. Model extraction defense:
   Prevent copying model via query access
   Watermarking, query limits
```

---

#### 22.2.3 AI Safety

**ID:** `ai-safety`
**Parent:** `22.2`

**Full Explanation:**
AI safety ensures systems behave as intended without causing harm. Concerns: specification gaming (optimizing proxy instead of true objective), distributional shift (failure on unseen inputs), adversarial attacks (malicious inputs causing failures), emergent behaviors (unexpected capabilities at scale). Approaches: robustness testing, red-teaming, formal verification, RLHF alignment, interpretability research. Increasingly important as AI capabilities grow.

**Simple Explanation:**
Make sure AI does what we want, not just what we said. Systems can find loopholes, fail unexpectedly, or be manipulated. Test thoroughly, consider failure modes, design for safety.

**Example:**
```
AI safety concerns:

1. Specification Gaming:
   Goal: "Maximize user engagement"
   Learned: "Show outrage-inducing content"
   Problem: Optimized metric, not true intent

2. Distributional Shift:
   Training: Normal conditions
   Deployment: Edge cases, adversarial inputs
   Result: Unexpected failures

3. Adversarial Examples:
   Small perturbation to input
   → Completely wrong output
   Example: Sticker on stop sign → "Speed limit 45"

4. Emergent Capabilities:
   Large models develop unexpected abilities
   May include capabilities we didn't intend

Safety measures:
- Red-teaming: Attack your own system
- Robustness testing: Edge cases, stress tests
- Human oversight: Keep humans in the loop
- Alignment research: Ensure AI goals match human values
```

---

### 22.3 Regulatory Compliance

---

#### 22.3.1 GDPR and ML

**ID:** `gdpr-ml`
**Parent:** `22.3`

**Full Explanation:**
GDPR (General Data Protection Regulation) affects ML through data rights and automated decision requirements. Key provisions: right to explanation for automated decisions, right to contest, data minimization, purpose limitation, consent requirements, right to deletion (may require retraining). ML implications: document data usage, provide explanations, enable data deletion, justify automated decisions, privacy by design.

**Simple Explanation:**
European privacy law affects ML. You must explain automated decisions. Users can request their data be deleted. Can't use data for purposes they didn't consent to. Design systems with privacy in mind.

**Example:**
```
GDPR compliance for ML:

1. Automated Decision Making (Article 22):
   If decision has significant impact:
   - User can request human review
   - Must explain logic of decision
   → Maintain explainability, human oversight

2. Right to Erasure (Article 17):
   User requests data deletion
   → Retrain model without their data?
   → Machine unlearning techniques

3. Purpose Limitation (Article 5):
   Data collected for "customer service"
   → Cannot use for "credit scoring" without consent

4. Data Minimization:
   Only collect necessary features
   → Feature selection, not just accuracy

5. Documentation:
   - What data is used
   - Why decisions are made
   - How model works
   → Model cards, data sheets
```

---

#### 22.3.2 Model Documentation

**ID:** `model-documentation`
**Parent:** `22.3`

**Full Explanation:**
Model documentation records essential information for accountability and governance. Model cards: intended use, limitations, performance across groups, ethical considerations. Data sheets: data collection, preprocessing, known biases. Required for regulatory compliance, reproducibility, and responsible deployment. Standardized formats emerging (Google Model Cards, Microsoft Datasheets for Datasets).

**Simple Explanation:**
Document everything about your model. What is it for? What are its limits? How well does it work for different groups? Essential for responsible AI and often legally required.

**Example:**
```
Model Card Example:

MODEL DETAILS
- Name: Customer Churn Predictor v2.1
- Type: Gradient Boosted Classifier
- Date: 2024-01-15
- Owner: Data Science Team

INTENDED USE
- Predict customer churn probability
- Used by retention team for outreach
- NOT for: automated service cancellation

TRAINING DATA
- 500K customers, 2022-2023
- Demographics: US customers only
- Known bias: Underrepresents rural customers

PERFORMANCE
Overall accuracy: 0.85

| Group | Accuracy | FPR | FNR |
|-------|----------|-----|-----|
| Urban | 0.87 | 0.10 | 0.15 |
| Rural | 0.78 | 0.15 | 0.25 | ← Lower!

LIMITATIONS
- Less accurate for rural customers
- Performance degrades beyond 6 months
- Not validated for B2B customers

ETHICAL CONSIDERATIONS
- May reinforce existing retention disparities
- Rural customers may receive less outreach
- Recommend human review for edge cases
```
