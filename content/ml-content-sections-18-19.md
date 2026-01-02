# ML Cheatsheet Content - Sections 18-19: Time Series & Recommendations

## 18. TIME SERIES

### 18.1 Time Series Fundamentals

---

#### 18.1.1 Time Series

**ID:** `time-series`
**Parent:** `18.1`

**Full Explanation:**
A time series is a sequence of data points indexed in time order. Components: trend (long-term direction), seasonality (regular patterns), cycles (irregular patterns), noise (random variation). Stationarity: statistical properties constant over time. Tasks: forecasting (predict future), classification, anomaly detection, imputation. Key property: temporal dependency—observations are not independent.

**Simple Explanation:**
Data points ordered by time. Stock prices, temperatures, sales figures. The order matters—yesterday affects today, today affects tomorrow. Patterns like trends and seasonal cycles repeat.

**Example:**
```
Time series components:

Monthly ice cream sales:
Jan: 100, Feb: 120, Mar: 200, Apr: 300, May: 400, Jun: 500
Jul: 550, Aug: 520, Sep: 350, Oct: 200, Nov: 120, Dec: 100
[Next year repeats similarly]

Decomposition:
- Trend: Slight increase year-over-year (+5%/year)
- Seasonality: Summer peaks, winter troughs
- Noise: Random day-to-day variation

Time series vs regular data:
Regular: (x₁, y₁), (x₂, y₂) - independent
Time series: y₁ → y₂ → y₃ - sequentially dependent
```

---

#### 18.1.2 Stationarity

**ID:** `stationarity`
**Parent:** `18.1`

**Full Explanation:**
A stationary time series has constant statistical properties over time: constant mean, constant variance, and autocorrelation depending only on lag (not time). Many models assume stationarity. Non-stationary series can be transformed: differencing (remove trend), log transform (stabilize variance), seasonal differencing (remove seasonality). Tests: ADF (Augmented Dickey-Fuller), KPSS.

**Simple Explanation:**
The series behaves the same way at all times. Average doesn't drift, variability stays constant. Many models require stationarity. If not stationary, transform it (e.g., use differences instead of raw values).

**Example:**
```
Stationary vs Non-stationary:

Non-stationary (trending):
Time:  1   2   3   4   5   6
Value: 10  15  20  25  30  35
Mean keeps increasing!

After differencing (Δyₜ = yₜ - yₜ₋₁):
Diff:  5   5   5   5   5
Now constant mean = 5 ✓ Stationary

Non-stationary (increasing variance):
Time:  1   2   3   4   5
Value: 10  12  15  25  50
Variance explodes!

After log transform:
Log:   2.3 2.5 2.7 3.2 3.9
Variance stabilized ✓

ADF test:
p < 0.05 → Stationary
p > 0.05 → Non-stationary (difference needed)
```

---

#### 18.1.3 Autocorrelation

**ID:** `autocorrelation`
**Parent:** `18.1`

**Full Explanation:**
Autocorrelation measures correlation of a time series with lagged versions of itself. ACF (Autocorrelation Function): correlation at each lag. PACF (Partial Autocorrelation): correlation at lag k after removing effects of shorter lags. ACF/PACF plots guide model selection: AR models show PACF cutoff, MA models show ACF cutoff. High autocorrelation indicates predictability.

**Simple Explanation:**
How much does today relate to yesterday, last week, last year? High autocorrelation at lag 7 means weekly patterns. Helps choose which past values to use for prediction.

**Example:**
```
ACF analysis for daily sales:

Lag 1:  r = 0.85 (yesterday matters)
Lag 7:  r = 0.70 (weekly pattern!)
Lag 14: r = 0.65 (biweekly)
Lag 30: r = 0.50 (monthly)

ACF plot:
     |*
0.8  |****
     |*******
0.4  |***************
     |*******************
0.0  +--------------------------
     1  7  14  21  28  35  lag

Interpretation:
- Strong short-term correlation
- Weekly seasonal pattern (spike at lag 7)
- Use lag 1 and lag 7 as features
```

---

### 18.2 Classical Methods

---

#### 18.2.1 Moving Average

**ID:** `moving-average`
**Parent:** `18.2`

**Full Explanation:**
Moving average smooths time series by averaging nearby values. Simple Moving Average (SMA): mean of last n values. Weighted Moving Average (WMA): weighted mean, recent values weighted higher. Exponential Moving Average (EMA): exponentially decreasing weights into past. Smoothing removes noise, reveals trends. Also refers to MA(q) model in ARIMA context: current value as linear combination of past errors.

**Simple Explanation:**
Average recent values to smooth out noise and see the trend. 7-day moving average shows weekly trend without daily fluctuations. Used in finance, weather, and data visualization.

**Example:**
```
Moving average smoothing:

Daily values: 10, 12, 8, 15, 11, 13, 9, 14, 10, 12

3-day SMA:
Day 3: (10+12+8)/3 = 10.0
Day 4: (12+8+15)/3 = 11.7
Day 5: (8+15+11)/3 = 11.3
...

Result:
Raw:    10, 12, 8, 15, 11, 13, 9, 14, 10, 12
SMA-3:  -, -, 10, 11.7, 11.3, 13, 11, 12, 11, 12

Smoothed line shows trend, noise reduced!
```

---

#### 18.2.2 ARIMA

**ID:** `arima`
**Parent:** `18.2`

**Full Explanation:**
ARIMA (AutoRegressive Integrated Moving Average) combines three components: AR(p): linear combination of past p values; I(d): differencing d times for stationarity; MA(q): linear combination of past q errors. Notation: ARIMA(p,d,q). Model selection via ACF/PACF analysis or AIC/BIC. SARIMA adds seasonal components: ARIMA(p,d,q)(P,D,Q)s. Standard benchmark for time series forecasting.

**Simple Explanation:**
A powerful formula combining past values, differencing for stability, and past errors. ARIMA(1,1,1) uses yesterday's value, today's change, and yesterday's prediction error. The classic forecasting approach.

**Example:**
```
ARIMA(1,1,1) model:

Components:
AR(1): Uses previous value
I(1): First difference (removes trend)
MA(1): Uses previous error

Equation:
yₜ - yₜ₋₁ = φ(yₜ₋₁ - yₜ₋₂) + θεₜ₋₁ + εₜ

Forecast:
ŷₜ = yₜ₋₁ + φ(yₜ₋₁ - yₜ₋₂) + θεₜ₋₁

Python:
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order=(1,1,1))
results = model.fit()
forecast = results.forecast(steps=10)
```

---

#### 18.2.3 Exponential Smoothing

**ID:** `exponential-smoothing`
**Parent:** `18.2`

**Full Explanation:**
Exponential smoothing forecasts using weighted averages of past observations, with exponentially decreasing weights. Simple ES (level only), Holt's (level + trend), Holt-Winters (level + trend + seasonality). Smoothing parameters α (level), β (trend), γ (seasonality) between 0-1. ETS framework: Error, Trend, Seasonality (additive or multiplicative). Fast, interpretable, competitive with complex models.

**Simple Explanation:**
Smoothing where recent data matters most. Weights decay exponentially into the past. Three versions: simple (flat), Holt's (with trend), Holt-Winters (with trend and seasonality). Works surprisingly well.

**Example:**
```
Simple Exponential Smoothing (α=0.3):

Forecast: ŷₜ₊₁ = αyₜ + (1-α)ŷₜ

Data: 100, 110, 105, 120, 115

Step 1: ŷ₂ = 0.3(100) + 0.7(100) = 100
Step 2: ŷ₃ = 0.3(110) + 0.7(100) = 103
Step 3: ŷ₄ = 0.3(105) + 0.7(103) = 103.6
Step 4: ŷ₅ = 0.3(120) + 0.7(103.6) = 108.5
Step 5: ŷ₆ = 0.3(115) + 0.7(108.5) = 110.4

Holt-Winters adds trend and seasonal components.
```

---

### 18.3 Deep Learning for Time Series

---

#### 18.3.1 LSTM for Time Series

**ID:** `lstm-time-series`
**Parent:** `18.3`

**Full Explanation:**
LSTMs model time series by maintaining long-term memory across sequences. Input: sequence of past observations (window). Output: next value(s) or sequence. Handles variable-length sequences, learns complex patterns, captures long-range dependencies. Many-to-one for forecasting, many-to-many for sequence-to-sequence. Stacked LSTMs, bidirectional for feature extraction. Requires sequence padding, careful window selection.

**Simple Explanation:**
Use LSTM's memory to learn patterns in sequential data. Feed in the last N days, predict tomorrow. LSTM remembers important patterns from weeks ago. Popular before transformers, still useful.

**Example:**
```
LSTM for stock price prediction:

Input: Past 30 days of prices
Output: Next day's price

Architecture:
Input: (30, 5) - 30 days, 5 features (OHLCV)
LSTM: 50 units, return sequences
LSTM: 50 units
Dense: 1 (predicted price)

Training:
X: [[day1-30], [day2-31], [day3-32], ...]
y: [day31_price, day32_price, day33_price, ...]

Sliding window creates training samples.
```

---

#### 18.3.2 Temporal Convolutional Network (TCN)

**ID:** `tcn`
**Parent:** `18.3`

**Full Explanation:**
TCN applies causal convolutions to sequences, using dilated convolutions for long-range dependencies. Causal: no future leakage (convolution only sees past). Dilated: exponentially increasing gaps (1, 2, 4, 8...) capture different timescales efficiently. Residual connections for deep networks. Parallelizable (unlike RNNs), stable gradients, fixed receptive field. Often matches or beats RNNs on sequence tasks.

**Simple Explanation:**
CNNs for time series. Look only at past data (causal). Dilated convolutions see long history without huge kernels. Can train in parallel (faster than LSTM). Works great for many sequence tasks.

**Example:**
```
TCN with dilated convolutions:

Input sequence: [t1, t2, t3, t4, t5, t6, t7, t8]

Layer 1 (dilation=1):
Each output uses 2 adjacent inputs
[t1,t2]→o1, [t2,t3]→o2, [t3,t4]→o3...
Receptive field: 2

Layer 2 (dilation=2):
[o1,_,o3]→p1, [o2,_,o4]→p2...
Receptive field: 4

Layer 3 (dilation=4):
[p1,_,_,_,p5]→q1...
Receptive field: 8

Exponentially growing receptive field!
8 time steps with just 3 layers.
```

---

#### 18.3.3 Transformers for Time Series

**ID:** `transformers-time-series`
**Parent:** `18.3`

**Full Explanation:**
Transformers apply self-attention to time series, enabling direct modeling of long-range dependencies. Temporal embedding replaces positional encoding. Informer, Autoformer, FEDformer address long-horizon forecasting with sparse attention. PatchTST treats time windows as patches. Advantages: parallelizable, explicit dependency modeling. Challenges: quadratic complexity, need for modifications for long sequences.

**Simple Explanation:**
Self-attention for time series. Each time step can directly attend to any other time step—no sequential bottleneck. Modern architectures handle very long sequences efficiently. State-of-the-art for many forecasting benchmarks.

**Example:**
```
Transformer for forecasting:

Input: 96 historical time steps
Output: 24 future time steps

Architecture:
1. Embed time steps (value embedding + temporal embedding)
2. Transformer encoder (self-attention over history)
3. Transformer decoder (cross-attention to encoder)
4. Linear projection to predictions

Attention patterns:
- Self-attention finds similar patterns in history
- "Day 50 was similar to day 10, use that pattern"
- Long-range dependencies captured directly

Specialized variants:
- Informer: ProbSparse attention (O(n log n))
- Autoformer: Decomposition + Auto-correlation
- PatchTST: Patch-based, channel-independent
```

---

## 19. RECOMMENDATION SYSTEMS

### 19.1 Collaborative Filtering

---

#### 19.1.1 Collaborative Filtering

**ID:** `collaborative-filtering`
**Parent:** `19.1`

**Full Explanation:**
Collaborative filtering recommends items based on collective user behavior. User-based: find similar users, recommend what they liked. Item-based: find similar items to what user liked. Only uses interaction data (ratings, purchases), no content features needed. Challenges: cold start (new users/items), sparsity (most user-item pairs unknown), scalability. Foundation of recommendation systems (Netflix, Amazon).

**Simple Explanation:**
"Users who liked this also liked that." Find people with similar taste, recommend what they enjoyed. Or find items similar to what you liked. Uses only the pattern of who-bought-what.

**Example:**
```
User-based collaborative filtering:

User ratings matrix:
         Movie1  Movie2  Movie3  Movie4
Alice:     5       4       ?       ?
Bob:       5       5       4       ?
Carol:     2       1       5       5

Finding recommendation for Alice (Movie3):

1. Find similar users:
   Alice-Bob similarity: High (both like M1, M2)
   Alice-Carol similarity: Low (opposite preferences)

2. Bob rated Movie3 = 4

3. Predict: Alice would rate Movie3 ≈ 4

Item-based:
Movie1 and Movie2 are similar (same users like both)
Alice likes Movie1, Movie2
Find items similar to those → Recommend
```

---

#### 19.1.2 Matrix Factorization

**ID:** `matrix-factorization`
**Parent:** `19.1`

**Full Explanation:**
Matrix factorization decomposes the user-item rating matrix into low-rank user and item embeddings. R ≈ UV^T where U (users × k) and V (items × k). Each user/item represented by k-dimensional latent vector. Trained to minimize reconstruction error on observed ratings. SVD-based or gradient descent. Handles sparsity efficiently, learns interpretable factors (genres, preferences).

**Simple Explanation:**
Find hidden features that explain ratings. Users have preferences for hidden factors (action, romance). Items have amounts of those factors. Rating = dot product of user preferences and item factors. Few numbers per user/item, explains millions of ratings.

**Example:**
```
Matrix Factorization:

Ratings matrix (1000 users × 5000 movies):
Most entries unknown (sparse)

Factorize into:
User matrix U: 1000 × 50 (50 latent factors)
Movie matrix V: 5000 × 50

User 123's vector: [0.8, -0.2, 0.5, ...]
(Likes: action, not romance, moderate comedy)

Movie "Die Hard" vector: [0.9, -0.4, 0.3, ...]
(Is: action, not romance, bit of comedy)

Predicted rating:
User123 × DieHard = 0.8×0.9 + (-0.2)×(-0.4) + 0.5×0.3 + ...
                  = 0.72 + 0.08 + 0.15 + ... = 4.2

High predicted rating → Recommend!
```

---

#### 19.1.3 Implicit Feedback

**ID:** `implicit-feedback`
**Parent:** `19.1`

**Full Explanation:**
Implicit feedback infers preferences from user behavior (views, clicks, time spent) rather than explicit ratings. Challenges: no negative signal (not clicking ≠ dislike), varying confidence levels, noisy data. Approaches: treat interactions as positive examples, sample negatives, use confidence weighting. BPR (Bayesian Personalized Ranking) optimizes pairwise ranking. More data available than explicit ratings.

**Simple Explanation:**
Users don't rate—they click, buy, watch. Use this implicit signal. Clicked = probably interested. Didn't click = maybe not interested, or maybe never saw it. Need special handling since no explicit "dislike."

**Example:**
```
Implicit feedback signals:

User Alice's behavior:
- Viewed product A (5 times)
- Purchased product B
- Added product C to cart
- Clicked product D (once)
- Never interacted with E, F, G

Confidence weighting:
A: confidence = 5 (multiple views)
B: confidence = 10 (purchased!)
C: confidence = 3 (cart)
D: confidence = 1 (single click)
E, F, G: confidence = 0 (unknown)

Training:
Positive pairs: (Alice, A), (Alice, B), ...
Negative sampling: (Alice, E), (Alice, F)
but with lower confidence (might just be unseen)

Optimize: Prefer positives over negatives
```

---

### 19.2 Content-Based Filtering

---

#### 19.2.1 Content-Based Filtering

**ID:** `content-based-filtering`
**Parent:** `19.2`

**Full Explanation:**
Content-based filtering recommends items similar to what user previously liked, using item features. Build user profile from liked item features. New items scored by similarity to profile. No cold-start for items (features known immediately). Doesn't need other users' data. Limitations: limited novelty (similar items), feature engineering required, user cold start remains.

**Simple Explanation:**
"You liked action movies with Tom Cruise, here are more action movies with Tom Cruise." Uses item attributes (genre, actors, keywords) not other users' behavior. Recommends based on content similarity.

**Example:**
```
Content-based movie recommendation:

User profile (from liked movies):
Action: 0.8, Sci-Fi: 0.6, Romance: 0.1
Tom Cruise: 0.7, Keanu Reeves: 0.5
1990s: 0.4, 2000s: 0.6

New movie: "The Matrix" (1999)
Features: Action: 0.9, Sci-Fi: 1.0, Romance: 0.0
          Keanu Reeves: 1.0, 1990s: 1.0

Similarity:
= 0.8×0.9 + 0.6×1.0 + 0.1×0.0 + 0.5×1.0 + 0.4×1.0
= 0.72 + 0.6 + 0 + 0.5 + 0.4 = 2.22 (high!)

Recommend "The Matrix" ✓
```

---

#### 19.2.2 Hybrid Recommendation

**ID:** `hybrid-recommendation`
**Parent:** `19.2`

**Full Explanation:**
Hybrid systems combine collaborative and content-based methods to leverage strengths of both. Approaches: weighted (blend scores), switching (use one or other based on context), feature combination (use content features in collaborative model), cascade (one refines other's results). Addresses cold start (content works for new items), sparsity (collaborative captures user patterns). Modern systems are hybrid.

**Simple Explanation:**
Best of both worlds. Use content features AND user behavior patterns. New items? Use content similarity. Established items? Use collaborative filtering. Blend the predictions for better results.

**Example:**
```
Hybrid recommendation:

For User Alice, Item X:

Collaborative score:
Similar users rated X: 4.2 average

Content-based score:
X's features match Alice's profile: 0.75 similarity

Hybrid combination (weighted):
Score = 0.6 × CF_score + 0.4 × CB_score
      = 0.6 × 4.2 + 0.4 × 3.75
      = 2.52 + 1.50 = 4.02

For new item Y (no ratings):
CF_score = unknown
CB_score = 0.60

Use CB only: Score = 3.0

Switch based on data availability!
```

---

### 19.3 Deep Learning Recommendations

---

#### 19.3.1 Neural Collaborative Filtering (NCF)

**ID:** `ncf`
**Parent:** `19.3`

**Full Explanation:**
NCF replaces matrix factorization's dot product with neural networks. User and item embeddings fed through MLPs to model complex interactions. Generalizes MF (can learn dot product) but learns non-linear patterns. GMF (Generalized MF) + MLP combined in NeuMF. Handles implicit feedback naturally. More expressive than linear models, standard neural recommendation baseline.

**Simple Explanation:**
Matrix factorization with neural networks. Instead of just multiplying user and item vectors, run them through neural networks to capture complex patterns. More powerful than simple dot products.

**Example:**
```
Neural Collaborative Filtering architecture:

Input:
User ID → User Embedding (64 dims)
Item ID → Item Embedding (64 dims)

GMF Path:
Element-wise product: user ⊙ item → 64 dims

MLP Path:
Concatenate: [user; item] → 128 dims
Dense: 128 → 64 → 32 → 16

Combine:
[GMF_output; MLP_output] → 80 dims
Dense: 80 → 1 (rating/probability)

Training:
Binary cross-entropy (implicit: interact/no-interact)
or MSE (explicit ratings)
```

---

#### 19.3.2 Two-Tower Model

**ID:** `two-tower`
**Parent:** `19.3`

**Full Explanation:**
Two-tower (dual encoder) architecture encodes users and items separately into a shared embedding space. User tower: user features → user embedding. Item tower: item features → item embedding. Similarity computed as dot product or cosine. Key advantage: item embeddings can be precomputed and indexed for fast retrieval. Used at scale (YouTube, Google). Retrieval stage in two-stage systems.

**Simple Explanation:**
Two separate networks: one for users, one for items. Both output embeddings in the same space. Compare with dot product. Items can be pre-computed and stored—at prediction time, just compute user embedding and find nearest items. Super fast at scale.

**Example:**
```
Two-Tower for YouTube recommendations:

User Tower:
User features: [watch_history, search_terms, demographics, ...]
         ↓
     Deep network
         ↓
User embedding (256 dims)

Video Tower:
Video features: [title, description, channel, duration, ...]
         ↓
     Deep network
         ↓
Video embedding (256 dims)

Serving:
1. Precompute all video embeddings → Index (ANN)
2. User request → Compute user embedding
3. Approximate nearest neighbor search
4. Top-K videos in milliseconds!

Billions of videos, millions of users → Still fast
```

---

#### 19.3.3 Sequential Recommendation

**ID:** `sequential-recommendation`
**Parent:** `19.3`

**Full Explanation:**
Sequential recommendation models the order of user interactions to predict next item. User history as sequence, predict next interaction. Architectures: RNN/LSTM, self-attention (SASRec), Transformers (BERT4Rec). Captures temporal dynamics, session patterns, evolving preferences. Contrasts with static collaborative filtering. Important for e-commerce, streaming, news.

**Simple Explanation:**
Order matters! If you browsed shoes → socks → pants, what's next? Sequential models learn patterns in browsing sequences. Transformers attend to relevant past items to predict the next click.

**Example:**
```
SASRec (Self-Attentive Sequential Recommendation):

User history: [item1, item2, item3, item4, item5]

Self-attention:
Position embeddings + Item embeddings
Causal masking (can't see future)
Attention: Which past items matter for next?

item5 attends to:
- item4 (0.4) - recent, similar
- item3 (0.3) - related category
- item2 (0.1)
- item1 (0.2) - same brand

Prediction:
Weighted history → Predict item6 distribution
Recommend top-K from prediction

Learns: "After browsing cameras, users often buy memory cards"
```
