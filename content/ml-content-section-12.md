# ML Cheatsheet Content - Section 12: Unsupervised Learning

## 12. UNSUPERVISED LEARNING

### 12.1 Clustering

---

#### 12.1.1 Clustering

**ID:** `clustering`
**Parent:** `12.1`

**Full Explanation:**
Clustering groups unlabeled data points into clusters where points within a cluster are more similar to each other than to points in other clusters. No ground truth labels exist—the algorithm discovers structure. Clustering can be partition-based (K-means), hierarchical (agglomerative), density-based (DBSCAN), or model-based (GMM). Evaluation metrics include silhouette score, Davies-Bouldin index, and domain-specific validation.

**Simple Explanation:**
Find natural groups in data without being told what groups exist. Customers might naturally cluster into "budget shoppers," "luxury buyers," etc. The algorithm discovers these groups from patterns in the data.

**Example:**
Customer data (no labels):
```
Customer purchase patterns plotted in 2D:

    $$$|           * *
       |          * * *    <- Luxury cluster
       |
       |   x x x
       |  x x x x          <- Mid-range cluster
       |
       |o o
       |o o o              <- Budget cluster
       +----------------
        Low      High
         Purchase Frequency
```

Clustering discovers 3 natural groups.

---

#### 12.1.2 K-Means Clustering

**ID:** `k-means`
**Parent:** `12.1`

**Full Explanation:**
K-Means partitions data into K clusters by minimizing within-cluster variance. Algorithm: (1) Initialize K centroids randomly, (2) Assign each point to nearest centroid, (3) Recompute centroids as cluster means, (4) Repeat until convergence. Finds spherical clusters of similar size. Requires specifying K beforehand. Sensitive to initialization—use K-means++ or run multiple times.

**Simple Explanation:**
Pick K cluster centers, assign each point to the nearest center, move centers to the middle of their points, repeat. Simple and fast. Works best when clusters are ball-shaped and similar size.

**Example:**
K=3 clustering:
```
Step 1: Random centroids: A, B, C

Step 2: Assign points to nearest centroid
        Points near A → Cluster 1
        Points near B → Cluster 2
        Points near C → Cluster 3

Step 3: Move centroids to cluster means
        A' = mean of Cluster 1
        B' = mean of Cluster 2
        C' = mean of Cluster 3

Step 4: Reassign points with new centroids
        Repeat until centroids stop moving
```

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
labels = kmeans.fit_predict(X)
```

---

#### 12.1.3 Elbow Method

**ID:** `elbow-method`
**Parent:** `12.1`

**Full Explanation:**
The elbow method helps select K for K-means by plotting within-cluster sum of squares (WCSS) against K. As K increases, WCSS decreases (more clusters = tighter fit). Look for an "elbow" where the rate of decrease sharply changes—additional clusters beyond this point provide diminishing returns. Subjective but widely used. Supplement with silhouette analysis or domain knowledge.

**Simple Explanation:**
Try different numbers of clusters, plot the "tightness" of clusters. Initially, adding clusters helps a lot. At some point, adding more doesn't help much—that's the elbow. Choose K at the elbow.

**Example:**
```
WCSS vs K:

WCSS |
     |*
     | *
     |  *
     |   *
     |    * <-- Elbow at K=4
     |     *  *  *  *
     +------------------
       1  2  3  4  5  6  7  8
                  K

K=4 is a good choice: significant drop before, minimal improvement after.
```

---

#### 12.1.4 Hierarchical Clustering

**ID:** `hierarchical-clustering`
**Parent:** `12.1`

**Full Explanation:**
Hierarchical clustering builds a tree (dendrogram) of clusters. Agglomerative (bottom-up): start with each point as its own cluster, repeatedly merge closest clusters. Divisive (top-down): start with one cluster, repeatedly split. Linkage methods define cluster distance: single (min), complete (max), average, Ward (minimizes variance). Cut dendrogram at desired level to get K clusters. No need to specify K upfront.

**Simple Explanation:**
Build a tree of clusters from bottom up. Start with each point alone. Merge the two closest clusters. Repeat until one cluster remains. Cut the tree at any level to get different numbers of clusters.

**Example:**
```
Agglomerative clustering of points A, B, C, D, E:

Step 1: {A}, {B}, {C}, {D}, {E} (5 clusters)
Step 2: Merge closest: {A,B}, {C}, {D}, {E} (4 clusters)
Step 3: Merge: {A,B}, {C,D}, {E} (3 clusters)
Step 4: Merge: {A,B,E}, {C,D} (2 clusters)
Step 5: Merge: {A,B,C,D,E} (1 cluster)

Dendrogram:
        ___|___
       |       |
    ___|___    |
   |       |   |
  _|_     _|_  |
 |   |   |   | |
 A   B   C   D E
```

---

#### 12.1.5 DBSCAN

**ID:** `dbscan`
**Parent:** `12.1`

**Full Explanation:**
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of arbitrary shape based on density. Parameters: ε (neighborhood radius) and minPts (minimum points to form dense region). Core points have ≥minPts neighbors within ε; border points are within ε of core points; noise points are neither. Doesn't require specifying K, handles noise naturally, finds non-spherical clusters.

**Simple Explanation:**
Find dense regions separated by sparse regions. A point is "core" if it has enough neighbors nearby. Clusters are connected dense regions. Points in sparse areas are labeled as noise. Great for finding oddly-shaped clusters and handling outliers.

**Example:**
```
DBSCAN with ε=1, minPts=3:

  * * *
   * *         <- Dense cluster (all core/border points)
    *

        * *
         *     <- Dense cluster

  *            <- Noise (isolated point)

Results:
- Cluster 1: Top group
- Cluster 2: Middle group
- Noise: Isolated point (not enough neighbors)
```

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)  # -1 indicates noise
```

---

#### 12.1.6 Gaussian Mixture Model

**ID:** `gmm`
**Parent:** `12.1`

**Full Explanation:**
GMM assumes data is generated from a mixture of K Gaussian distributions. Each Gaussian has its own mean μₖ and covariance Σₖ with mixing coefficient πₖ. Training via EM algorithm: E-step computes soft cluster assignments (probabilities), M-step updates parameters. Provides probabilistic cluster membership (soft clustering) and models elliptical clusters of different sizes/orientations.

**Simple Explanation:**
Assume data comes from K bell curves (Gaussians) overlapping. Find the best K bell curves to explain the data. Unlike K-means, gives probabilities (soft assignments) and handles elliptical clusters.

**Example:**
```
GMM with K=2:

Gaussian 1: mean=[0,0], cov=[[1,0],[0,1]]
Gaussian 2: mean=[3,3], cov=[[2,1],[1,2]]

For a new point x=[1,1]:
P(from Gaussian 1) = 0.7
P(from Gaussian 2) = 0.3
→ Soft assignment: mostly Gaussian 1

K-means would give hard assignment: Cluster 1 only.
GMM gives probabilities: 70% Cluster 1, 30% Cluster 2.
```

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
probabilities = gmm.predict_proba(X)  # Soft assignments
labels = gmm.predict(X)  # Hard assignments
```

---

### 12.2 Dimensionality Reduction

---

#### 12.2.1 Dimensionality Reduction

**ID:** `dimensionality-reduction`
**Parent:** `12.2`

**Full Explanation:**
Dimensionality reduction maps high-dimensional data to lower dimensions while preserving important structure. Motivations: visualization (2D/3D), noise reduction, computational efficiency, fighting curse of dimensionality. Linear methods (PCA, LDA) find linear subspaces; nonlinear methods (t-SNE, UMAP) can unfold complex manifolds. Trade-off between preserving global vs local structure.

**Simple Explanation:**
Reduce the number of features while keeping important information. 1000 features → 50 features. Helps visualization, speeds up training, and removes noise. Different methods preserve different aspects of the data.

**Example:**
```
Gene expression data: 20,000 genes × 100 patients

Too many dimensions to visualize or analyze!

After PCA → 50 dimensions:
- Captures 95% of variance
- Visualization possible (first 2-3 PCs)
- Clustering works better
- Model training 400× faster

After t-SNE → 2 dimensions:
- Beautiful visualization
- Cancer patients cluster separately
- Reveals subtypes invisible in raw data
```

---

#### 12.2.2 Principal Component Analysis (PCA)

**ID:** `pca`
**Parent:** `12.2`

**Full Explanation:**
PCA finds orthogonal directions (principal components) of maximum variance. First PC captures most variance, second PC captures most remaining variance orthogonal to first, and so on. Computed via eigendecomposition of covariance matrix or SVD. Linear, fast, deterministic. Preserves global structure but may miss local nonlinear patterns. Use explained variance ratio to choose dimensionality.

**Simple Explanation:**
Find the directions where data varies most. Project data onto these directions. First direction (PC1) is most important, then PC2, etc. Like finding the "main axes" of your data cloud.

**Example:**
```
2D data with strong correlation:

     *   *
   *   *   *
 *   *   *   *      PC1 (most variance)
   *   *   *        ─────────>
     *   *            /
                     /
                    / PC2 (orthogonal)

Project onto PC1 only: 2D → 1D
Keep most of the information (high variance direction)
Lose little information (low variance direction)
```

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_.sum())  # e.g., 0.95 = 95% variance kept
```

---

#### 12.2.3 t-SNE

**ID:** `tsne`
**Parent:** `12.2`

**Full Explanation:**
t-SNE (t-distributed Stochastic Neighbor Embedding) creates low-dimensional embeddings preserving local neighborhood structure. Converts point similarities to probabilities in high and low dimensions, minimizes KL divergence between them. Uses t-distribution in low-D to prevent crowding. Excellent for visualization (2D/3D). Non-deterministic, slow, not suitable for new data projection. Perplexity parameter controls local vs global emphasis.

**Simple Explanation:**
Arrange points in 2D so nearby points stay nearby, far points stay far. Optimizes a "similarity" match between original and reduced space. Amazing for visualization, reveals clusters and patterns. Don't use for actual dimensionality reduction—only visualization.

**Example:**
```
MNIST digits (784D) → t-SNE → 2D:

Original: 784-dimensional (28×28 pixels)
Cannot visualize!

After t-SNE:
  0 0        1 1
   0 0      1  1
    0 0    1    1

      2 2    3
     2  2  3 3
       2    3

Each digit class forms a cluster in 2D!
Beautiful visualization of 784D structure.
```

---

#### 12.2.4 UMAP

**ID:** `umap`
**Parent:** `12.2`

**Full Explanation:**
UMAP (Uniform Manifold Approximation and Projection) learns manifold structure using fuzzy topology. Faster than t-SNE, preserves more global structure, and can embed new data (has transform method). Uses cross-entropy loss instead of KL divergence. Parameters: n_neighbors (local vs global), min_dist (cluster tightness), metric (distance function). Widely used for visualization and dimensionality reduction.

**Simple Explanation:**
Like t-SNE but faster, preserves more global structure, and can apply to new data. The current state-of-the-art for visualization. Good for both visualization and actual dimensionality reduction.

**Example:**
```python
import umap

# Fast dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X)

# Can transform new data (unlike t-SNE)
new_embedding = reducer.transform(X_new)
```

UMAP vs t-SNE:
- Speed: UMAP 10-100× faster
- Global structure: UMAP preserves better
- New data: UMAP can transform, t-SNE cannot
- Quality: Both excellent for visualization

---

#### 12.2.5 Autoencoders for Dimensionality Reduction

**ID:** `autoencoder-dim-reduction`
**Parent:** `12.2`

**Full Explanation:**
Autoencoders learn compressed representations by training to reconstruct inputs. Encoder compresses input to bottleneck (latent space); decoder reconstructs from latent space. Bottleneck dimension is the reduced dimensionality. Nonlinear compression (unlike PCA). Variational autoencoders add probabilistic structure. Can handle very high-dimensional data (images, text) and learn complex manifolds.

**Simple Explanation:**
Train a neural network to compress and decompress data. The compression in the middle is your reduced representation. Learns whatever compression works best for reconstruction. Can capture nonlinear patterns that PCA misses.

**Example:**
```
Image autoencoder:

Input: 28×28 = 784 pixels
       ↓
Encoder: 784 → 512 → 256 → 128 → 32
       ↓
Bottleneck: 32 dimensions (compressed representation!)
       ↓
Decoder: 32 → 128 → 256 → 512 → 784
       ↓
Output: 28×28 = 784 pixels (reconstruction)

Loss: ||input - output||² (reconstruction error)

After training:
- Encoder gives 32D representation
- 32D captures essence of image
- 96% compression with good reconstruction
```

---

### 12.3 Density Estimation

---

#### 12.3.1 Density Estimation

**ID:** `density-estimation`
**Parent:** `12.3`

**Full Explanation:**
Density estimation models the probability distribution of data. Parametric methods assume a distribution form (Gaussian, GMM) and estimate parameters. Non-parametric methods (KDE, histograms) make minimal assumptions. Used for: understanding data distribution, anomaly detection (low-density = anomaly), sampling new data, and computing likelihoods. Foundation for many generative models.

**Simple Explanation:**
Learn the shape of your data distribution. Where is data concentrated? Where is it sparse? Useful for understanding data, detecting outliers, and generating new samples.

**Example:**
```
Customer transaction amounts:

Histogram (discrete):
Count |****
      |*********
      |**************
      |********
      |***
      +----------------
      $0    $50   $100

Continuous density estimate:
p(x) |    ___
     |   /   \
     |  /     \
     | /       \_
     |/          \__
     +----------------
      $0    $50   $100

P(transaction > $100) ≈ area under curve after $100
```

---

#### 12.3.2 Kernel Density Estimation (KDE)

**ID:** `kde`
**Parent:** `12.3`

**Full Explanation:**
KDE estimates density by placing a kernel (typically Gaussian) at each data point and summing. p(x) = (1/nh)Σᵢ K((x-xᵢ)/h), where h is bandwidth (smoothing parameter). Small h = spiky, overfit; large h = smooth, underfit. Non-parametric—no assumption about distribution shape. Choice of bandwidth is crucial; use cross-validation or rules like Silverman's.

**Simple Explanation:**
Put a small bump (Gaussian) at each data point, add them all up. More data points nearby = higher density. Bandwidth controls smoothness: small = bumpy, large = smooth.

**Example:**
```
Data points: [2, 3, 3.5, 10]

Small bandwidth (h=0.5):
      _   __
     / \ /  \
    /   v    \                   _
___/          \_________________/ \___
   2  3 3.5                     10

Large bandwidth (h=2):
       ____
      /    \
     /      \                 _
    /        \_______________/ \
___/                            \___
   2  3 3.5                     10

Medium bandwidth best: captures two-peak structure clearly.
```

---

### 12.4 Anomaly Detection

---

#### 12.4.1 Anomaly Detection

**ID:** `anomaly-detection`
**Parent:** `12.4`

**Full Explanation:**
Anomaly detection identifies rare observations that differ significantly from the majority. Approaches: statistical (z-score, IQR), density-based (low density = anomaly), distance-based (far from neighbors), reconstruction-based (high reconstruction error), and isolation-based (easy to isolate = anomaly). Applications: fraud detection, intrusion detection, equipment failure prediction, quality control.

**Simple Explanation:**
Find the weird ones. Anomalies are data points that don't fit the normal pattern. Could be fraud, equipment failure, or data errors. Different methods define "unusual" differently.

**Example:**
```
Credit card transactions:

Normal: $50 grocery, $30 gas, $15 coffee, $80 restaurant
Anomaly: $5000 electronics in foreign country at 3am

Detection methods:
1. Statistical: Spending way above mean (z-score > 3)
2. Density: Very few transactions with this pattern
3. Distance: Far from typical transaction clusters
4. Isolation: Very easy to separate from other transactions
```

---

#### 12.4.2 Isolation Forest

**ID:** `isolation-forest`
**Parent:** `12.4`

**Full Explanation:**
Isolation Forest detects anomalies by isolating points using random trees. Algorithm: randomly select feature and split value; anomalies require fewer splits to isolate (they're different from the crowd). Anomaly score based on average path length across many trees. Fast O(n log n), works well in high dimensions, no density estimation needed. Score interpretation: < 0.5 normal, > 0.5 anomaly.

**Simple Explanation:**
Try to isolate each point with random splits. Normal points are buried in clusters—need many splits. Anomalies are different—can be isolated quickly. Points that isolate easily are anomalies.

**Example:**
```
Random tree isolation:

Point A (normal, in cluster):
Split 1: Still with 50% of data
Split 2: Still with 25% of data
Split 3: Still with 10% of data
Split 4: Still with 5% of data
Split 5: Finally isolated!
→ Path length = 5 (normal)

Point B (anomaly, isolated):
Split 1: Isolated!
→ Path length = 1 (very short = anomaly!)
```

```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.01)  # Expect 1% anomalies
predictions = iso_forest.fit_predict(X)  # -1 = anomaly, 1 = normal
```

---

#### 12.4.3 One-Class SVM

**ID:** `one-class-svm`
**Parent:** `12.4`

**Full Explanation:**
One-Class SVM learns a boundary around normal data. Training uses only normal examples; the model finds a hyperplane (in kernel space) that separates normal data from the origin with maximum margin. New points falling outside the boundary are anomalies. Uses kernel trick for nonlinear boundaries. Parameter ν controls the fraction of outliers expected. Effective but computationally expensive for large datasets.

**Simple Explanation:**
Draw a boundary around normal data. Everything outside the boundary is an anomaly. Uses SVM techniques to find the best boundary. Train on normal examples only—no need for labeled anomalies.

**Example:**
```
Normal data forms a cluster:

    Boundary
   /--------\
  |  * * *   |
  | * * * *  |   x ← Anomaly (outside)
  |  * * *   |
   \--------/
              x ← Anomaly (outside)

Training: Learn tight boundary around normal points
Testing: Points outside boundary = anomalies
```

```python
from sklearn.svm import OneClassSVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.01)
oc_svm.fit(X_normal)  # Train on normal data only
predictions = oc_svm.predict(X_test)  # -1 = anomaly
```

---

#### 12.4.4 Autoencoder Anomaly Detection

**ID:** `autoencoder-anomaly`
**Parent:** `12.4`

**Full Explanation:**
Autoencoders detect anomalies via reconstruction error. Train on normal data; the model learns to compress and reconstruct normal patterns. Anomalies, being different from training data, have high reconstruction error. Threshold on reconstruction error determines anomaly classification. Works well for complex data (images, sequences). Threshold selection is crucial—use validation data or percentile-based approaches.

**Simple Explanation:**
Train autoencoder on normal data. It learns to reconstruct normal patterns well. Anomalies look different → can't reconstruct them well → high error = anomaly.

**Example:**
```
Autoencoder for network traffic:

Training: Normal traffic patterns
Model learns: "Normal traffic looks like this"

Testing:
Normal packet: input ≈ output, error = 0.01 ✓
Normal packet: input ≈ output, error = 0.02 ✓
Attack packet: input ≠ output, error = 0.85 ← Anomaly!

Threshold = 0.1:
Error < 0.1 → Normal
Error > 0.1 → Anomaly
```
