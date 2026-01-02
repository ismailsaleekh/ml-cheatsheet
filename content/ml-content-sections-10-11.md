# ML Cheatsheet Content - Sections 10-11: Neural Networks & Deep Learning

## 10. NEURAL NETWORKS FOUNDATIONS

### 10.1 Building Blocks

---

#### 10.1.1 Artificial Neuron

**ID:** `artificial-neuron`
**Parent:** `10.1`

**Full Explanation:**
An artificial neuron (perceptron) computes a weighted sum of inputs plus a bias, then applies an activation function: output = f(Σwᵢxᵢ + b). It loosely mimics biological neurons: inputs are dendrites, weights are synapse strengths, activation is the firing threshold, output is the axon. Multiple neurons form layers, and multiple layers form neural networks capable of learning complex functions.

**Simple Explanation:**
A tiny decision-maker. Takes several inputs, multiplies each by a weight (importance), adds them up, then decides whether to "fire" based on the total. Many neurons working together can recognize patterns, classify images, and more.

**Example:**
Neuron deciding if someone will buy a product:
- Input 1: Age = 25, Weight = 0.5
- Input 2: Income = 50000, Weight = 0.0001
- Input 3: Past purchases = 10, Weight = 0.3
- Bias = -5

Weighted sum = 0.5×25 + 0.0001×50000 + 0.3×10 + (-5)
            = 12.5 + 5 + 3 - 5 = 15.5

Apply sigmoid: σ(15.5) = 0.999 → Will buy

---

#### 10.1.2 Perceptron

**ID:** `perceptron`
**Parent:** `10.1`

**Full Explanation:**
The perceptron is the simplest neural network: a single neuron with a step activation function. It classifies linearly separable data by finding a hyperplane. Learning rule: w ← w + η(y - ŷ)x for misclassified examples. Converges in finite steps for linearly separable data. Cannot learn XOR or other nonlinear functions—this limitation motivated multilayer networks.

**Simple Explanation:**
The original neural network: one neuron with a yes/no output. Can only draw a straight line to separate classes. Works great for simple problems, fails completely on complex ones like XOR. The starting point that led to deep learning.

**Example:**
AND gate with perceptron:
- Inputs: (0,0), (0,1), (1,0), (1,1)
- Outputs: 0, 0, 0, 1

Learned weights: w1=0.5, w2=0.5, bias=-0.7
- (0,0): 0.5×0 + 0.5×0 - 0.7 = -0.7 < 0 → Output 0 ✓
- (0,1): 0.5×0 + 0.5×1 - 0.7 = -0.2 < 0 → Output 0 ✓
- (1,0): 0.5×1 + 0.5×0 - 0.7 = -0.2 < 0 → Output 0 ✓
- (1,1): 0.5×1 + 0.5×1 - 0.7 = +0.3 > 0 → Output 1 ✓

---

#### 10.1.3 Multilayer Perceptron (MLP)

**ID:** `mlp`
**Parent:** `10.1`

**Full Explanation:**
A Multilayer Perceptron stacks multiple layers of neurons: input layer, one or more hidden layers, and output layer. Each layer is fully connected to the next. Hidden layers with nonlinear activations enable learning complex, nonlinear functions. Universal approximation theorem: MLPs with one hidden layer can approximate any continuous function given enough neurons. Trained via backpropagation.

**Simple Explanation:**
Stack multiple layers of neurons together. Input layer receives data, hidden layers learn patterns, output layer makes predictions. Adding hidden layers allows learning complex relationships that a single neuron cannot. The foundation of deep learning.

**Example:**
MLP for digit recognition (0-9):
```
Input Layer:  784 neurons (28×28 pixel image)
       ↓
Hidden Layer 1: 256 neurons (learns edges, curves)
       ↓
Hidden Layer 2: 128 neurons (learns parts of digits)
       ↓
Output Layer: 10 neurons (one per digit 0-9)
```

Total parameters: 784×256 + 256×128 + 128×10 = ~235K weights

---

#### 10.1.4 Feedforward Network

**ID:** `feedforward-network`
**Parent:** `10.1`

**Full Explanation:**
Feedforward networks process information in one direction: from input through hidden layers to output with no cycles. Each layer's output feeds only into subsequent layers. This acyclic structure enables efficient computation and training via backpropagation. Contrasts with recurrent networks that have feedback connections. MLPs, CNNs (for a single image), and transformers are feedforward.

**Simple Explanation:**
Information flows forward only, never backward or in loops. Input goes in, passes through layers, output comes out. No memory of previous inputs. Simple, efficient, and the basis for most neural networks.

**Example:**
Feedforward pass:
```
Input [x1, x2, x3]
       ↓
Hidden Layer 1: h1 = relu(W1·x + b1)
       ↓
Hidden Layer 2: h2 = relu(W2·h1 + b2)
       ↓
Output: y = softmax(W3·h2 + b3)
```

Each arrow goes forward. No connections going backward.

---

#### 10.1.5 Dense Layer (Fully Connected)

**ID:** `dense-layer`
**Parent:** `10.1`

**Full Explanation:**
A dense (fully connected) layer connects every neuron in the previous layer to every neuron in the current layer. For input dimension n and output dimension m, there are n×m weights plus m biases. Dense layers can learn arbitrary relationships between inputs and outputs. High parameter count makes them prone to overfitting and computationally expensive. Often used as final layers after convolutional or recurrent layers.

**Simple Explanation:**
Every input connects to every output. If the previous layer has 100 neurons and this layer has 50, there are 100×50 = 5000 connections. Flexible but expensive. Used when spatial structure isn't important.

**Example:**
Dense layer computation:
```
Input: [x1, x2, x3] (3 neurons)
Output: [y1, y2] (2 neurons)

y1 = σ(w11×x1 + w12×x2 + w13×x3 + b1)
y2 = σ(w21×x1 + w22×x2 + w23×x3 + b2)

Parameters: 3×2 weights + 2 biases = 8
```

```python
from tensorflow.keras.layers import Dense
model.add(Dense(64, activation='relu', input_dim=100))
# 100×64 + 64 = 6464 parameters
```

---

### 10.2 Activation Functions

---

#### 10.2.1 Activation Function

**ID:** `activation-function`
**Parent:** `10.2`

**Full Explanation:**
Activation functions introduce nonlinearity into neural networks. Without them, stacking linear layers would still be linear (composition of linear functions is linear). Activations enable networks to learn complex, nonlinear patterns. Properties to consider: nonlinearity, differentiability (for backprop), range (bounded vs unbounded), zero-centered outputs, and computational efficiency. Choice affects training dynamics and final performance.

**Simple Explanation:**
The "decision-maker" inside each neuron. Without activation functions, a 100-layer network would behave like a single layer. Activations add the nonlinearity that allows deep learning to work.

**Example:**
Without activation (linear):
Layer 1: y1 = W1·x
Layer 2: y2 = W2·y1 = W2·W1·x = W'·x
→ Still just a linear transformation!

With activation (ReLU):
Layer 1: y1 = ReLU(W1·x)
Layer 2: y2 = ReLU(W2·y1)
→ Nonlinear! Can learn complex patterns.

---

#### 10.2.2 ReLU (Rectified Linear Unit)

**ID:** `relu`
**Parent:** `10.2`

**Full Explanation:**
ReLU: f(x) = max(0, x). Outputs x for positive values, 0 for negative. Advantages: computationally efficient, mitigates vanishing gradient (gradient is 1 for positive inputs), induces sparsity. Drawback: "dying ReLU" problem—neurons with large negative weights may never activate again (gradient is 0 for negative inputs). Default choice for hidden layers in most architectures.

**Simple Explanation:**
If positive, output as-is. If negative, output zero. Super simple, super fast. Works great in practice. The default choice for most neural networks.

**Example:**
```
ReLU function:
Input:  -5   -2    0    2    5
Output:  0    0    0    2    5

Graph:
    |      /
  0 +-----/
    |    /
    ----|----
```

Dying ReLU problem:
Neuron with bias = -10 → Always negative input → Always 0 output
→ Gradient = 0 → Never updates → "Dead" neuron

---

#### 10.2.3 Leaky ReLU

**ID:** `leaky-relu`
**Parent:** `10.2`

**Full Explanation:**
Leaky ReLU: f(x) = x if x > 0, else αx (typically α = 0.01). Unlike ReLU, it has a small gradient for negative inputs, preventing dying neurons. The small negative slope keeps neurons "alive" and allows learning even when inputs are negative. Parametric ReLU (PReLU) learns α from data. Often performs similarly to ReLU but more robust.

**Simple Explanation:**
ReLU but with a small slope for negative values instead of flat zero. Negative inputs still produce a small output (and gradient), preventing dead neurons.

**Example:**
```
Leaky ReLU (α=0.01):
Input:  -100   -10    0    10   100
Output:  -1   -0.1    0    10   100

Graph:
      |       /
    0 +------/
      |     /
      ----/----  (slight slope for negatives)
```

No dying neurons: gradient is 0.01 for negatives, not 0.

---

#### 10.2.4 Tanh

**ID:** `tanh`
**Parent:** `10.2`

**Full Explanation:**
Tanh: f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ). Outputs range (-1, 1), zero-centered unlike sigmoid. Larger gradients than sigmoid (steeper), potentially faster learning. Suffers from vanishing gradients for large |x|. Often used in RNNs and when zero-centered outputs are beneficial. Related to sigmoid: tanh(x) = 2σ(2x) - 1.

**Simple Explanation:**
Squashes values to range -1 to 1. Zero-centered (unlike sigmoid's 0 to 1). Often used in recurrent networks. Gradient vanishes for very large or small inputs.

**Example:**
```
Tanh function:
Input:  -3    -1     0     1     3
Output: -0.99 -0.76  0   0.76  0.99

Graph:
   1 |        ****
     |     ***
   0 +---**-------
     | **
  -1 |**
```

---

#### 10.2.5 GELU

**ID:** `gelu`
**Parent:** `10.2`

**Full Explanation:**
GELU (Gaussian Error Linear Unit): f(x) = x × Φ(x), where Φ is the Gaussian CDF. Smoothly interpolates between 0 and x based on input value probability. Unlike ReLU's hard threshold, GELU provides smooth, probabilistic gating. Default activation in transformers (BERT, GPT). Slightly more expensive than ReLU but often performs better.

**Simple Explanation:**
A smooth version of ReLU that weights inputs by how positive they are (probabilistically). The activation used in transformers like BERT and GPT. Smoother than ReLU, often works better for language models.

**Example:**
```
GELU approximate:
Input:  -3    -1     0     1     3
Output: -0.04 -0.16  0   0.84  2.96

GELU ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

Comparison with ReLU:
- ReLU(-1) = 0
- GELU(-1) ≈ -0.16 (small negative output)
- Smoother transition around 0
```

---

#### 10.2.6 Swish

**ID:** `swish`
**Parent:** `10.2`

**Full Explanation:**
Swish: f(x) = x × σ(βx), where σ is sigmoid and β is learnable (often fixed at 1). Self-gated: uses the input itself to control how much passes through. Non-monotonic: slightly negative for some negative inputs before approaching 0. Outperforms ReLU on deeper networks. Used in EfficientNet and other modern architectures. SiLU (Sigmoid Linear Unit) is Swish with β=1.

**Simple Explanation:**
The input multiplied by its own sigmoid. Self-gated: large positive inputs pass through, negative inputs are suppressed but not hard zeroed. Smooth, works well in deep networks.

**Example:**
```
Swish (β=1):
Input:  -4    -2     0     2     4
Output: -0.07 -0.24  0   1.76  3.93

Swish(x) = x × sigmoid(x)
- Swish(2) = 2 × 0.88 = 1.76
- Swish(-2) = -2 × 0.12 = -0.24 (small negative)

Smooth, non-monotonic near 0.
```

---

### 10.3 Backpropagation

---

#### 10.3.1 Backpropagation

**ID:** `backpropagation`
**Parent:** `10.3`

**Full Explanation:**
Backpropagation efficiently computes gradients of the loss with respect to all weights using the chain rule. Forward pass: compute activations layer by layer. Backward pass: compute gradients from output to input, propagating error derivatives. For weight wᵢⱼ: ∂L/∂wᵢⱼ = ∂L/∂aⱼ × ∂aⱼ/∂zⱼ × ∂zⱼ/∂wᵢⱼ. Enables training deep networks with gradient descent.

**Simple Explanation:**
How neural networks learn. After making a prediction, figure out how much each weight contributed to the error. Start from the output, work backward through each layer, adjusting weights to reduce error. The chain rule from calculus makes this efficient.

**Example:**
Simple network: x → [w1] → h → [w2] → y
Target: t, Prediction: y, Loss: L = (y-t)²

Forward pass:
h = relu(w1 × x)
y = w2 × h

Backward pass:
∂L/∂y = 2(y-t)
∂L/∂w2 = ∂L/∂y × h
∂L/∂h = ∂L/∂y × w2
∂L/∂w1 = ∂L/∂h × relu'(w1×x) × x

Update: w1 -= lr × ∂L/∂w1, w2 -= lr × ∂L/∂w2

---

#### 10.3.2 Computational Graph

**ID:** `computational-graph`
**Parent:** `10.3`

**Full Explanation:**
A computational graph represents operations as nodes and data flow as edges. Forward mode: traverse graph input-to-output computing values. Reverse mode (backprop): traverse output-to-input computing gradients. Intermediate values are cached for gradient computation. Frameworks like PyTorch and TensorFlow build these graphs automatically, enabling automatic differentiation for any differentiable operation.

**Simple Explanation:**
A flowchart of math operations. Draw how inputs become outputs through operations. Forward: follow arrows to compute result. Backward: follow arrows in reverse to compute gradients. Frameworks build this automatically.

**Example:**
```
Expression: L = (wx + b - y)²

Computational graph:
x ─────┐
       ├─[×]──┐
w ─────┘      ├─[+]───┐
              │       ├─[-]───[²]──→ L
b ────────────┘       │
                      │
y ────────────────────┘

Forward: x=2, w=3, b=1, y=5
wx=6, wx+b=7, 7-5=2, 2²=4

Backward: dL/dw = 2(wx+b-y)×x = 2×2×2 = 8
```

---

#### 10.3.3 Vanishing Gradient

**ID:** `vanishing-gradient`
**Parent:** `10.3`

**Full Explanation:**
Vanishing gradients occur when gradients become exponentially small as they propagate backward through many layers. With sigmoid/tanh (gradients < 1), multiplying many small gradients → near-zero. Early layers receive tiny gradients, learning extremely slowly or not at all. Solutions: ReLU activation (gradient = 1), batch normalization, residual connections, proper initialization.

**Simple Explanation:**
Gradients shrink as they travel backward through layers. By the time they reach early layers, they're so small the network can't learn. Deep networks used to be impossible to train until solutions like ReLU and skip connections were developed.

**Example:**
100-layer network with sigmoid:
- Sigmoid gradient max ≈ 0.25
- Gradient at layer 1 ≈ 0.25¹⁰⁰ ≈ 10⁻⁶⁰
- Effectively zero—no learning!

With ReLU:
- ReLU gradient = 1 (for positive inputs)
- Gradient at layer 1 ≈ 1¹⁰⁰ = 1
- Full gradient signal—learning happens!

---

#### 10.3.4 Exploding Gradient

**ID:** `exploding-gradient`
**Parent:** `10.3`

**Full Explanation:**
Exploding gradients occur when gradients become exponentially large, causing unstable training (loss becomes NaN, weights overflow). Common in RNNs processing long sequences and deep networks with improper initialization. Solutions: gradient clipping (cap gradient magnitude), proper weight initialization (Xavier/He), batch normalization, LSTM/GRU for sequences.

**Simple Explanation:**
Gradients grow so large they break training. Weights update by huge amounts, oscillating wildly or overflowing to infinity. Common in recurrent networks. Fixed by clipping gradients to a maximum value.

**Example:**
Gradient explosion:
Step 1: weight = 1.0, gradient = 2.0, lr = 0.1
        weight = 1.0 - 0.1×2.0 = 0.8

Step 2: gradient = 20.0 (exploding!)
        weight = 0.8 - 0.1×20.0 = -1.2

Step 3: gradient = 200.0 (worse!)
        weight = -1.2 - 0.1×200.0 = -21.2

Gradient clipping (max=5):
Step 2: gradient = clip(20.0, 5) = 5.0
        weight = 0.8 - 0.1×5.0 = 0.3 ✓ Stable

---

### 10.4 Weight Initialization

---

#### 10.4.1 Weight Initialization

**ID:** `weight-initialization`
**Parent:** `10.4`

**Full Explanation:**
Weight initialization sets starting values for network parameters before training. Poor initialization causes vanishing/exploding gradients, slow convergence, or getting stuck in bad optima. Goals: maintain signal variance through layers, prevent saturation of activations, enable gradient flow. Modern initializations (Xavier, He) scale weights based on layer dimensions. Biases typically initialized to zero.

**Simple Explanation:**
How to set initial weights before training. Too small: signal dies out. Too large: activations saturate or explode. Just right: signal flows through all layers. The starting point matters a lot.

**Example:**
Bad initialization:
- Weights initialized to 0: All neurons compute same thing, never differentiate
- Weights initialized to 100: Activations saturate, gradients vanish

Good initialization (He):
- Weights ∼ N(0, 2/n_in)
- Maintains variance across layers
- Enables training of deep networks

---

#### 10.4.2 Xavier Initialization

**ID:** `xavier-initialization`
**Parent:** `10.4`

**Full Explanation:**
Xavier (Glorot) initialization scales weights by layer dimensions: W ∼ Uniform(-√(6/(n_in+n_out)), √(6/(n_in+n_out))) or W ∼ Normal(0, √(2/(n_in+n_out))). Designed for sigmoid/tanh activations to maintain variance of activations and gradients across layers. Key insight: both forward (activations) and backward (gradients) signals should preserve variance.

**Simple Explanation:**
Initialize weights based on how many neurons connect in and out. Smaller weights for layers with many connections. Keeps signal strength consistent as it flows through the network. Designed for sigmoid and tanh activations.

**Example:**
Layer with 784 inputs, 256 outputs:

Xavier uniform:
limit = √(6 / (784 + 256)) = √(6/1040) = 0.076
weights ∼ Uniform(-0.076, 0.076)

Xavier normal:
std = √(2 / (784 + 256)) = √(2/1040) = 0.044
weights ∼ Normal(0, 0.044)

```python
import torch.nn as nn
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

---

#### 10.4.3 He Initialization

**ID:** `he-initialization`
**Parent:** `10.4`

**Full Explanation:**
He (Kaiming) initialization is designed for ReLU activations: W ∼ Normal(0, √(2/n_in)). Accounts for ReLU zeroing half the values (factor of 2 vs Xavier). Maintains variance when using ReLU/Leaky ReLU. Essential for training very deep networks with ReLU. The default choice for modern networks using ReLU-family activations.

**Simple Explanation:**
Like Xavier but adjusted for ReLU. Since ReLU zeros out half the values, He init uses larger weights (√2 factor) to compensate. The go-to for ReLU networks.

**Example:**
Layer with 784 inputs, ReLU activation:

He normal:
std = √(2 / 784) = 0.0505
weights ∼ Normal(0, 0.0505)

He uniform:
limit = √(6 / 784) = 0.0875
weights ∼ Uniform(-0.0875, 0.0875)

```python
import torch.nn as nn
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

## 11. DEEP LEARNING ARCHITECTURES

### 11.1 CNNs (Convolutional Neural Networks)

---

#### 11.1.1 Convolutional Neural Network

**ID:** `cnn`
**Parent:** `11.1`

**Full Explanation:**
CNNs are specialized networks for grid-like data (images, time series). They use convolutional layers that apply learned filters to local regions, capturing spatial hierarchies. Key properties: local connectivity (each neuron sees only a region), weight sharing (same filter applied everywhere), and translation equivariance (patterns detected anywhere). Typically: Conv → Activation → Pool, repeated, then Dense layers for classification.

**Simple Explanation:**
Neural networks designed for images. Instead of connecting every pixel to every neuron (expensive!), use small filters that slide across the image detecting patterns. First layers find edges, deeper layers find shapes, deepest layers find objects.

**Example:**
```
Input: 224×224×3 image

Conv1: 64 filters, 3×3 → 224×224×64 (edges, colors)
Pool1: 2×2 → 112×112×64
Conv2: 128 filters, 3×3 → 112×112×128 (textures)
Pool2: 2×2 → 56×56×128
Conv3: 256 filters, 3×3 → 56×56×256 (parts)
Pool3: 2×2 → 28×28×256
...
Flatten: 7×7×512 → 25088
Dense: 25088 → 1000 (ImageNet classes)
```

---

#### 11.1.2 Convolution Operation

**ID:** `convolution`
**Parent:** `11.1`

**Full Explanation:**
Convolution slides a kernel (filter) across input, computing dot products at each position to produce an output feature map. For 2D: output[i,j] = Σₘ Σₙ input[i+m, j+n] × kernel[m,n]. Multiple kernels produce multiple channels. Learns edge detectors, texture patterns, and higher-level features. Kernel size, stride, and padding control output dimensions.

**Simple Explanation:**
A small filter slides across the image, multiplying and summing at each position. Different filters detect different patterns: one might find vertical edges, another horizontal edges. The network learns which filters are useful.

**Example:**
3×3 edge detection kernel:
```
[-1, 0, 1]
[-1, 0, 1]
[-1, 0, 1]

Apply to image patch:
[10, 10, 50]
[10, 10, 50]   → Sum of element-wise product
[10, 10, 50]     = -10-10-10 + 0 + 50+50+50 = 120

High value = vertical edge detected!
```

---

#### 11.1.3 Pooling Layer

**ID:** `pooling`
**Parent:** `11.1`

**Full Explanation:**
Pooling reduces spatial dimensions by aggregating values in local regions. Max pooling takes the maximum value; average pooling takes the mean. Reduces parameters and computation, provides translation invariance (small shifts don't change output), and enlarges receptive field. Typically 2×2 with stride 2, reducing each dimension by half. Global pooling reduces to single value per channel.

**Simple Explanation:**
Shrink the image by summarizing small regions. Max pooling: keep the largest value in each 2×2 block (preserves strongest features). Average pooling: keep the average (smoother). Makes the network faster and more robust to small shifts.

**Example:**
Max pooling 2×2:
```
Input 4×4:                Output 2×2:
[1, 3, 2, 4]
[5, 6, 7, 8]  →  [6,  8]
[2, 2, 1, 1]     [4,  3]
[3, 4, 2, 3]

Each 2×2 region → max value
Top-left: max(1,3,5,6) = 6
```

---

#### 11.1.4 Stride and Padding

**ID:** `stride-padding`
**Parent:** `11.1`

**Full Explanation:**
Stride is the step size of the convolution filter. Stride 1: move one pixel at a time. Stride 2: skip every other position, halving output size. Padding adds border pixels (usually zeros) to control output dimensions. "Same" padding: output size equals input size. "Valid" padding: no padding, output shrinks. Together they determine spatial dimensions of feature maps.

**Simple Explanation:**
Stride: how far the filter moves each step. Bigger stride = smaller output.
Padding: add border around the image. Keeps size constant and allows filters to see edge pixels.

**Example:**
Input: 6×6, Kernel: 3×3

Stride 1, no padding:
Output = (6-3)/1 + 1 = 4×4

Stride 2, no padding:
Output = (6-3)/2 + 1 = 2×2

Stride 1, padding 1:
Output = (6+2-3)/1 + 1 = 6×6 (same size!)

---

#### 11.1.5 Receptive Field

**ID:** `receptive-field`
**Parent:** `11.1`

**Full Explanation:**
The receptive field is the input region that affects a particular output neuron. Early layers have small receptive fields (see local patterns); deeper layers have larger receptive fields (see global patterns). Grows through convolutions and pooling. Larger receptive field = more context. Critical for understanding what scale of patterns a network can detect. Effective receptive field often smaller than theoretical.

**Simple Explanation:**
How much of the original image does one neuron "see"? First layer neurons see just a small patch. Deep layer neurons effectively see most of the image because they combine info from many earlier neurons.

**Example:**
```
Layer 1 (3×3 conv): Receptive field = 3×3
Layer 2 (3×3 conv): RF = 5×5 (each position sees 3×3 of layer 1, which sees 3×3)
Layer 3 (3×3 conv): RF = 7×7
+ 2×2 pooling: RF doubles

After several layers:
- A single neuron might "see" 128×128 of original image
- Can detect large objects, context, relationships
```

---

#### 11.1.6 Feature Map

**ID:** `feature-map`
**Parent:** `11.1`

**Full Explanation:**
A feature map is the output of applying one filter to the input, representing where a particular pattern is detected. Each convolutional layer produces multiple feature maps (one per filter), stacked as channels. Early maps show edges and colors; later maps show textures, parts, and objects. Feature maps are visualizable—useful for understanding what the network learns.

**Simple Explanation:**
The result of one filter scanning an image. High values where the pattern is found, low values elsewhere. Like a heat map of "where is this pattern?" Many filters = many feature maps = many different patterns detected.

**Example:**
```
Filter: Horizontal edge detector
Applied to image of a building:

Feature map:
[0.1, 0.2, 0.1]
[0.9, 0.9, 0.9]  ← Roof edge detected (high values)
[0.1, 0.2, 0.1]
[0.8, 0.8, 0.8]  ← Window ledge detected
[0.1, 0.1, 0.1]
```

---

### 11.2 RNNs (Recurrent Neural Networks)

---

#### 11.2.1 Recurrent Neural Network

**ID:** `rnn`
**Parent:** `11.2`

**Full Explanation:**
RNNs process sequential data by maintaining a hidden state that carries information across time steps. At each step: hₜ = f(hₜ₋₁, xₜ). This creates cycles in the computational graph, enabling memory of past inputs. Backpropagation through time (BPTT) unfolds the network across time for gradient computation. Struggles with long-term dependencies due to vanishing gradients.

**Simple Explanation:**
A network that remembers. Processes sequences one element at a time, keeping a "memory" (hidden state) that gets updated at each step. Good for text, audio, time series—anything where order matters.

**Example:**
Processing "hello":
```
h0 = 0 (initial state)
h1 = tanh(W_h×h0 + W_x×embed("h")) = [0.5, -0.2, ...]
h2 = tanh(W_h×h1 + W_x×embed("e")) = [0.3, 0.4, ...]
h3 = tanh(W_h×h2 + W_x×embed("l")) = [0.1, 0.6, ...]
h4 = tanh(W_h×h3 + W_x×embed("l")) = [0.2, 0.5, ...]
h5 = tanh(W_h×h4 + W_x×embed("o")) = [0.4, 0.3, ...]

h5 contains accumulated information about "hello"
```

---

#### 11.2.2 Hidden State

**ID:** `hidden-state`
**Parent:** `11.2`

**Full Explanation:**
The hidden state is an RNN's memory vector, updated at each time step and carrying information from past inputs. Dimension is a hyperparameter (typically 64-1024). Contains a compressed representation of the sequence seen so far. Can be used for classification (use final state), sequence generation (sample from state at each step), or encoding (state as sequence representation).

**Simple Explanation:**
The RNN's memory. A vector that summarizes everything it's seen so far. Gets updated with each new input. Can be used to predict the next word, classify the sequence, or encode it for another task.

**Example:**
Sentiment classification:
```
"This movie was amazing!"

h0 = [0, 0, 0]
h1 = update(h0, "This") = [0.1, 0.2, 0.1]
h2 = update(h1, "movie") = [0.2, 0.3, 0.0]
h3 = update(h2, "was") = [0.2, 0.2, 0.1]
h4 = update(h3, "amazing") = [0.8, -0.1, 0.5]
h5 = update(h4, "!") = [0.9, -0.2, 0.6]

Final hidden state h5 → Dense → sigmoid → 0.95 (positive sentiment)
```

---

#### 11.2.3 LSTM (Long Short-Term Memory)

**ID:** `lstm`
**Parent:** `11.2`

**Full Explanation:**
LSTM addresses vanishing gradients with a gated architecture. Cell state (Cₜ) carries long-term memory; hidden state (hₜ) carries short-term. Three gates control information flow: forget gate (what to remove from cell state), input gate (what to add to cell state), output gate (what to output). Gates use sigmoid (0-1) to control how much passes through. Enables learning dependencies over 100s of time steps.

**Simple Explanation:**
RNN with a better memory. Has gates that decide: what to forget, what to remember, what to output. The special "cell state" can carry important information unchanged across many time steps. Solves the vanishing gradient problem.

**Example:**
```
LSTM at time t:

Forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
"Should I forget any of the old memory?"

Input gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
Candidate: C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
"What new info should I store?"

Cell update: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
"Forget some old + add some new"

Output gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
Hidden: hₜ = oₜ ⊙ tanh(Cₜ)
"What part of memory to output?"
```

---

#### 11.2.4 GRU (Gated Recurrent Unit)

**ID:** `gru`
**Parent:** `11.2`

**Full Explanation:**
GRU simplifies LSTM with two gates instead of three, combining forget and input into an update gate. Reset gate (rₜ) controls how much past to ignore; update gate (zₜ) controls how much past state to keep vs new candidate. Fewer parameters than LSTM, faster training, similar performance on many tasks. Often preferred for smaller datasets or simpler sequences.

**Simple Explanation:**
A simpler LSTM. Two gates instead of three, no separate cell state. Reset gate: how much to ignore the past. Update gate: balance between old state and new state. Faster, lighter, often works just as well.

**Example:**
```
GRU at time t:

Reset gate: rₜ = σ(Wr·[hₜ₋₁, xₜ])
"How much to reset/ignore past?"

Update gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])
"How much to update with new info?"

Candidate: h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
"New hidden state candidate"

Hidden: hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
"Blend old and new based on update gate"
```

---

#### 11.2.5 Bidirectional RNN

**ID:** `bidirectional-rnn`
**Parent:** `11.2`

**Full Explanation:**
Bidirectional RNNs process sequences in both directions with two separate RNNs. Forward RNN reads left-to-right; backward RNN reads right-to-left. Outputs are concatenated, giving each position context from both past and future. Essential for tasks where future context matters (NER, translation). Doubles parameters and computation but often significantly improves accuracy.

**Simple Explanation:**
Run the sequence forward AND backward. Each position knows what came before AND what comes after. For understanding "bank" in "I deposited money at the bank," seeing future words helps disambiguate.

**Example:**
```
Sentence: "The cat sat on the mat"

Forward RNN:
→ The → cat → sat → on → the → mat →
h1f    h2f   h3f   h4f  h5f   h6f

Backward RNN:
← The ← cat ← sat ← on ← the ← mat ←
h1b    h2b   h3b   h4b  h5b   h6b

Combined for "sat":
[h3f || h3b] contains context from both directions
"sat" knows: subject is "cat" (before), action is on "mat" (after)
```

---

### 11.3 Attention Mechanisms

---

#### 11.3.1 Attention Mechanism

**ID:** `attention`
**Parent:** `11.3`

**Full Explanation:**
Attention computes a weighted combination of values based on query-key similarity. Given queries Q, keys K, and values V: Attention(Q,K,V) = softmax(QKᵀ/√d)V. Each query attends to all keys, producing attention weights that determine how much each value contributes to the output. Enables direct connections between any positions, bypassing sequential processing limitations of RNNs.

**Simple Explanation:**
Let the model focus on relevant parts. Like highlighting important words when reading. For each position, calculate how relevant every other position is, then take a weighted average. Enables direct long-range connections.

**Example:**
Translating "The cat sat on the mat":
When generating "le" (French for "the"):

Attention weights:
- "The": 0.8 (highly relevant!)
- "cat": 0.1
- "sat": 0.05
- "on": 0.02
- "the": 0.02
- "mat": 0.01

Output = 0.8×embed("The") + 0.1×embed("cat") + ...
Focus on "The" to generate "le"

---

#### 11.3.2 Self-Attention

**ID:** `self-attention`
**Parent:** `11.3`

**Full Explanation:**
Self-attention applies attention within a single sequence—each position attends to all positions in the same sequence. Q, K, V all come from the same input (projected with different learned matrices). Captures relationships between any positions regardless of distance. Foundation of transformers. Computational cost O(n²) in sequence length is a key limitation for very long sequences.

**Simple Explanation:**
Each word looks at every other word in the same sentence to understand context. "Bank" in "river bank" attends to "river" to understand its meaning. Every position can directly connect to every other position.

**Example:**
```
Sentence: "The animal didn't cross the road because it was too tired"

What does "it" refer to?

Self-attention for "it":
- "The": 0.05
- "animal": 0.7 ← High attention! "it" refers to "animal"
- "didn't": 0.02
- "cross": 0.03
- "road": 0.15
- "tired": 0.05

"it" attends strongly to "animal" → understands the reference
```

---

#### 11.3.3 Multi-Head Attention

**ID:** `multi-head-attention`
**Parent:** `11.3`

**Full Explanation:**
Multi-head attention runs multiple attention operations in parallel, each with different learned projections. Each "head" can focus on different aspects (e.g., one on syntax, another on semantics). Outputs are concatenated and projected. MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᵒ where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV). Typically 8-16 heads.

**Simple Explanation:**
Multiple attention patterns at once. One head might focus on nearby words, another on verbs, another on pronouns. Different perspectives combined give richer understanding than single attention.

**Example:**
```
8-head attention on "The cat sat on the mat":

Head 1: Focuses on subject-verb relationships
        "sat" → "cat" (subject)

Head 2: Focuses on positional relationships
        "on" → "mat" (location)

Head 3: Focuses on article-noun relationships
        "the" → "cat", "the" → "mat"

Head 4-8: Other patterns...

Concatenate all heads → Rich representation
```

---

### 11.4 Transformers

---

#### 11.4.1 Transformer Architecture

**ID:** `transformer`
**Parent:** `11.4`

**Full Explanation:**
Transformers use self-attention to process sequences in parallel, unlike sequential RNNs. Architecture: embedding + positional encoding → N encoder/decoder blocks → output. Each block has multi-head attention and feed-forward layers with residual connections and layer normalization. Enables massive parallelization, better long-range dependencies, and scales to billions of parameters. Foundation for GPT, BERT, and modern LLMs.

**Simple Explanation:**
A powerful architecture using attention instead of recurrence. Processes all positions simultaneously (fast!), directly connects any positions (good for long-range patterns), and scales to huge sizes. The basis of ChatGPT, BERT, and most modern AI.

**Example:**
```
Transformer Encoder Block:
Input
  ↓
[Multi-Head Self-Attention] ← All positions attend to all
  ↓ + Residual Connection
[Layer Norm]
  ↓
[Feed-Forward Network] ← Process each position independently
  ↓ + Residual Connection
[Layer Norm]
  ↓
Output

Stack 6-96 blocks for complete model.
```

---

#### 11.4.2 Positional Encoding

**ID:** `positional-encoding`
**Parent:** `11.4`

**Full Explanation:**
Since transformers have no inherent position sense (attention is permutation equivariant), positional encodings add position information. Original approach: sinusoidal functions at different frequencies. PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)). Learned position embeddings are an alternative. Enables the model to use position information despite parallel processing.

**Simple Explanation:**
Tell the model where each word is. Without positions, "dog bites man" = "man bites dog" to attention. Positional encoding adds a unique pattern to each position. Model learns to use these patterns to understand word order.

**Example:**
```
"The cat sat"

Without positional encoding:
embed("The"), embed("cat"), embed("sat") ← No position info!

With positional encoding:
embed("The") + pos_enc(0)
embed("cat") + pos_enc(1)
embed("sat") + pos_enc(2)

Position 0: [sin(0), cos(0), sin(0/ω), cos(0/ω), ...]
Position 1: [sin(1), cos(1), sin(1/ω), cos(1/ω), ...]

Each position has unique signature.
```

---

#### 11.4.3 Encoder-Decoder Architecture

**ID:** `encoder-decoder`
**Parent:** `11.4`

**Full Explanation:**
Encoder-decoder separates understanding input (encoder) from generating output (decoder). Encoder processes source sequence, creating representations. Decoder generates target sequence, attending to encoder outputs via cross-attention. Encoder is bidirectional (sees full input); decoder is unidirectional (only sees past outputs during generation). Used for sequence-to-sequence tasks: translation, summarization, question answering.

**Simple Explanation:**
Two parts: encoder reads and understands the input, decoder generates the output. Encoder sees the whole input at once. Decoder generates one token at a time, looking back at the input (via attention) and its own previous outputs.

**Example:**
```
Translation: "Hello world" → "Bonjour monde"

Encoder:
"Hello world" → [rich representations of each word]
                → Memory for decoder to attend to

Decoder (generating):
Step 1: [START] + attend to encoder → "Bonjour"
Step 2: "Bonjour" + attend to encoder → "monde"
Step 3: "Bonjour monde" + attend to encoder → [END]

Cross-attention: "Bonjour" attends to "Hello"
                 "monde" attends to "world"
```

---

#### 11.4.4 Decoder-Only Architecture

**ID:** `decoder-only`
**Parent:** `11.4`

**Full Explanation:**
Decoder-only transformers use only the decoder stack, trained for autoregressive language modeling. Each position attends only to previous positions (causal masking). Given context, predicts next token. Simpler architecture, scales well, and achieves strong performance on many tasks when large. GPT series, LLaMA, and most modern LLMs use this architecture. Prompt engineering replaces explicit encoder input.

**Simple Explanation:**
Just the decoder part, no encoder. Predicts the next word based on all previous words. The architecture behind ChatGPT. Can handle many tasks by framing them as text generation.

**Example:**
```
GPT-style generation:

Input: "The capital of France is"

Model attends causally:
Position 5 ("is") can attend to: "The", "capital", "of", "France", "is"
Cannot see future (masked)

Output distribution at position 5:
P(next = "Paris") = 0.95 ← Most likely
P(next = "Lyon") = 0.02
...

Generate: "Paris"
```

---

### 11.5 Efficient Transformers

---

#### 11.5.1 Sparse Attention

**ID:** `sparse-attention`
**Parent:** `11.5`

**Full Explanation:**
Sparse attention reduces O(n²) complexity by attending only to a subset of positions. Patterns include: local attention (nearby positions), strided attention (every k positions), random attention, or learned sparsity. Longformer combines local windows with global tokens. BigBird uses random + window + global. Enables processing longer sequences (thousands to millions of tokens) with manageable compute.

**Simple Explanation:**
Don't attend to everything—pick important positions. Nearby words, global summary tokens, and some random connections. Much faster than full attention, can handle much longer documents.

**Example:**
```
Full attention (1000 tokens):
1000 × 1000 = 1,000,000 attention computations

Sparse attention:
- Local window (128 tokens): 1000 × 128 = 128,000
- Global tokens (10): 1000 × 10 = 10,000
- Random (16 per position): 1000 × 16 = 16,000
Total: 154,000 ← 85% reduction!

Pattern for position 500:
Attends to: [436-564] (local) + [0,1,2...] (global) + random positions
```

---

#### 11.5.2 Linear Attention

**ID:** `linear-attention`
**Parent:** `11.5`

**Full Explanation:**
Linear attention approximates softmax attention with O(n) complexity. Key insight: rewrite attention as kernel function, decompose so computation is linear in sequence length. Methods include Linformer (low-rank projection), Performer (random features for softmax approximation), and Linear Transformer. Trade-off: slight accuracy loss for dramatic speed gains on long sequences.

**Simple Explanation:**
Make attention linear instead of quadratic. Use math tricks to avoid computing all n² pairs. Much faster for long sequences, with some quality trade-off.

**Example:**
```
Standard attention:
softmax(QKᵀ)V = O(n²d)

Linear attention (Performer):
Replace softmax with φ(Q)φ(K)ᵀ where φ is random feature map
φ(Q)(φ(K)ᵀV) = O(nd²)

For sequence length n=10000, d=64:
Standard: 100,000,000 × 64 = 6.4B ops
Linear: 10,000 × 64 × 64 = 41M ops ← 150× faster!
```

---

### 11.6 State Space Models

---

#### 11.6.1 State Space Model (SSM)

**ID:** `ssm`
**Parent:** `11.6`

**Full Explanation:**
State Space Models are a family of sequence models based on continuous-time systems: dx/dt = Ax + Bu, y = Cx + Du. Discretized for discrete sequences. Key advantage: can be computed as RNN (sequential) or convolution (parallel), getting benefits of both. Linear time complexity O(n), constant memory. Mamba and S4 are prominent examples achieving transformer-level performance with better efficiency.

**Simple Explanation:**
An alternative to transformers for sequences. Based on control theory mathematics. Can process sequences as efficiently as CNNs while handling long-range dependencies like RNNs. Linear time complexity—much faster than transformers for long sequences.

**Example:**
```
SSM computation modes:

1. Recurrent mode (inference):
   h₁ = Āh₀ + B̄x₁
   h₂ = Āh₁ + B̄x₂
   ... sequential, O(1) memory per step

2. Convolutional mode (training):
   K = [CB̄, CĀB̄, CĀ²B̄, ...]  ← Precompute kernel
   y = K * x  ← Single convolution, parallel

Same computation, different modes for different use cases.
```

---

#### 11.6.2 Mamba

**ID:** `mamba`
**Parent:** `11.6`

**Full Explanation:**
Mamba introduces selective state spaces with input-dependent parameters. Unlike fixed SSM matrices, Mamba's A, B, C parameters depend on the input, enabling content-aware reasoning similar to attention. Achieves transformer-quality results with O(n) time and memory. Hardware-optimized implementation enables 5× faster training and inference than transformers of similar size. Strong results on language modeling and long-context tasks.

**Simple Explanation:**
A selective state space model where the "memory" adapts based on what it's reading. Can decide what to remember and forget based on content, like attention but with linear complexity. Much faster than transformers for long sequences.

**Example:**
```
Standard SSM (fixed parameters):
A, B, C are constant → Same processing for all inputs

Mamba (selective, input-dependent):
A(x), B(x), C(x) depend on input x
"Interesting content → remember more"
"Boring content → forget faster"

Result: Content-aware long-range modeling
Speed: Linear in sequence length
Quality: Matches transformers on language modeling
```
