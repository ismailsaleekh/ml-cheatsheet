# ML Cheatsheet Content - Section 13: Generative Models

## 13. GENERATIVE MODELS

### 13.1 Autoencoders

---

#### 13.1.1 Autoencoder

**ID:** `autoencoder`
**Parent:** `13.1`

**Full Explanation:**
Autoencoders learn compressed representations by training to reconstruct inputs. Architecture: encoder f(x) → latent code z, decoder g(z) → reconstruction x̂. Loss: ||x - x̂||². The bottleneck forces compression, learning essential features. Not truly generative (can't sample new data) but foundation for VAEs. Applications: dimensionality reduction, denoising, pretraining, anomaly detection.

**Simple Explanation:**
Compress data, then decompress it back. The compressed form in the middle captures the essence. Like learning to describe images with few words, then recreating images from those descriptions.

**Example:**
```
Face autoencoder:

Input: 64×64×3 face image
       ↓
Encoder: Conv layers → 128D vector
       ↓
Latent: 128 numbers describing the face
        (pose, expression, lighting, identity)
       ↓
Decoder: Deconv layers → 64×64×3
       ↓
Output: Reconstructed face

The 128D latent space captures face attributes.
Similar faces have similar latent vectors.
```

---

#### 13.1.2 Variational Autoencoder (VAE)

**ID:** `vae`
**Parent:** `13.1`

**Full Explanation:**
VAE adds probabilistic structure to autoencoders. Encoder outputs distribution parameters (μ, σ) instead of a point; z is sampled from N(μ, σ²). Loss combines reconstruction and KL divergence regularization: L = ||x-x̂||² + KL(q(z|x) || p(z)). KL term keeps latent space organized (close to N(0,1)), enabling sampling and interpolation. Reparameterization trick: z = μ + σ⊙ε (ε~N(0,1)) enables backprop.

**Simple Explanation:**
Autoencoder where the compressed form is a probability distribution, not a fixed point. Adds randomness during training. Result: can sample from the latent space to generate new data. The first truly generative autoencoder.

**Example:**
```
VAE generation:

Training:
Face → Encoder → μ=[0.5, -0.3, ...], σ=[0.1, 0.2, ...]
Sample z ~ N(μ, σ²)
z → Decoder → Reconstructed face

Generating new faces:
Sample z ~ N(0, 1)  (random point in latent space)
z → Decoder → New face!

Interpolation:
z₁ = encode(face1), z₂ = encode(face2)
z_interp = 0.5×z₁ + 0.5×z₂
decode(z_interp) → face between face1 and face2
```

---

#### 13.1.3 Denoising Autoencoder

**ID:** `denoising-autoencoder`
**Parent:** `13.1`

**Full Explanation:**
Denoising autoencoders add noise to inputs and train to reconstruct clean originals. Noise can be Gaussian, dropout (masking), or salt-and-pepper. Forces the model to learn robust features that capture essential structure, not pixel-level details. Prevents learning identity function. Better representations than standard autoencoders. Foundation for denoising diffusion models.

**Simple Explanation:**
Add noise to input, try to reconstruct the clean version. Can't just memorize—must truly understand the data to remove noise. Learns more meaningful representations than regular autoencoders.

**Example:**
```
Denoising autoencoder training:

Original: Clean image of a cat
       ↓ Add noise
Corrupted: Fuzzy image with random dots
       ↓
Encoder → Latent
       ↓
Decoder
       ↓
Output: Should be clean cat!

Loss: Compare output to ORIGINAL clean image

The model learns:
- What cats look like (to denoise)
- Not just pixel copying (noise makes that fail)
```

---

### 13.2 GANs (Generative Adversarial Networks)

---

#### 13.2.1 Generative Adversarial Network

**ID:** `gan`
**Parent:** `13.2`

**Full Explanation:**
GANs train two networks adversarially: Generator G creates fake samples from noise; Discriminator D distinguishes real from fake. Training alternates: D learns to detect fakes, G learns to fool D. Game-theoretic equilibrium: G produces samples indistinguishable from real. min_G max_D E[log D(x)] + E[log(1-D(G(z)))]. Produces high-quality samples but training is notoriously unstable.

**Simple Explanation:**
Two networks compete: one creates fakes, one detects fakes. Generator gets better at making fakes; Discriminator gets better at detecting them. Eventually, Generator creates such good fakes that Discriminator can't tell the difference.

**Example:**
```
GAN for face generation:

Generator:
Random noise z ~ N(0,1) (100 numbers)
       ↓
Neural network
       ↓
Fake face image

Discriminator:
Face image (real or fake)
       ↓
Neural network
       ↓
P(real) between 0 and 1

Training loop:
1. D sees real faces → learns "this is real"
2. D sees G's fakes → learns "this is fake"
3. G adjusts to fool D → makes better fakes
4. Repeat until G makes convincing faces
```

---

#### 13.2.2 Generator Network

**ID:** `generator`
**Parent:** `13.2`

**Full Explanation:**
The generator maps random noise to data samples: G: z → x̂. Typically uses transposed convolutions (deconvolutions) for images, upsampling latent vectors to full resolution. Trained to maximize log(D(G(z)))—make Discriminator think fakes are real. Architecture choices: progressive growing, style injection (StyleGAN), conditional inputs. Generates increasingly realistic samples as training progresses.

**Simple Explanation:**
Creates fake data from random numbers. Starts with random noise, applies neural network transformations to produce images, audio, or text. Learns to produce outputs that fool the Discriminator.

**Example:**
```
Image Generator architecture:

z ~ N(0,1) [100 dims]
       ↓
Dense: 100 → 4×4×512
       ↓
DeConv: 4×4×512 → 8×8×256
       ↓
DeConv: 8×8×256 → 16×16×128
       ↓
DeConv: 16×16×128 → 32×32×64
       ↓
DeConv: 32×32×64 → 64×64×3
       ↓
Output: 64×64 RGB image

Random 100 numbers → Full image!
```

---

#### 13.2.3 Discriminator Network

**ID:** `discriminator`
**Parent:** `13.2`

**Full Explanation:**
The discriminator classifies inputs as real or fake: D: x → [0,1]. For images, typically uses convolutions to downsample. Trained on both real data (label 1) and generator outputs (label 0). Acts as a learned loss function—provides gradient signal to Generator about what makes samples unrealistic. Should not be too strong or too weak relative to Generator.

**Simple Explanation:**
A detective trying to spot fakes. Looks at samples and outputs probability of being real. Trained on both real data and Generator's fakes. Provides feedback to Generator about what's still fake-looking.

**Example:**
```
Image Discriminator architecture:

Input: 64×64×3 image
       ↓
Conv: 64×64×3 → 32×32×64
       ↓
Conv: 32×32×64 → 16×16×128
       ↓
Conv: 16×16×128 → 8×8×256
       ↓
Conv: 8×8×256 → 4×4×512
       ↓
Flatten + Dense → 1 (sigmoid)
       ↓
Output: P(real)

P(real) > 0.5 → Probably real
P(real) < 0.5 → Probably fake
```

---

#### 13.2.4 Mode Collapse

**ID:** `mode-collapse`
**Parent:** `13.2`

**Full Explanation:**
Mode collapse occurs when the Generator produces only a limited variety of samples, ignoring data distribution diversity. Generator finds a few outputs that reliably fool Discriminator and sticks with them. Example: face GAN generating only young white women, ignoring other demographics. Causes: Discriminator forgets earlier modes, Generator exploits Discriminator weaknesses. Solutions: minibatch discrimination, unrolled GANs, diversity-promoting regularization.

**Simple Explanation:**
Generator takes shortcuts—produces only a few types of samples that fool Discriminator. Like a counterfeiter who only makes $20 bills perfectly but can't make other denominations. Variety is lost.

**Example:**
```
MNIST digit GAN with mode collapse:

Expected: Generate all digits 0-9

Collapsed: Only generates 1s and 7s
          (found these fool D reliably)

Symptoms:
- Generated samples look similar
- Some categories never appear
- Discriminator accuracy on real varies by category

Solutions:
- Add minibatch statistics
- Penalize similar generated samples
- Use WGAN or other stable architectures
```

---

#### 13.2.5 Wasserstein GAN (WGAN)

**ID:** `wgan`
**Parent:** `13.2`

**Full Explanation:**
WGAN uses Wasserstein distance (Earth Mover's distance) instead of JS divergence, providing smoother gradients and more stable training. Critic (not discriminator) outputs unbounded scores; difference between real and fake scores is minimized/maximized. Requires Lipschitz constraint on Critic—achieved via weight clipping (original) or gradient penalty (WGAN-GP). More stable training, meaningful loss values, reduced mode collapse.

**Simple Explanation:**
A more stable GAN using a different way to measure how different fake and real distributions are. The Critic gives scores instead of probabilities. Training is smoother—loss actually correlates with sample quality.

**Example:**
```
WGAN-GP training:

Critic loss:
L_C = E[C(fake)] - E[C(real)] + λ·gradient_penalty

Generator loss:
L_G = -E[C(fake)]

Gradient penalty:
For interpolated samples x̂ = εx_real + (1-ε)x_fake
GP = (||∇C(x̂)||₂ - 1)²

Benefits:
- Loss correlates with sample quality
- More stable training
- Less mode collapse
- Can train critic more (5:1 ratio)
```

---

#### 13.2.6 StyleGAN

**ID:** `stylegan`
**Parent:** `13.2`

**Full Explanation:**
StyleGAN revolutionized face generation with style-based architecture. Latent z maps to intermediate space W via mapping network. W controls generator via adaptive instance normalization (AdaIN) at each layer. Coarse layers control pose/face shape, fine layers control textures/colors. Progressive growing during training. Enables unprecedented control over generated attributes and mixing of styles.

**Simple Explanation:**
Advanced GAN for high-quality face generation. Separates "style" (attributes like hair color, age) from "structure." Can mix styles from different latents—take face shape from one, hair from another. State-of-the-art face synthesis.

**Example:**
```
StyleGAN architecture:

z ~ N(0,1) → Mapping Network → w (style vector)
                                   ↓
Constant 4×4 input → Style injection → 4×4
                  → Style injection → 8×8
                  → Style injection → 16×16
                  ...
                  → Style injection → 1024×1024

Style mixing:
Take w₁ for layers 1-4 (coarse: face shape)
Take w₂ for layers 5-8 (medium: features)
Take w₃ for layers 9-12 (fine: texture)
→ Mix attributes from different sources!
```

---

### 13.3 Normalizing Flows

---

#### 13.3.1 Normalizing Flow

**ID:** `normalizing-flow`
**Parent:** `13.3`

**Full Explanation:**
Normalizing flows learn invertible transformations between simple distributions (Gaussian) and complex data distributions. Chain of bijective functions: x = f(z) = fₙ ∘ fₙ₋₁ ∘ ... ∘ f₁(z). Log-likelihood tractable via change of variables: log p(x) = log p(z) - Σ log|det(∂fᵢ/∂z)|. Unlike VAEs and GANs, provides exact likelihood. Requires invertible architectures with efficient Jacobian computation.

**Simple Explanation:**
Transform simple distribution to complex one through a chain of reversible steps. Can compute exact probability of any data point. Can generate new samples by sampling from simple distribution and transforming.

**Example:**
```
Normalizing flow:

Simple distribution (Gaussian):
z ~ N(0, 1)

Chain of transformations:
z → f₁ → h₁ → f₂ → h₂ → f₃ → x

Each fᵢ is:
- Invertible (can go backwards)
- Has tractable Jacobian determinant

Generation: z → f₁ → f₂ → f₃ → x
Density: log p(x) = log p(z) - sum of log|det(Jacobians)|
Inversion: x → f₃⁻¹ → f₂⁻¹ → f₁⁻¹ → z
```

---

### 13.4 Diffusion Models

---

#### 13.4.1 Diffusion Model

**ID:** `diffusion-model`
**Parent:** `13.4`

**Full Explanation:**
Diffusion models gradually add noise to data (forward process) and learn to reverse this (reverse process). Forward: x₀ → x₁ → ... → xₜ (pure noise) via Gaussian noise addition. Reverse: learn neural network to denoise xₜ → xₜ₋₁ → ... → x₀. Training: predict noise added at each step. Generation: start from noise, iteratively denoise. State-of-the-art image synthesis (DALL-E, Stable Diffusion, Midjourney).

**Simple Explanation:**
Slowly destroy data by adding noise, then learn to reverse the destruction. To generate: start with pure noise, gradually remove it to reveal a realistic sample. Like watching film grain accumulate, then learning to clean it up.

**Example:**
```
Diffusion process:

Forward (training, fixed):
Clean image → Add noise → Add noise → ... → Pure noise
    x₀     →    x₁     →    x₂    → ... →    xₜ

Reverse (learned):
Pure noise → Remove noise → Remove noise → ... → Clean image
    xₜ     →    xₜ₋₁     →    xₜ₋₂    → ... →    x₀

Training:
Given xₜ and t, predict the noise ε that was added
Loss = ||ε - ε_predicted||²

Generation:
1. Sample pure noise xₜ ~ N(0,1)
2. For t = T down to 0:
   - Predict noise using neural network
   - Remove predicted noise: xₜ₋₁ = denoise(xₜ)
3. Return x₀
```

---

#### 13.4.2 DDPM (Denoising Diffusion Probabilistic Model)

**ID:** `ddpm`
**Parent:** `13.4`

**Full Explanation:**
DDPM formalized diffusion models with rigorous probabilistic framework. Forward process: q(xₜ|xₜ₋₁) = N(√(1-βₜ)xₜ₋₁, βₜI). Reverse process: pθ(xₜ₋₁|xₜ) learned to approximate q(xₜ₋₁|xₜ,x₀). Training maximizes variational lower bound. Simple implementation: U-Net predicts noise, linear noise schedule. Produced high-quality samples rivaling GANs with more stable training.

**Simple Explanation:**
The foundational paper that made diffusion models work well. Defined exact math for noising/denoising. Uses a U-Net to predict noise at each step. Simpler and more stable to train than GANs.

**Example:**
```
DDPM specifics:

Noise schedule:
β₁ = 0.0001, β₂ = 0.0002, ..., β₁₀₀₀ = 0.02

Forward: xₜ = √(αₜ)x₀ + √(1-αₜ)ε
where αₜ = ∏(1-βᵢ)

Neural network εθ predicts noise:
Input: (xₜ, t)
Output: Predicted noise ε̂

Loss: ||ε - ε̂||² (simple MSE!)

Generation (1000 steps):
Start with pure noise
Denoise step by step using εθ
```

---

#### 13.4.3 Stable Diffusion

**ID:** `stable-diffusion`
**Parent:** `13.4`

**Full Explanation:**
Stable Diffusion operates in learned latent space rather than pixel space, dramatically reducing computation. VAE encodes images to latent (4× smaller), diffusion happens in latent space, VAE decodes back. Cross-attention conditions on text embeddings (from CLIP). Open-source with various fine-tuned models. Enables high-resolution generation (512×512+) on consumer hardware. Supports text-to-image, image-to-image, inpainting.

**Simple Explanation:**
Diffusion in a compressed space instead of pixel space—much faster! Use a text description to guide what image is generated. Open source and runs on regular GPUs. Powers many image generation tools.

**Example:**
```
Stable Diffusion pipeline:

Text-to-Image:
"A cat wearing a hat, digital art"
       ↓
CLIP text encoder → text embedding
       ↓
Start with latent noise (64×64×4)
       ↓
Iterative denoising with text conditioning
(U-Net with cross-attention to text)
       ↓
Denoised latent (64×64×4)
       ↓
VAE decoder
       ↓
Final image (512×512×3)

50 denoising steps in latent space
= 512×512 image in seconds on GPU
```

---

### 13.5 Language Models

---

#### 13.5.1 Language Model

**ID:** `language-model`
**Parent:** `13.5`

**Full Explanation:**
Language models estimate probability distributions over sequences of tokens: P(w₁, w₂, ..., wₙ) = ∏P(wᵢ|w₁,...,wᵢ₋₁). Autoregressive models predict next token given previous tokens. Used for text generation, scoring fluency, feature extraction. Traditional: n-gram models. Neural: RNN LMs, Transformer LMs. Large-scale transformers (GPT) achieve remarkable generalization across tasks.

**Simple Explanation:**
Predict the next word given previous words. "The cat sat on the ___" → "mat" (high probability). Trained on massive text, learns grammar, facts, and reasoning patterns. Foundation of ChatGPT and similar AI.

**Example:**
```
Language model prediction:

Input: "The quick brown fox"
Model outputs probability for each word in vocabulary:

P("jumps") = 0.15  ← Most likely
P("runs") = 0.08
P("is") = 0.05
P("the") = 0.001
...

Sample or take argmax:
"The quick brown fox jumps"

Continue:
"The quick brown fox jumps over" → "the"
"The quick brown fox jumps over the" → "lazy"
...
```

---

#### 13.5.2 Perplexity

**ID:** `perplexity`
**Parent:** `13.5`

**Full Explanation:**
Perplexity measures how well a language model predicts a held-out test set. PPL = exp(-(1/N)Σ log P(wᵢ|context)). Interpretation: effective number of equally likely words the model is uncertain between. Lower is better—perplexity of 10 means ~10 equally likely choices per word on average. Used to compare language models. Related to cross-entropy: PPL = 2^H.

**Simple Explanation:**
How surprised is the model by the text? Lower perplexity = less surprised = better predictions. If perplexity is 100, it's like the model is confused between 100 equally likely words. Good models have low perplexity.

**Example:**
```
Test sentence: "The cat sat on the mat"

Good model:
P(cat|The) = 0.1
P(sat|The cat) = 0.2
P(on|The cat sat) = 0.3
P(the|...) = 0.4
P(mat|...) = 0.2

PPL = exp(-1/6 × (log(0.1)+log(0.2)+...)) ≈ 15

Bad model:
Each probability ≈ 0.0001
PPL ≈ 10,000

Interpretation: Good model like choosing from 15 words
              Bad model like choosing from 10,000 words
```

---

### 13.6 Large Language Models

---

#### 13.6.1 Large Language Model (LLM)

**ID:** `llm`
**Parent:** `13.6`

**Full Explanation:**
LLMs are language models with billions of parameters trained on massive text corpora. Scale enables emergent capabilities: in-context learning, reasoning, code generation, instruction following. Key innovations: transformer architecture, scaled training (compute optimal laws), RLHF for alignment. Examples: GPT-4, Claude, LLaMA, PaLM. Foundation models adaptable to many tasks via prompting or fine-tuning.

**Simple Explanation:**
Very large neural networks trained on internet-scale text. At sufficient size, they learn to follow instructions, reason about problems, and perform tasks they weren't explicitly trained for. ChatGPT, Claude, and similar AI assistants are LLMs.

**Example:**
```
LLM scale evolution:

GPT-2 (2019): 1.5B parameters
GPT-3 (2020): 175B parameters
GPT-4 (2023): ~1T parameters (estimated)

Emergent capabilities at scale:
- Few-shot learning
- Chain-of-thought reasoning
- Code generation
- Following complex instructions
- Multilingual transfer

Training:
- Trillions of tokens from internet
- Months of training on thousands of GPUs
- RLHF for instruction-following
```

---

#### 13.6.2 In-Context Learning

**ID:** `in-context-learning`
**Parent:** `13.6`

**Full Explanation:**
In-context learning enables LLMs to perform tasks based on examples in the prompt, without parameter updates. Zero-shot: task description only. Few-shot: include examples of input-output pairs. The model learns the pattern from examples and applies to new inputs. Emergent capability at scale—small models cannot do this. Limited by context window and example quality.

**Simple Explanation:**
Give the model examples in the prompt, and it figures out the pattern. No training needed—just show it what you want. "Q: 2+2? A: 4. Q: 3+5? A: 8. Q: 7+3? A:" → Model outputs "10".

**Example:**
```
Few-shot translation:

Prompt:
English: Hello
French: Bonjour
English: Goodbye
French: Au revoir
English: Thank you
French: Merci
English: How are you?
French:

Model output: Comment allez-vous?

Zero-shot:
"Translate to French: How are you?"
→ Comment allez-vous?

The model learned translation from examples!
```

---

#### 13.6.3 RLHF (Reinforcement Learning from Human Feedback)

**ID:** `rlhf`
**Parent:** `13.6`

**Full Explanation:**
RLHF aligns LLMs with human preferences through three stages: (1) Supervised fine-tuning on high-quality examples, (2) Train reward model on human preference comparisons (which response is better?), (3) Use RL (PPO) to optimize policy against reward model. Makes models more helpful, harmless, and honest. Key for chat assistants. Challenge: reward hacking, distribution shift.

**Simple Explanation:**
Train the model to match human preferences. Humans rank different outputs. Train a "reward model" to predict rankings. Use RL to make the language model produce higher-reward outputs. How ChatGPT became helpful and conversational.

**Example:**
```
RLHF pipeline:

Stage 1: Supervised Fine-Tuning
Human writes: "What is 2+2?" → "2+2 equals 4"
Model learns to follow instruction format

Stage 2: Reward Model
Prompt: "Tell me a joke"
Response A: "Why did the chicken cross the road?"
Response B: "I don't tell jokes."
Human preference: A > B
Train reward model to predict: R(A) > R(B)

Stage 3: RL Fine-Tuning
Model generates response
Reward model scores it
PPO updates model to increase reward
Repeat many times

Result: Model produces responses humans prefer!
```

---

#### 13.6.4 Prompt Engineering

**ID:** `prompt-engineering`
**Parent:** `13.6`

**Full Explanation:**
Prompt engineering crafts inputs to elicit desired LLM outputs. Techniques: clear instructions, role specification ("You are an expert..."), few-shot examples, chain-of-thought ("Let's think step by step"), structured output formats, constraints. Quality of output heavily depends on prompt. An art and science—systematic exploration often needed. Increasingly important skill as LLMs proliferate.

**Simple Explanation:**
Craft the right question to get the right answer. Small changes in phrasing dramatically affect outputs. Adding "Let's think step by step" improves reasoning. Examples in the prompt guide format and style.

**Example:**
```
Bad prompt:
"Fix this code"
→ Vague, unclear what's wrong

Good prompt:
"You are an expert Python developer. The following code
throws an IndexError on line 5. Identify the bug and
provide a corrected version with explanation.

```python
[code here]
```"

Chain-of-thought prompt:
"Solve this math problem step by step, showing your work:
If a train travels at 60mph for 2.5 hours, how far does
it go?"

→ Model: "Let's solve this step by step:
1. Distance = Speed × Time
2. Speed = 60 mph
3. Time = 2.5 hours
4. Distance = 60 × 2.5 = 150 miles"
```

---

#### 13.6.5 Fine-Tuning LLMs

**ID:** `llm-finetuning`
**Parent:** `13.6`

**Full Explanation:**
Fine-tuning adapts pretrained LLMs to specific tasks or domains. Full fine-tuning updates all parameters—expensive but most flexible. Parameter-efficient methods (LoRA, adapters, prefix tuning) update small fractions of parameters—cheaper, avoids catastrophic forgetting. Instruction tuning teaches following diverse instructions. Domain adaptation improves performance on specialized corpora (medical, legal).

**Simple Explanation:**
Adapt the general LLM to your specific needs. Full fine-tuning: expensive but thorough. LoRA: add small trainable adapters—cheap and effective. Like teaching a generalist to specialize in your domain.

**Example:**
```
Fine-tuning approaches:

Full Fine-Tuning:
- Update all 7B parameters
- Needs 8× A100 GPUs
- Risk of forgetting general knowledge

LoRA (Low-Rank Adaptation):
- Freeze base model
- Add small rank-16 matrices: +0.1% parameters
- Train only adapters
- Works on single GPU!

Adapter architecture:
Original layer: h = Wx
With LoRA: h = Wx + BAx
where B (d×r), A (r×d), r << d

Example: 7B model
Full: 7B trainable parameters
LoRA (r=16): ~4M trainable parameters (0.06%)
```
