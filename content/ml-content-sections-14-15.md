# ML Cheatsheet Content - Sections 14-15: Specialized Learning & Structured Prediction

## 14. SPECIALIZED LEARNING PARADIGMS

### 14.1 Transfer Learning

---

#### 14.1.1 Transfer Learning

**ID:** `transfer-learning`
**Parent:** `14.1`

**Full Explanation:**
Transfer learning applies knowledge from a source task/domain to a target task/domain. Pretrain on large dataset (ImageNet for vision, text corpora for NLP), then fine-tune on smaller target dataset. Works because early features (edges, syntax) are universal; later features are task-specific. Dramatically reduces data requirements and training time. Foundation of modern deep learning practice.

**Simple Explanation:**
Use knowledge from one task to help with another. A model trained to recognize 1000 objects already knows about edges, textures, and shapes. Fine-tune it for your specific 10 classes instead of training from scratch. Much faster, works with less data.

**Example:**
```
Image classification transfer:

Source: ImageNet (14M images, 1000 classes)
Pretrained model: ResNet-50

Target: Medical X-ray classification (5000 images, 4 classes)

Without transfer:
- Train from scratch
- Needs ~100K images for good results
- Weeks of training

With transfer:
1. Load pretrained ResNet-50
2. Replace final layer: 1000 → 4 classes
3. Fine-tune on 5000 X-rays
4. 95% accuracy in hours!
```

---

#### 14.1.2 Domain Adaptation

**ID:** `domain-adaptation`
**Parent:** `14.1`

**Full Explanation:**
Domain adaptation handles distribution shift between source and target domains. Source: labeled training data. Target: different distribution (domain shift), often unlabeled. Techniques: feature alignment (minimize domain discrepancy), adversarial training (domain-invariant features), self-training on target. Examples: synthetic-to-real, lab-to-production, cross-dataset transfer.

**Simple Explanation:**
Make models work when training and deployment data look different. Trained on photos, deployed on sketches. Trained in lab conditions, deployed in the real world. Adapt the model to handle these differences.

**Example:**
```
Synthetic to Real adaptation:

Source domain: Simulated car images (cheap to generate)
Target domain: Real car photos (expensive to label)

Problem:
Model trained on simulation
Performs poorly on real images (domain gap)

Solution - Domain Adversarial:
1. Feature extractor (shared)
2. Task classifier: "Is this a car?"
3. Domain classifier: "Is this simulated or real?"

Train to:
- Maximize task accuracy
- Confuse domain classifier (domain-invariant features)

Result: Features work for both domains!
```

---

#### 14.1.3 Zero-Shot Learning

**ID:** `zero-shot-learning`
**Parent:** `14.1`

**Full Explanation:**
Zero-shot learning classifies instances of classes never seen during training. Relies on auxiliary information connecting seen and unseen classes: attributes ("has stripes," "is large"), text descriptions, or semantic embeddings. Model learns to map images to semantic space where similar classes cluster. At test time, matches to unseen class descriptions. CLIP enables zero-shot via image-text alignment.

**Simple Explanation:**
Recognize things you've never seen examples of. Trained on cats and dogs. Can classify "zebra" by knowing it's "horse-like with stripes." Uses descriptions or attributes to bridge to new classes.

**Example:**
```
Zero-shot animal classification:

Training classes: cat, dog, elephant
Unseen class: zebra

Attribute-based:
Zebra = [has_stripes: 1, is_large: 0.5, has_hooves: 1, ...]

Model learns:
Image → attribute predictions
[stripes: 0.9, large: 0.4, hooves: 0.95]

Match to class attributes:
Closest to zebra attributes → Classify as zebra!

CLIP-style:
Image embedding ≈ Text embedding("a photo of a zebra")
→ Zero-shot without explicit attributes
```

---

#### 14.1.4 Few-Shot Learning

**ID:** `few-shot-learning`
**Parent:** `14.1`

**Full Explanation:**
Few-shot learning trains models to learn new classes from very few examples (1-5 per class). Meta-learning approach: train on many tasks, each with few examples, to learn how to learn. Methods: Prototypical Networks (class prototypes from examples), MAML (learn good initialization), Matching Networks (attention over support set). Critical for personalization, rare categories, rapid adaptation.

**Simple Explanation:**
Learn from just a few examples. Humans can recognize a new animal from one picture. Few-shot learning teaches models this ability. Show 5 examples of a new class, and the model can classify it.

**Example:**
```
5-shot, 5-way classification:

Support set:
Class A: [img1, img2, img3, img4, img5]
Class B: [img1, img2, img3, img4, img5]
...
Class E: [img1, img2, img3, img4, img5]

Query: New image → Which class?

Prototypical Network:
1. Embed all support images
2. Compute class prototype (mean embedding)
   Prototype_A = mean(embed(A_images))
3. Embed query image
4. Find nearest prototype
   → Classify!

No retraining needed for new classes.
```

---

### 14.2 Reinforcement Learning

---

#### 14.2.1 Reinforcement Learning

**ID:** `reinforcement-learning`
**Parent:** `14.2`

**Full Explanation:**
RL learns through interaction with an environment to maximize cumulative reward. Agent observes state, takes action, receives reward, transitions to new state. Policy π(a|s) maps states to actions. Value function V(s) estimates future reward from state. Key challenges: exploration vs exploitation, credit assignment, sample efficiency. Algorithms: Q-learning, Policy Gradient, Actor-Critic.

**Simple Explanation:**
Learn by trial and error. Take actions, get rewards (or punishments), learn what works. Like training a dog with treats. No labeled examples—just try things and learn from outcomes.

**Example:**
```
RL for game playing:

State: Current game screen
Actions: Up, Down, Left, Right, Jump
Reward: +1 for points, -1 for losing life

Learning loop:
1. See game state
2. Choose action (e.g., Jump)
3. Game updates
4. Receive reward (+10 for collecting coin)
5. See new state
6. Update policy: "Jump was good in that situation"
7. Repeat millions of times

Result: Agent masters the game!
```

---

#### 14.2.2 Markov Decision Process (MDP)

**ID:** `mdp`
**Parent:** `14.2`

**Full Explanation:**
MDP formalizes RL problems with tuple (S, A, P, R, γ). S: state space. A: action space. P(s'|s,a): transition probability. R(s,a): reward function. γ: discount factor for future rewards. Markov property: future depends only on current state, not history. Optimal policy maximizes expected discounted return: E[Σγᵗrₜ]. Bellman equations describe optimal values recursively.

**Simple Explanation:**
The math framework for RL. States you can be in, actions you can take, probabilities of what happens next, and rewards you get. The goal: find the best strategy (policy) to maximize total rewards.

**Example:**
```
MDP for robot navigation:

States: Grid positions (x, y)
Actions: North, South, East, West
Transitions: Move in direction (80% success, 10% left, 10% right)
Rewards: +100 at goal, -1 per step, -50 for falling off
Discount: γ = 0.99

Bellman equation:
V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]

"Value of state = best action's immediate reward +
 discounted expected value of next states"
```

---

#### 14.2.3 Q-Learning

**ID:** `q-learning`
**Parent:** `14.2`

**Full Explanation:**
Q-learning learns action-value function Q(s,a) = expected return from taking action a in state s, then following optimal policy. Update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]. Off-policy: learns optimal Q even from exploratory behavior. Tabular version requires discrete states; Deep Q-Network (DQN) uses neural network for continuous/large state spaces.

**Simple Explanation:**
Learn the value of each action in each situation. "In this state, going left gives +10 future reward, going right gives +5." Pick actions with highest Q-values. Learn from any experience, not just optimal behavior.

**Example:**
```
Q-table for simple grid:

           Up    Down   Left   Right
State A:  10.5   5.2    3.1    8.4
State B:   7.3   9.1    2.0    6.5
State C:   ...

Learning update:
In state A, took "Up", got reward 5, landed in state B

Old Q(A, Up) = 10.5
New estimate = 5 + 0.9 × max(Q(B)) = 5 + 0.9 × 9.1 = 13.19
Updated Q(A, Up) = 10.5 + 0.1 × (13.19 - 10.5) = 10.77
```

---

#### 14.2.4 Policy Gradient

**ID:** `policy-gradient`
**Parent:** `14.2`

**Full Explanation:**
Policy gradient directly optimizes the policy πθ(a|s) by gradient ascent on expected reward. REINFORCE: ∇J(θ) = E[∇log πθ(a|s) × R]. Increase probability of actions that led to high rewards. Works with continuous actions (unlike Q-learning). High variance—addressed by baseline subtraction (A = R - V(s)), actor-critic methods. Foundation for PPO, A3C, and modern RL.

**Simple Explanation:**
Directly improve the policy (decision-making rules). If an action led to good outcomes, make it more likely. If it led to bad outcomes, make it less likely. Adjusts probabilities based on results.

**Example:**
```
Policy gradient for robot arm:

Policy: Neural network
Input: Joint angles, target position
Output: Probability distribution over motor commands

Episode 1:
Action: Move left
Reward: -10 (moved away from target)
Update: Decrease P(move left | this situation)

Episode 2:
Action: Move right
Reward: +50 (reached target!)
Update: Increase P(move right | this situation)

After many episodes:
Policy learns to move efficiently to targets.
```

---

#### 14.2.5 Actor-Critic

**ID:** `actor-critic`
**Parent:** `14.2`

**Full Explanation:**
Actor-Critic combines policy gradient (actor) with value function (critic). Actor: policy πθ(a|s) selects actions. Critic: value function Vφ(s) estimates expected returns, provides baseline for variance reduction. Advantage: A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s). Actor updated with advantage-weighted policy gradient; critic updated to minimize TD error. More stable than pure policy gradient.

**Simple Explanation:**
Two networks working together. Actor decides what to do. Critic evaluates how good the situation is. Critic's feedback helps Actor learn faster and more stably. Like having both a player and a coach.

**Example:**
```
Actor-Critic for game:

Actor network:
State → Action probabilities
"In this state, 70% attack, 30% defend"

Critic network:
State → Value estimate
"This state is worth +25 expected future reward"

Training step:
1. Actor chooses: Attack
2. Game gives: +10 reward, new state
3. Critic evaluates: New state worth +20
4. TD error = 10 + γ×20 - 25 = +3 (better than expected!)
5. Update actor: "Attack was good here, do more"
6. Update critic: "Adjust value estimate"
```

---

### 14.3 Self-Supervised Learning

---

#### 14.3.1 Self-Supervised Learning

**ID:** `self-supervised-learning`
**Parent:** `14.3`

**Full Explanation:**
Self-supervised learning creates supervisory signals from the data itself, avoiding manual labeling. Pretext tasks: predict masked words, rotated images, next frames, contrastive matching. Learns representations useful for downstream tasks. Bridges unsupervised and supervised learning. Foundation for modern NLP (BERT) and increasingly computer vision (SimCLR, MAE). Enables leveraging massive unlabeled datasets.

**Simple Explanation:**
Create your own labels from the data. Hide part of an image and predict it. Mask words and guess them. These "fake" tasks teach useful representations. No human labeling needed.

**Example:**
```
Self-supervised pretext tasks:

Text (BERT):
"The [MASK] sat on the mat"
Task: Predict [MASK] = "cat"

Images (MAE):
Show 25% of patches
Task: Reconstruct missing 75%

Images (Contrastive):
Same image, two augmentations → should be similar
Different images → should be different

Video:
Frame 1, Frame 2 → Predict Frame 3

After pretraining:
Features work great for downstream tasks!
Fine-tune on small labeled set → State-of-the-art
```

---

#### 14.3.2 Contrastive Learning

**ID:** `contrastive-learning`
**Parent:** `14.3`

**Full Explanation:**
Contrastive learning learns representations by pulling similar samples together and pushing dissimilar samples apart in embedding space. Positives: augmentations of the same image. Negatives: other images in batch. Loss (InfoNCE): maximize similarity of positive pairs relative to negatives. SimCLR, MoCo, CLIP use this. Requires large batch sizes or memory banks for sufficient negatives.

**Simple Explanation:**
Learn by comparison. "These two augmented versions of the same image should have similar embeddings. This other image should be different." Pushes and pulls in embedding space until similar things cluster together.

**Example:**
```
SimCLR contrastive learning:

Batch of N images
Each image augmented twice: 2N views

For image i:
Positive pair: (augment1_i, augment2_i)
Negative pairs: All other 2(N-1) views

Loss for image i:
-log(exp(sim(z_i, z_i+)/τ) / Σ exp(sim(z_i, z_k)/τ))

"Maximize similarity with positive, minimize with negatives"

Result: Images of cats cluster together,
        images of dogs cluster separately.
```

---

#### 14.3.3 Masked Language Modeling

**ID:** `masked-language-modeling`
**Parent:** `14.3`

**Full Explanation:**
MLM masks random tokens in input and trains model to predict them. BERT masks 15% of tokens: 80% replaced with [MASK], 10% random token, 10% unchanged. Forces model to understand context from both directions. Pretraining objective for bidirectional transformers. Creates powerful text encoders for downstream tasks. Differs from autoregressive LM which only sees past context.

**Simple Explanation:**
Hide some words, guess them from context. "The ___ chased the mouse" → "cat". Both left and right context help. After seeing millions of sentences, the model deeply understands language.

**Example:**
```
Masked Language Modeling:

Original: "The quick brown fox jumps over the lazy dog"
Masked:   "The quick brown [MASK] jumps over the lazy dog"
Task:     Predict [MASK] = "fox"

Multi-mask:
Input:  "The [MASK] brown fox [MASK] over the [MASK] dog"
Target: "The quick brown fox jumps over the lazy dog"

Model learns:
- Grammar: "jumps" (verb after subject)
- Semantics: "fox" (animal that jumps)
- Common phrases: "lazy dog"
```

---

## 15. STRUCTURED PREDICTION

### 15.1 Sequence Labeling

---

#### 15.1.1 Sequence Labeling

**ID:** `sequence-labeling`
**Parent:** `15.1`

**Full Explanation:**
Sequence labeling assigns a label to each element in a sequence. Input: sequence x = (x₁, ..., xₙ). Output: labels y = (y₁, ..., yₙ). Tasks: POS tagging (word → part of speech), NER (word → entity type), chunking. Models must capture dependencies between labels. Approaches: HMM, CRF, BiLSTM-CRF, Transformer + CRF.

**Simple Explanation:**
Label each item in a sequence. For each word: is it a noun, verb, adjective? Is it a person name, location, or organization? Each position gets its own label.

**Example:**
```
Named Entity Recognition:

Input:  "John works at Google in California"
Labels: "B-PER O O B-ORG O B-LOC"

B-PER = Beginning of Person name
B-ORG = Beginning of Organization
B-LOC = Beginning of Location
O = Outside (not an entity)

Part-of-Speech tagging:
Input:  "The cat sat on the mat"
Labels: "DET NOUN VERB PREP DET NOUN"
```

---

#### 15.1.2 Conditional Random Field (CRF)

**ID:** `crf`
**Parent:** `15.1`

**Full Explanation:**
CRF models P(y|x) directly, considering the entire output sequence. Linear-chain CRF: P(y|x) ∝ exp(Σ score(yᵢ₋₁, yᵢ, x, i)). Scores include transition features (label-to-label) and emission features (input-to-label). Viterbi algorithm finds optimal sequence; forward-backward computes marginals. Unlike HMM, CRF is discriminative and handles overlapping features. Often combined with neural networks (BiLSTM-CRF).

**Simple Explanation:**
Model the whole label sequence, not just individual labels. "B-PER should be followed by I-PER, not I-ORG." Learns label transition patterns. Predicts the globally best label sequence.

**Example:**
```
CRF for NER:

Emission scores (from neural network):
           O    B-PER  I-PER  B-LOC
"John"   -2.0   5.0    0.1   -1.0
"works"   4.0  -1.0   -2.0   -1.0
"at"      5.0  -2.0   -2.0   -1.0

Transition scores:
         O    B-PER  I-PER  B-LOC
O        1.0   0.5   -5.0    0.5
B-PER    0.5  -1.0    3.0   -1.0  ← B-PER → I-PER likely
I-PER    0.8  -1.0    2.0   -1.0
B-LOC    1.0   0.5   -5.0   -1.0

Viterbi finds best path considering both!
```

---

#### 15.1.3 Named Entity Recognition (NER)

**ID:** `ner`
**Parent:** `15.1`

**Full Explanation:**
NER identifies and classifies named entities in text into predefined categories: Person, Organization, Location, Date, etc. BIO tagging: B-type (beginning), I-type (inside), O (outside). Challenges: ambiguity ("Apple" = company or fruit), nested entities, rare entities. Modern approaches: fine-tuned BERT + CRF layer. Key for information extraction, question answering, knowledge graphs.

**Simple Explanation:**
Find names of people, places, companies, dates in text. "Apple announced new products" → Apple is B-ORG (organization). Crucial for extracting structured information from unstructured text.

**Example:**
```
NER example:

Text: "Elon Musk founded SpaceX in California in 2002"

Entities found:
- Elon Musk: PERSON
- SpaceX: ORGANIZATION
- California: LOCATION
- 2002: DATE

BIO format:
Elon    → B-PER
Musk    → I-PER
founded → O
SpaceX  → B-ORG
in      → O
California → B-LOC
in      → O
2002    → B-DATE
```

---

### 15.2 Sequence-to-Sequence

---

#### 15.2.1 Sequence-to-Sequence (Seq2Seq)

**ID:** `seq2seq`
**Parent:** `15.2`

**Full Explanation:**
Seq2Seq maps variable-length input sequences to variable-length output sequences. Encoder-decoder architecture: encoder compresses input to fixed representation (or sequence of representations with attention), decoder generates output token by token. Applications: machine translation, summarization, dialogue, code generation. Original: LSTM encoder-decoder. Modern: Transformer-based.

**Simple Explanation:**
Transform one sequence into another. Input sentence in English → output sentence in French. Input document → output summary. The input and output can be different lengths.

**Example:**
```
Machine translation:

Input:  "I love machine learning"
Output: "J'aime l'apprentissage automatique"

Encoder processes: "I love machine learning"
→ Context representation

Decoder generates:
Step 1: [START] → "J'"
Step 2: "J'" → "aime"
Step 3: "J'aime" → "l'"
Step 4: "J'aime l'" → "apprentissage"
Step 5: → "automatique"
Step 6: → [END]

With attention: decoder focuses on relevant source words
"aime" attends to "love"
```

---

#### 15.2.2 Encoder-Decoder

**ID:** `encoder-decoder`
**Parent:** `15.2`

**Full Explanation:**
Encoder-decoder separates sequence understanding from sequence generation. Encoder: reads entire input, produces representations. Decoder: generates output conditioned on encoder representations. In attention-based models, decoder attends to encoder outputs at each step. Enables handling input/output of different lengths and modalities. Foundation for translation, summarization, image captioning.

**Simple Explanation:**
Two-part system: one part understands the input, one part produces the output. Like having a listener who takes notes and a speaker who talks from those notes. The speaker can refer back to the notes (attention) while talking.

**Example:**
```
Encoder-Decoder for summarization:

Encoder:
"The quick brown fox jumps over the lazy dog.
 The dog was sleeping peacefully."
        ↓
[h₁, h₂, h₃, ..., h₁₅] (encoder hidden states)

Decoder:
[START] + attention over encoder →  "A"
"A" + attention → "fox"
"A fox" + attention → "jumps"
"A fox jumps" + attention → "over"
"A fox jumps over" + attention → "a"
"A fox jumps over a" + attention → "sleeping"
... → "dog" → [END]

Output: "A fox jumps over a sleeping dog"
```

---

#### 15.2.3 Beam Search

**ID:** `beam-search`
**Parent:** `15.2`

**Full Explanation:**
Beam search is a decoding algorithm that maintains the k most promising partial sequences (beam width k). At each step, expand each beam by all possible next tokens, score, keep top k. Balances exploration (multiple hypotheses) and exploitation (focus on likely). Better than greedy (k=1) which may miss globally optimal sequences. Larger k = better quality but slower. Length normalization handles length bias.

**Simple Explanation:**
Keep track of several promising paths, not just the best one. At each step, extend all paths, keep the best few. Like exploring a maze by following multiple routes simultaneously. Usually finds better solutions than greedy search.

**Example:**
```
Beam search with k=2:

Step 1:
"The" → P=0.4
"A"   → P=0.3
Keep: ["The": 0.4, "A": 0.3]

Step 2:
"The cat" → 0.4 × 0.3 = 0.12
"The dog" → 0.4 × 0.5 = 0.20 ✓
"A cat"   → 0.3 × 0.4 = 0.12
"A dog"   → 0.3 × 0.3 = 0.09
Keep: ["The dog": 0.20, "The cat": 0.12]

Step 3:
Continue expanding best 2...

vs. Greedy:
Step 1: "The" (best)
Step 2: "The dog" (best)
Same result here, but beam often finds better sequences!
```

---

### 15.3 Structured Output

---

#### 15.3.1 Structured Output Prediction

**ID:** `structured-output`
**Parent:** `15.3`

**Full Explanation:**
Structured output prediction produces complex, interdependent outputs rather than single labels. Examples: parse trees, graphs, alignments, segmentations. Output space is exponentially large; specialized algorithms needed for inference. Methods: dynamic programming (parsing), graph algorithms, iterative refinement, neural set prediction. Loss functions must handle structure (tree kernels, edit distance).

**Simple Explanation:**
Predict complex outputs like trees, graphs, or structured documents. Not just one label but many interconnected pieces. The structure between output pieces matters—can't just predict each piece independently.

**Example:**
```
Structured outputs:

1. Dependency Parse Tree:
Input: "The cat sat on the mat"
Output:    sat
          / | \
       cat  on  .
       /     |
     The    mat
              |
             the

2. Scene Graph:
Input: Image of dog on couch
Output:
  dog --sitting_on--> couch
   |
  --has_color--> brown

3. Document Layout:
Input: PDF page
Output: {title: rect1, paragraph: [rect2,rect3], figure: rect4}
```

---

#### 15.3.2 Graph Neural Network (GNN)

**ID:** `gnn`
**Parent:** `15.3`

**Full Explanation:**
GNNs learn on graph-structured data by aggregating information from neighboring nodes. Message passing: each node collects messages from neighbors, updates its representation. After K layers, each node has K-hop neighborhood information. Types: GCN (spectral), GraphSAGE (sampling), GAT (attention-based). Applications: social networks, molecules, knowledge graphs, recommendation, program analysis.

**Simple Explanation:**
Neural networks that work on graphs. Each node looks at its neighbors to update its representation. After several rounds, each node knows about its extended neighborhood. Perfect for social networks, molecules, and any connected data.

**Example:**
```
GNN for molecular property prediction:

Molecule as graph:
Atoms = nodes (with features: element type, charge)
Bonds = edges (with features: bond type)

Message passing:
Round 1: Each atom aggregates from direct neighbors
Round 2: Each atom aggregates from 2-hop neighborhood
Round 3: Each atom knows 3-hop context

        O
        ‖
    H-C-C-OH
        |
        H

After 3 rounds:
Carbon atom knows about: attached H, attached O, the OH group

Final:
Aggregate all atom representations → Molecule representation
→ Predict: Solubility = 2.3
```
