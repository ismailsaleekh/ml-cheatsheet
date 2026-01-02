# ML Cheatsheet Content - Sections 16-17: Computer Vision & NLP

## 16. COMPUTER VISION

### 16.1 Image Classification

---

#### 16.1.1 Image Classification

**ID:** `image-classification`
**Parent:** `16.1`

**Full Explanation:**
Image classification assigns a single label to an entire image from a predefined set of categories. Input: image (H×W×C). Output: class probabilities. Pipeline: CNN extracts features, fully connected layers classify. Key benchmarks: ImageNet (1000 classes), CIFAR-10/100. Architectures evolved from AlexNet through VGG, ResNet, to modern EfficientNet and Vision Transformers.

**Simple Explanation:**
Look at an image, say what it is. Is it a cat, dog, or car? The model outputs probabilities for each possible class. The class with highest probability is the prediction.

**Example:**
```
Image classification pipeline:

Input: 224×224×3 image of a cat

CNN Feature Extraction:
Conv layers → 7×7×512 feature map
Global pooling → 512-dim vector

Classification:
512 → 1000 (ImageNet classes)
Softmax → Probabilities

Output:
Cat: 0.85, Dog: 0.10, Tiger: 0.03, ...
Prediction: Cat
```

---

#### 16.1.2 ResNet

**ID:** `resnet`
**Parent:** `16.1`

**Full Explanation:**
ResNet (Residual Networks) introduced skip connections enabling training of very deep networks (50-152+ layers). Instead of learning H(x), learn residual F(x) = H(x) - x, so output is F(x) + x. If optimal is identity, F(x) = 0 is easy to learn. Solves vanishing gradient problem by providing gradient highways. Variants: ResNet-50, ResNet-101, ResNeXt, ResNeSt.

**Simple Explanation:**
Add shortcuts that skip layers. Instead of transforming everything, just add small changes to the original. Allows training of very deep networks—gradients can flow through shortcuts. The breakthrough that enabled 100+ layer networks.

**Example:**
```
ResNet residual block:

Input x
    ↓
[Conv → BN → ReLU]
    ↓
[Conv → BN]
    ↓
  + x ← Skip connection!
    ↓
 ReLU
    ↓
Output

If optimal is identity (no change needed):
Weights learn to be zero → Output = 0 + x = x

Deep network can "skip" unnecessary layers!
```

---

#### 16.1.3 Vision Transformer (ViT)

**ID:** `vit`
**Parent:** `16.1`

**Full Explanation:**
ViT applies transformer architecture to images by treating images as sequences of patches. Split image into 16×16 patches, linearly embed each patch, add positional embeddings, process with standard transformer encoder. Outperforms CNNs when pretrained on large datasets (ImageNet-21k, JFT-300M). Less inductive bias (no convolutions), requires more data. Foundation for modern vision models.

**Simple Explanation:**
Use the same attention mechanism as language models, but for images. Cut image into patches (like words), process with transformer. No convolutions! Works great when you have lots of training data.

**Example:**
```
ViT processing:

Image: 224×224×3

1. Split into patches:
   14×14 grid of 16×16 patches = 196 patches

2. Embed patches:
   Each 16×16×3 patch → 768-dim vector
   + Positional embedding
   + [CLS] token prepended

3. Transformer:
   197 tokens × 768 dims
   12 transformer layers
   Self-attention across all patches

4. Classification:
   [CLS] token → MLP → Class prediction

Key insight: Patches are like words!
```

---

### 16.2 Object Detection

---

#### 16.2.1 Object Detection

**ID:** `object-detection`
**Parent:** `16.2`

**Full Explanation:**
Object detection locates and classifies multiple objects in images. Output: bounding boxes (x, y, width, height) + class labels + confidence scores. Two-stage detectors (R-CNN family): propose regions, then classify. One-stage detectors (YOLO, SSD): directly predict boxes and classes. Metrics: mAP (mean Average Precision), IoU (Intersection over Union). Applications: autonomous driving, surveillance, retail.

**Simple Explanation:**
Find all objects in an image and draw boxes around them. For each box, say what's inside and how confident you are. Unlike classification (one label for whole image), detection finds multiple objects at specific locations.

**Example:**
```
Object detection output:

Image: Street scene

Detections:
[Box1] Car,    confidence=0.95, bbox=[100,200,150,100]
[Box2] Person, confidence=0.88, bbox=[300,150,50,120]
[Box3] Car,    confidence=0.72, bbox=[450,210,140,90]
[Box4] Dog,    confidence=0.65, bbox=[320,300,40,35]

Visual:
+-------+
|       | Car (0.95)
+-------+
            +--+
            |  | Person (0.88)
            +--+
```

---

#### 16.2.2 YOLO (You Only Look Once)

**ID:** `yolo`
**Parent:** `16.2`

**Full Explanation:**
YOLO treats detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one pass. Divides image into grid; each cell predicts boxes and classes for objects whose center falls in that cell. Extremely fast (real-time), trades some accuracy for speed. Versions: YOLOv1-v8, each improving accuracy while maintaining speed.

**Simple Explanation:**
Look at the whole image once, predict all boxes simultaneously. Super fast—can process video in real-time. Grid-based: each grid cell detects objects centered there. The go-to for real-time detection.

**Example:**
```
YOLO processing:

1. Divide image into 7×7 grid

2. Each cell predicts:
   - 2 bounding boxes (x, y, w, h, confidence)
   - 20 class probabilities (for PASCAL VOC)

3. Output tensor: 7×7×30
   (7×7 grid × (2 boxes × 5 values + 20 classes))

4. Post-processing:
   - Filter low-confidence boxes
   - Non-maximum suppression (remove overlaps)

Speed: 45-155 FPS depending on version
Real-time on video! 🎥
```

---

#### 16.2.3 Non-Maximum Suppression (NMS)

**ID:** `nms`
**Parent:** `16.2`

**Full Explanation:**
NMS removes duplicate detections for the same object. Multiple boxes often predict the same object; NMS keeps only the best. Algorithm: (1) Select highest confidence box, (2) Remove all boxes with IoU > threshold with selected box, (3) Repeat until no boxes remain. Threshold typically 0.5. Variants: Soft-NMS (decay scores instead of removing), learned NMS.

**Simple Explanation:**
Remove duplicate boxes. If many boxes found the same car, keep only the best one. For each object, one box should remain. Compare overlap (IoU) between boxes; if too similar, remove the weaker one.

**Example:**
```
NMS example:

Before NMS:
Box A: Car, conf=0.95, [100,200,150,100]
Box B: Car, conf=0.88, [105,195,145,95]  ← Overlaps A
Box C: Car, conf=0.72, [400,200,150,100]
Box D: Car, conf=0.45, [102,198,148,98]  ← Overlaps A

IoU(A,B) = 0.85 (high overlap)
IoU(A,C) = 0.01 (different objects)
IoU(A,D) = 0.80 (high overlap)

NMS with threshold 0.5:
1. Select A (highest: 0.95)
2. Remove B (IoU 0.85 > 0.5) ✗
3. Keep C (IoU 0.01 < 0.5) ✓
4. Remove D (IoU 0.80 > 0.5) ✗

After NMS: Box A, Box C (one per object)
```

---

### 16.3 Image Segmentation

---

#### 16.3.1 Image Segmentation

**ID:** `image-segmentation`
**Parent:** `16.3`

**Full Explanation:**
Image segmentation assigns labels to each pixel. Semantic segmentation: label per pixel, no instance distinction (all cats are "cat"). Instance segmentation: separate different instances (cat1, cat2). Panoptic: combines both. Architectures: FCN (fully convolutional), U-Net (encoder-decoder with skip connections), DeepLab (atrous convolutions). Applications: medical imaging, autonomous driving, photo editing.

**Simple Explanation:**
Label every pixel in the image. Which pixels are sky? Which are road? Which are cars? Like coloring a coloring book where each region has a specific category. Instance segmentation goes further—distinguishes different cars from each other.

**Example:**
```
Segmentation types:

Image: Two cats on a sofa

Semantic segmentation:
Every pixel labeled as: cat/sofa/background
All cat pixels have same label "cat"

Instance segmentation:
Every pixel labeled with instance:
cat_1, cat_2, sofa_1, background

Panoptic segmentation:
"Stuff" (amorphous): sofa, background
"Things" (countable): cat_1, cat_2

Output: H×W mask with pixel-wise labels
```

---

#### 16.3.2 U-Net

**ID:** `u-net`
**Parent:** `16.3`

**Full Explanation:**
U-Net is an encoder-decoder architecture with skip connections for precise segmentation. Encoder (contracting path): captures context through downsampling. Decoder (expanding path): enables precise localization through upsampling. Skip connections concatenate encoder features to decoder, preserving spatial details lost in downsampling. Originally for biomedical segmentation, now used broadly.

**Simple Explanation:**
U-shaped network that compresses then expands. Shortcuts copy detailed information from early layers to later layers. Combines "what" (from deep features) with "where" (from skip connections). Excellent for medical images and precise boundaries.

**Example:**
```
U-Net architecture:

Encoder (left side):
Input: 572×572×1
↓ Conv, Conv, Pool → 284×284×64
↓ Conv, Conv, Pool → 140×140×128
↓ Conv, Conv, Pool → 68×68×256
↓ Conv, Conv, Pool → 32×32×512
↓ Conv, Conv → 28×28×1024 (bottleneck)

Decoder (right side):
↑ Up-conv → 52×52×512
  + Skip from encoder 256-level
↑ Up-conv → 100×100×256
  + Skip from encoder 128-level
↑ Up-conv → 196×196×128
  + Skip from encoder 64-level
↑ Up-conv → 388×388×64
↓ 1×1 Conv → 388×388×2 (output)
```

---

### 16.4 Face Recognition

---

#### 16.4.1 Face Recognition

**ID:** `face-recognition`
**Parent:** `16.4`

**Full Explanation:**
Face recognition identifies or verifies individuals from facial images. Verification (1:1): "Is this the same person?" Identification (1:N): "Who is this person?" Pipeline: detect face, align, extract embedding, compare. Embeddings learned with contrastive losses (triplet loss, ArcFace). Challenges: pose, lighting, occlusion, aging. Applications: phone unlock, access control, surveillance.

**Simple Explanation:**
Recognize who someone is from their face. Create a "face fingerprint" (embedding) that's similar for the same person, different for different people. Compare embeddings to identify or verify.

**Example:**
```
Face recognition pipeline:

1. Detection: Find face in image
   Crop: 112×112 face region

2. Alignment: Normalize pose
   Align eyes, nose to standard positions

3. Embedding: Extract features
   Face → CNN → 512-dim vector
   [0.23, -0.15, 0.82, ...]

4. Comparison:
   Euclidean distance or cosine similarity

   Same person: distance < 0.6
   Different person: distance > 0.8

Verification:
embed(photo) vs embed(ID_card)
Distance = 0.3 → Same person ✓
```

---

#### 16.4.2 Triplet Loss

**ID:** `triplet-loss`
**Parent:** `16.4`

**Full Explanation:**
Triplet loss learns embeddings where similar samples are closer than dissimilar ones. Triplet: anchor (a), positive (same class as anchor, p), negative (different class, n). Loss: max(0, ||a-p||² - ||a-n||² + margin). Pushes positive closer, negative farther. Hard negative mining: select challenging negatives. Foundation for face recognition (FaceNet), general metric learning.

**Simple Explanation:**
Learn embeddings using triplets: anchor, same-person example, different-person example. Make anchor closer to same-person than to different-person. Like teaching: "this matches, that doesn't."

**Example:**
```
Triplet loss training:

Triplet:
Anchor: Photo of Alice (front)
Positive: Photo of Alice (side) - same person
Negative: Photo of Bob - different person

Goal:
distance(Alice_front, Alice_side) + margin
    < distance(Alice_front, Bob)

Before training:
d(anchor, positive) = 1.5
d(anchor, negative) = 1.2
Loss = max(0, 1.5 - 1.2 + 0.3) = 0.6 ✗

After training:
d(anchor, positive) = 0.4
d(anchor, negative) = 1.5
Loss = max(0, 0.4 - 1.5 + 0.3) = 0 ✓
```

---

## 17. NATURAL LANGUAGE PROCESSING

### 17.1 Text Preprocessing

---

#### 17.1.1 Tokenization

**ID:** `tokenization`
**Parent:** `17.1`

**Full Explanation:**
Tokenization splits text into tokens (words, subwords, or characters). Word tokenization: split on whitespace/punctuation. Subword tokenization: BPE, WordPiece, SentencePiece learn vocabulary balancing frequency and coverage. Subwords handle unknown words by breaking into known pieces. Character-level: maximum coverage, longer sequences. Choice affects vocabulary size, OOV handling, and sequence length.

**Simple Explanation:**
Break text into pieces the model can process. Words, word parts, or characters. Modern models use subwords: common words stay whole, rare words split into pieces. "unhappiness" → "un", "happiness" or "un", "happ", "iness".

**Example:**
```
Tokenization methods:

Text: "Tokenization is awesome!"

Word tokenization:
["Tokenization", "is", "awesome", "!"]

Subword (BPE/WordPiece):
["Token", "ization", "is", "awesome", "!"]

Character:
["T","o","k","e","n","i","z","a","t","i","o","n"," ","i","s",...]

GPT-style (with unknown word):
"cryptocurrency" → ["crypt", "ocur", "rency"]
Unknown word handled by splitting!
```

---

#### 17.1.2 Word Embeddings

**ID:** `word-embeddings`
**Parent:** `17.1`

**Full Explanation:**
Word embeddings map words to dense vectors capturing semantic relationships. Similar words have similar vectors. Training: Word2Vec (predict context from word or vice versa), GloVe (matrix factorization of co-occurrence). Properties: king - man + woman ≈ queen. Static embeddings: one vector per word. Contextual embeddings (BERT): different vectors based on context.

**Simple Explanation:**
Convert words to numbers that capture meaning. Similar words get similar numbers. "King" and "queen" are close. "King - man + woman = queen" works with vectors! The foundation of modern NLP.

**Example:**
```
Word embedding properties:

Embedding dimensions: 300

Semantic similarity:
cosine(embed("cat"), embed("dog")) = 0.8 (high)
cosine(embed("cat"), embed("car")) = 0.2 (low)

Analogies:
king - man + woman ≈ queen
Paris - France + Japan ≈ Tokyo

embed("king") = [0.2, -0.4, 0.1, ...]
embed("man") = [0.3, -0.2, 0.2, ...]
embed("woman") = [0.4, -0.1, 0.3, ...]

king - man + woman = [-0.1, -0.2, -0.1, ...] + [0.4, -0.1, 0.3, ...]
                   = [0.3, -0.3, 0.2, ...] ≈ embed("queen")
```

---

### 17.2 Language Understanding

---

#### 17.2.1 BERT

**ID:** `bert`
**Parent:** `17.2`

**Full Explanation:**
BERT (Bidirectional Encoder Representations from Transformers) pretrained bidirectional transformer on masked language modeling and next sentence prediction. Sees full context (left and right) unlike GPT (left only). Fine-tune for downstream tasks: classification, NER, QA by adding task-specific heads. Revolutionized NLP benchmarks. Variants: RoBERTa (better training), ALBERT (parameter sharing), DistilBERT (smaller).

**Simple Explanation:**
Powerful language understanding model. Reads text in both directions to understand context. Pretrained on massive text, fine-tuned for specific tasks. The word "bank" gets different meanings in "river bank" vs "bank account."

**Example:**
```
BERT for text classification:

1. Pretrained BERT understands language

2. Fine-tune for sentiment:
   Input: "[CLS] This movie was great! [SEP]"

   BERT processes:
   - [CLS] token aggregates sentence meaning
   - Bidirectional attention captures context

   [CLS] embedding → Linear layer → Softmax

   Output: [Negative: 0.02, Positive: 0.98]

3. Few epochs of fine-tuning:
   BERT already understands "great" is positive
   Just needs to learn classification task
```

---

#### 17.2.2 Text Classification

**ID:** `text-classification`
**Parent:** `17.2`

**Full Explanation:**
Text classification assigns labels to text documents. Binary (spam/not spam), multiclass (topic categories), or multilabel (multiple tags). Traditional: TF-IDF + classifier. Neural: CNN on word embeddings, RNN/LSTM, Transformers (BERT). For BERT: use [CLS] token representation with classification head. Data augmentation, class imbalance handling crucial. Applications: sentiment analysis, intent detection, content moderation.

**Simple Explanation:**
Assign a category to text. Is this email spam? Is this review positive or negative? What topic is this article about? Modern approaches use pretrained models fine-tuned on labeled examples.

**Example:**
```
Sentiment classification:

Input texts:
"I love this product!" → Positive
"Terrible experience, never again" → Negative
"It's okay, nothing special" → Neutral

BERT approach:
1. Tokenize: ["[CLS]", "I", "love", "this", "product", "!", "[SEP]"]
2. BERT forward pass
3. Take [CLS] embedding (768 dims)
4. Linear: 768 → 3 (classes)
5. Softmax → [0.01, 0.98, 0.01]
6. Prediction: Positive

Accuracy on benchmarks: 95%+ with BERT
```

---

#### 17.2.3 Question Answering

**ID:** `question-answering`
**Parent:** `17.2`

**Full Explanation:**
Question answering extracts or generates answers from context. Extractive QA: identify answer span in given passage (start/end positions). Generative QA: generate answer text (seq2seq). Reading comprehension: answer from passage. Open-domain QA: retrieve relevant documents first, then answer. BERT for extractive QA: predict start and end token positions. Benchmarks: SQuAD, Natural Questions.

**Simple Explanation:**
Find answers in text. Given a question and a passage, point to where the answer is. Or generate an answer based on the passage. Like finding info in a textbook.

**Example:**
```
Extractive QA with BERT:

Context: "The Eiffel Tower is a wrought iron lattice tower
         in Paris. It was built in 1889."

Question: "When was the Eiffel Tower built?"

BERT processing:
Input: "[CLS] When was the Eiffel Tower built? [SEP]
       The Eiffel Tower is a wrought iron lattice tower
       in Paris. It was built in 1889. [SEP]"

Model predicts:
Start position: token 18 ("1889")
End position: token 18 ("1889")

Answer: "1889"
```

---

### 17.3 Text Generation

---

#### 17.3.1 GPT (Generative Pre-trained Transformer)

**ID:** `gpt`
**Parent:** `17.3`

**Full Explanation:**
GPT is a decoder-only transformer trained autoregressively (predict next token). Pretrained on massive text corpora, scales to billions/trillions of parameters. Unlike BERT (bidirectional encoder), GPT is unidirectional (causal), enabling text generation. GPT-3/4: few-shot learning via prompting. Foundation for ChatGPT, coding assistants, content generation. RLHF makes it follow instructions.

**Simple Explanation:**
The model behind ChatGPT. Predicts the next word given previous words. Trained on internet text, learns grammar, facts, reasoning. Large enough models can follow instructions, answer questions, write code.

**Example:**
```
GPT text generation:

Prompt: "The secret to happiness is"

Generation (token by token):
"The secret to happiness is" → "finding"
"The secret to happiness is finding" → "joy"
"The secret to happiness is finding joy" → "in"
"The secret to happiness is finding joy in" → "small"
"The secret to happiness is finding joy in small" → "moments"

Output: "The secret to happiness is finding joy in small moments."

Each step:
- Feed all previous tokens
- Model outputs probability for each vocab word
- Sample or take most likely
- Append to sequence, repeat
```

---

#### 17.3.2 Autoregressive Generation

**ID:** `autoregressive-generation`
**Parent:** `17.3`

**Full Explanation:**
Autoregressive generation produces outputs one token at a time, conditioning on all previous tokens. P(x₁,...,xₙ) = ∏P(xᵢ|x₁,...,xᵢ₋₁). Cannot parallelize during generation (sequential dependency). Decoding strategies: greedy (argmax), beam search (top-k paths), sampling (temperature), nucleus sampling (top-p). Enables coherent long-form generation. Used in GPT, LLaMA, and text generation models.

**Simple Explanation:**
Generate one word at a time, each word based on all previous words. Can't generate word 10 until words 1-9 exist. Like writing left to right—each word influences the next.

**Example:**
```
Autoregressive generation:

P("I love ML") = P("I") × P("love"|"I") × P("ML"|"I love")

Step-by-step:
1. P(w₁): Sample first word → "I"
2. P(w₂|"I"): Sample second word → "love"
3. P(w₃|"I love"): Sample third word → "ML"

Decoding strategies:
Greedy: Always pick highest probability word
        Deterministic but can be repetitive

Temperature sampling (T=1.0):
        Sample proportional to probabilities
        More diverse but can be incoherent

Nucleus (top-p=0.9):
        Sample from smallest set covering 90% probability
        Good balance of quality and diversity
```

---

#### 17.3.3 Machine Translation

**ID:** `machine-translation`
**Parent:** `17.3`

**Full Explanation:**
Machine translation converts text from source to target language. Statistical MT: phrase-based with language models. Neural MT: encoder-decoder with attention, now transformer-based. Training: parallel corpora (aligned sentence pairs). Challenges: morphology, word order, idioms, rare words. Evaluation: BLEU score (n-gram overlap with references). Modern systems approach human quality for common language pairs.

**Simple Explanation:**
Translate text from one language to another. Encode source sentence, decode into target language. Attention allows focusing on relevant source words while generating each target word. Powers Google Translate and similar services.

**Example:**
```
Neural Machine Translation:

Source (English): "I love machine learning"
Target (French): "J'aime l'apprentissage automatique"

Transformer translation:
1. Encode source:
   ["I", "love", "machine", "learning"]
   → [h₁, h₂, h₃, h₄] encoder hidden states

2. Decode with attention:
   [START] + attend to source → "J'"
   "J'" + attend to source → "aime"
   ...

Attention pattern:
"J'" attends to → "I" (subject)
"aime" attends to → "love" (verb)
"automatique" attends to → "machine" (adj)
```

---

### 17.4 Semantic Understanding

---

#### 17.4.1 Semantic Similarity

**ID:** `semantic-similarity`
**Parent:** `17.4`

**Full Explanation:**
Semantic similarity measures meaning closeness between texts. Lexical approaches: word overlap, edit distance. Embedding approaches: cosine similarity of sentence embeddings. Sentence-BERT: fine-tuned for similarity with siamese/triplet networks. Applications: duplicate detection, search ranking, clustering, recommendation. Datasets: STS benchmark, paraphrase detection.

**Simple Explanation:**
How similar is the meaning of two texts? "I love dogs" and "I adore canines" are semantically similar despite different words. Embed texts as vectors, measure distance between them.

**Example:**
```
Semantic similarity:

Sentence 1: "The cat is sleeping on the couch"
Sentence 2: "A feline is napping on the sofa"

Lexical similarity (word overlap): Low
(Only "the" and "on" match)

Semantic similarity (SBERT):
embed(S1) = [0.2, -0.5, 0.8, ...]
embed(S2) = [0.3, -0.4, 0.7, ...]
cosine(S1, S2) = 0.95 (very similar!)

Same meaning, different words → High semantic similarity

Use cases:
- Find duplicate questions
- Match queries to documents
- Cluster similar complaints
```

---

#### 17.4.2 Sentiment Analysis

**ID:** `sentiment-analysis`
**Parent:** `17.4`

**Full Explanation:**
Sentiment analysis detects subjective opinions and emotions in text. Levels: document, sentence, aspect. Polarity: positive/negative/neutral. Intensity: rating (1-5 stars). Aspects: sentiment toward specific features ("Great camera, terrible battery"). Methods: lexicon-based (word polarity dictionaries), ML classifiers, deep learning (BERT). Challenges: sarcasm, negation, domain adaptation.

**Simple Explanation:**
Detect feelings in text. Is this review positive or negative? How strongly? About which aspects? Crucial for understanding customer feedback, social media monitoring, brand perception.

**Example:**
```
Sentiment analysis levels:

Document-level:
"Great product! Fast shipping, excellent quality,
 will buy again."
→ Positive (0.95)

Sentence-level:
"The food was delicious but service was slow."
→ Sentence 1: Positive (0.9)
→ Sentence 2: Negative (0.7)

Aspect-based:
"The camera is amazing but battery life is terrible."
→ Camera: Positive (0.95)
→ Battery: Negative (0.90)
→ Overall: Mixed

Handling sarcasm:
"Oh great, another software update that breaks everything."
→ Naive: Positive (sees "great")
→ Sarcasm-aware: Negative
```

---

#### 17.4.3 Information Extraction

**ID:** `information-extraction`
**Parent:** `17.4`

**Full Explanation:**
Information extraction structures unstructured text. Tasks: Named Entity Recognition (find entities), Relation Extraction (find relationships between entities), Event Extraction (identify events, participants, attributes), Coreference Resolution (link entity mentions). Builds knowledge bases from text. Pipeline or joint models. Applications: knowledge graph construction, news analysis, scientific literature mining.

**Simple Explanation:**
Pull structured facts from text. "Apple CEO Tim Cook announced..." → Extract: (Apple, CEO, Tim Cook), (Tim Cook, announced, ...). Turn articles into database entries.

**Example:**
```
Information extraction pipeline:

Text: "Apple CEO Tim Cook announced the new iPhone 15
      at the September event in Cupertino."

1. Named Entity Recognition:
   Apple → ORG
   Tim Cook → PERSON
   iPhone 15 → PRODUCT
   September → DATE
   Cupertino → LOCATION

2. Relation Extraction:
   (Tim Cook, CEO_of, Apple)
   (iPhone 15, made_by, Apple)
   (Event, located_in, Cupertino)

3. Event Extraction:
   Event: Product_Announcement
   Agent: Tim Cook
   Product: iPhone 15
   Time: September
   Location: Cupertino

→ Structured knowledge base entries!
```
