# English Text as Markov Chain & Entropy Rate

## 1. Project Overview

This project investigates English as a stochastic process modeled using Markov chains of varying order. The goals are to understand:

- how dependencies between characters (or words) govern the structure of English,
- how these dependencies affect the entropy rate (average unpredictability per symbol),
- how this connects to Shannon’s Asymptotic Equipartition Property (AEP),
- and what these results imply for lossless compression.

This project extends concepts from Lecture 3 (AEP, entropy rate, Markov chains, stationary processes) with empirical experiments on real text.

## 2. Objectives

### Core Objectives

1. Build order-k Markov models (k = 0, 1, 2, 3, …).
2. Generate artificial English text from each model.
3. Estimate the entropy rate using:
   - theoretical conditional entropy formulas,
   - empirical negative log-likelihood.
4. Compare entropy across model orders and datasets.
5. Interpret results in relation to:
   - AEP,
   - Shannon’s source coding theorem,
   - compressibility of language.

### Extended Objectives (Optional)

1. Compare character-level and word-level models.
2. Compare entropy rate estimates with real compression ratios (gzip).
3. Evaluate non-stationarity: entropy per chapter.
4. Plot convergence of `-(1/n) log P(X_1^n)` toward the entropy rate.
5. Test multiple corpora or multiple languages.
6. Explore high-order models and sparsity effects.

## 3. Hypotheses

### H1. Higher-order Markov models reduce entropy.
As k increases, the model captures more context. Expected trend:

- H₀ (i.i.d.) ≈ 4.0 bits/char  
- H₁ ≈ 3.0–3.3 bits/char  
- H₂ ≈ 2.3–2.8 bits/char  
- H₃ ≈ 1.7–2.1 bits/char  

### H2. Generated text becomes more coherent with higher k.
- k = 0 → nonsense  
- k = 1 → correct frequency distribution  
- k = 2 → plausible bigrams  
- k = 3 → locally coherent English  

### H3. Negative log-likelihood converges to the entropy rate (AEP).

### H4. Lower entropy correlates with better gzip compression ratios.

## 4. Dataset

Use any public-domain English text (Project Gutenberg):

- *Alice’s Adventures in Wonderland*  
- *Pride and Prejudice*  
- Sherlock Holmes stories  
- King James Bible  

Place your chosen file at: data/corpus.txt

## 5. Implementation Plan

### 5.1 Preprocessing
- convert text to lowercase  
- remove unsupported characters  
- convert to a sequence of characters (or words)

### 5.2 N-gram Counting (Order k)
For each index `i`:

context = sequence[i-k : i]
next_symbol = sequence[i]
count[context][next_symbol] += 1

### 5.3 Probability Normalization
Convert counts to conditional probabilities `P(next | context)`.

### 5.4 Text Generation
Start from an initial context and repeatedly sample:

X_{t+1} ~ P(· | X_{t-k+1:t})

Generate 200–1000 characters for demonstration.

### 5.5 Entropy Rate Estimation

#### Method A: Model-based conditional entropy

H = Σ_context π(context) * H(X | context)

#### Method B: Negative log-likelihood (AEP)

H ≈ -(1/n) Σ log2 P(x_i | x_{i-k:i-1})

## 6. Extensions

### E1. Character-level vs Word-level Models
Compute entropy in bits/char and bits/word.

### E2. Compression Comparison
Compute:

gzip_size / original_size

Compare to entropy estimates.

### E3. Entropy per Chapter
Demonstrate non-stationarity in language.

### E4. Convergence of Hₙ
Plot:

H_n = -(1/n) log2 P(X_1^n)

### E5. Higher-order Models
Analyze sparsity and overfitting.

## 7. Expected Output

### Tables
| Order k | H_model | H_sequence | gzip ratio | Notes |
|--------:|---------|------------|------------|-------|

### Plots
- Entropy rate vs model order  
- Generated text examples  
- Convergence of `H_n`  
- Context frequency distribution  

### Generated Samples
Include excerpts for k = 0, 1, 2, 3.

## 8. Project Structure

```
project/
│
├── README.md
├── main.py
├── utils.py
├── data/
│   └── corpus.txt
├── results/
│   ├── generated_k1.txt
│   ├── entropy_plot.png
│   └── convergence_plot.png
└── plots/
```

## 9. Code Outline

### Build model
```python
counts = build_ngram_counts(sequence, order=k)
model = normalize_to_probs(counts)

# Generate text

generated = generate_sequence(model, order=k, length=500)

# Entropy estimation

H_model = estimate_entropy_rate_from_model(sequence, model, k)
H_seq   = estimate_entropy_rate_from_sequence(sequence, model, k)
```

## 10. Interpretation Goals

Discuss:
- how dependency structure reduces entropy,
- why English is compressible,
- how empirical results validate the AEP,
- differences between low and high-order models,
- limitations of Markov chain modeling.

## 11. Limitations
- English text is not perfectly stationary.
- High-order models suffer from sparse statistics.
- Markov models capture local but not global structure.
- True entropy rate is lower than these simple estimates.
