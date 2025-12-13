import math
import collections
import random
import re
import zlib
import string

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text, level='char', keep_punctuation=False):
    """
    Preprocesses text.
    level: 'char' or 'word'
    keep_punctuation: bool
    Returns: list of tokens
    """
    text = text.lower()

    # Define allowed characters
    if keep_punctuation:
        # Allow letters, numbers, and standard punctuation
        # For simplicity, let's keep basic ascii punctuation and spaces
        allowed = set(string.ascii_lowercase + string.digits + string.punctuation + " \n\t")
    else:
        # Strict: only a-z and space
        allowed = set(string.ascii_lowercase + " ")
        text = text.replace('\n', ' ').replace('\t', ' ') # normalize whitespace for strict mode

    # Filter characters for both modes initially to clean up weird stuff
    # But for word mode, we might want to do regex split instead

    if level == 'char':
        # Filter char by char
        cleaned = [c for c in text if c in allowed]
        # Collapse multiple spaces if strict? The user didn't say, but it helps coherence.
        # Let's just join them.
        cleaned_str = "".join(cleaned)
        if not keep_punctuation:
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()
        return list(cleaned_str)

    elif level == 'word':
        if keep_punctuation:
            # Tokenize: words and punctuation are separate tokens
            # regex: capture words or non-whitespace characters
            # This is a simple tokenizer
            tokens = re.findall(r"[\w']+|[.,!?;:]", text)
            return tokens
        else:
            # Remove punctuation, split by whitespace
            # First replace punctuation with nothing (or space?)
            # Usually strict word model removes punctuation.
            cleaned_str = re.sub(r'[^a-z\s]', '', text)
            tokens = cleaned_str.split()
            return tokens

    return []

def split_chapters(text):
    """
    Splits text into chapters using heuristic 'CHAPTER [ROMAN/NUM]'
    Returns a list of strings (chapters).
    """
    # Gutenberg Alice often has "CHAPTER I", "CHAPTER II", etc.
    # We look for CHAPTER followed by something, until the next CHAPTER or end.

    # This regex looks for CHAPTER followed by Roman numerals or numbers
    pattern = r'(CHAPTER\s+[IVXLC0-9]+.*?)(?=CHAPTER\s+[IVXLC0-9]+|$)'
    # The text might have "CHAPTER I" ... content ... "CHAPTER II"
    # re.split includes the delimiter if capturing group is used, which complicates things.
    # Let's try finding all start indices.

    matches = list(re.finditer(r'CHAPTER\s+[IVXLC0-9]+', text, re.IGNORECASE))

    if not matches:
        return [text] # No chapters found, return whole text

    chapters = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        chapters.append(text[start:end])

    return chapters

def build_ngram_counts(tokens, order):
    """
    Builds counts of next_token given context of size 'order'.
    Returns: dict { context_tuple: { next_token: count } }
    """
    counts = collections.defaultdict(collections.Counter)

    # For order 0, context is empty tuple ()
    if order == 0:
        for token in tokens:
            counts[()][token] += 1
        return counts

    # For order > 0
    for i in range(len(tokens) - order):
        context = tuple(tokens[i : i+order])
        next_token = tokens[i+order]
        counts[context][next_token] += 1

    return counts

def normalize_to_probs(counts):
    """
    Converts counts to probabilities.
    Returns: dict { context_tuple: { next_token: prob } }
    """
    model = {}
    for context, counter in counts.items():
        total = sum(counter.values())
        model[context] = { token: count/total for token, count in counter.items() }
    return model

def generate_text(model, order, length, seed_context=None):
    """
    Generates a sequence of tokens.
    model: dict returned by normalize_to_probs
    order: int
    length: int (number of tokens)
    seed_context: tuple (optional)
    """
    # Setup initial context
    contexts = list(model.keys())
    if not contexts:
        return []

    if seed_context is None or seed_context not in model:
        # Pick a random starting context
        current_context = random.choice(contexts)
    else:
        current_context = seed_context

    generated = list(current_context) if order > 0 else []

    for _ in range(length):
        if current_context not in model:
            # Dead end (shouldn't happen often in large text if circular or large enough, but possible)
            break

        probs = model[current_context]
        candidates = list(probs.keys())
        weights = list(probs.values())

        next_token = random.choices(candidates, weights=weights, k=1)[0]
        generated.append(next_token)

        # Update context
        if order > 0:
            current_context = tuple(generated[-order:])

    return generated

def calculate_entropy_model(model):
    """
    Calculates theoretical entropy rate of the Markov model.
    H = Sum( P(context) * H(X|context) )
    We estimate P(context) from the counts implicitly if we had them,
    but here 'model' only has conditional probs.
    We need stationary distribution of contexts for exact calc,
    OR we can just weight by the frequency of contexts in the *training data*.

    Since we don't have the original counts passed here, let's assume
    the user passes counts to a helper or we change the signature.

    Let's change signature to take `counts` instead?
    Or we pass the sequence to estimate context probability?

    Let's implement: H = - Sum ( P(context) * Sum( P(x|context) * log2 P(x|context) ) )
    We need P(context). The empirical probability of context in the source text is the best estimator.
    """
    # This function is hard to implement correctly without the source counts to weight the contexts.
    # I'll modify the approach:
    #   Calculate conditional entropy for each context.
    #   Weight by occurence of that context.
    pass

def calculate_entropy_from_counts(counts):
    """
    H = - Sum_ctx ( P(ctx) * Sum_x ( P(x|ctx) * log2 P(x|ctx) ) )
    """
    total_observations = sum(sum(counter.values()) for counter in counts.values())

    entropy = 0.0
    for context, counter in counts.items():
        ctx_count = sum(counter.values())
        p_ctx = ctx_count / total_observations

        h_cond = 0.0
        for token, count in counter.items():
            p_x_given_ctx = count / ctx_count
            h_cond -= p_x_given_ctx * math.log2(p_x_given_ctx)

        entropy += p_ctx * h_cond

    return entropy

def calculate_nll(sequence, model, order):
    """
    Calculates Negative Log Likelihood (per symbol) on a sequence using the model.
    H ≈ - (1/N) * Sum( log2 P(x_i | context_i) )
    """
    n = 0
    log_prob_sum = 0.0

    # We can only evaluate where we have a full context and the transition exists
    # If transition is missing (zero prob), it blows up (infinity).
    # Usually we smooth or ignore. For this exercise, simple ignoring or epsilon?
    # README says "Empirical negative log-likelihood".
    # We will skip transitions that have 0 probability in the model (which means they weren't in training).
    # Ideally we evaluate on the TRAINING set (to check model entropy) or TEST set.
    # If evaluating on training set, 0 prob shouldn't happen.

    for i in range(len(sequence) - order):
        if order > 0:
            context = tuple(sequence[i : i+order])
        else:
            context = ()

        next_token = sequence[i+order]

        if context in model and next_token in model[context]:
            prob = model[context][next_token]
            log_prob_sum += math.log2(prob)
            n += 1

    if n == 0:
        return 0.0

    return -log_prob_sum / n

def calculate_gzip_ratio(original_text):
    """
    Returns compressed_size / original_size
    """
    if not original_text:
        return 0.0

    original_bytes = original_text.encode('utf-8')
    compressed_bytes = zlib.compress(original_bytes)

    return len(compressed_bytes) / len(original_bytes)
