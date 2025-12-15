import re
import math
import zlib
import time
import jieba
import string
import random
import collections
from tqdm import tqdm


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# HELPER FUNCTIONS
def is_han(char):
    """Checks if a character is a CJK Unified Ideograph."""
    return "\u4e00" <= char <= "\u9fff"


def simplify_whitespace(text):
    """
    Consolidates multiple spaces into one, and multiple newlines into one.
    Also ensures space doesn't exist around newlines for cleanliness.
    """
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def split_chapters(text):
    """
    Splits text into chapters using heuristic 'CHAPTER [ROMAN/NUM]'
    Returns a list of strings (chapters).
    """
    # Gutenberg Alice often has "CHAPTER I", "CHAPTER II", etc.
    # We look for CHAPTER followed by something, until the next CHAPTER or end.

    # This regex looks for CHAPTER followed by Roman numerals or numbers
    pattern = r"(CHAPTER\s+[IVXLC0-9]+.*?)(?=CHAPTER\s+[IVXLC0-9]+|$)"
    # The text might have "CHAPTER I" ... content ... "CHAPTER II"
    # re.split includes the delimiter if capturing group is used, which complicates things.
    # Let's try finding all start indices.

    matches = list(re.finditer(r"CHAPTER\s+[IVXLC0-9]+", text, re.IGNORECASE))

    if not matches:
        return [text]  # No chapters found, return whole text

    chapters = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append(text[start:end])

    return chapters


# CLEAN TEXT
def _clean_text_en(text, level, keep_punctuation, keep_whitespace):
    text = text.lower()
    text = text.replace("\t", " ")  # Tabs are always spaces

    # 1. Define sets
    # User requested: . , ? and linebreak
    SIMPLE_PUNCTUATION = {".", ",", "?", "\n"}

    # drop multiple spaces/newlines
    text = simplify_whitespace(text)

    # --- Character Level ---
    if level == "char":
        # Priority 1: Keep Punctuation
        if keep_punctuation:
            # Allow: a-z, space, and simple punctuation
            # Regex: Keep a-z, space, and chars in SIMPLE_PUNCTUATION. Replace others with space.
            # We construct regex safe string: \n . , ?
            text = re.sub(r"[^a-z \n.,?]", " ", text)

            # Simplify and tokenize

            return list(text.strip())

        # Priority 2: Keep Whitespace (No Punctuation)
        elif keep_whitespace:
            # Allow: a-z, space ONLY.
            # Drop ALL punctuation (.,?\n) by replacing with space
            text = re.sub(r"[^a-z ]", " ", text)
            return list(text.strip())

        # Priority 3: Strict (No nothing)
        else:
            # Allow: a-z ONLY.
            # We want a continuous stream of characters: "helloworld"
            # Remove everything that is not a-z
            text = re.sub(r"[^a-z]", "", text)
            return list(text)

    # --- Word Level ---
    elif level == "word":
        # Priority 1: Keep Punctuation
        if keep_punctuation:
            # Punctuation marks become distinct tokens
            text = re.sub(r"[^a-z \n.,?']", " ", text)  # Keep ' for possessives
            # Find words OR punctuation tokens
            tokens = re.findall(r"[\w']+|\n|\.|,|?", text)
            return tokens

        # Priority 2: Keep Whitespace (No Punctuation)
        elif keep_whitespace:
            # Punctuation removed. \n remains as a token.
            text = re.sub(r"[^a-z \n']", " ", text)
            # Find words OR newlines
            tokens = re.findall(r"[\w']+|\n", text)
            return tokens

        # Priority 3: Strict (Words only)
        else:
            # Words only. No \n tokens.
            text = re.sub(r"[^a-z\s]", "", text)
            return text.split()

    return []


def _clean_text_cn(text, level, keep_punctuation, keep_whitespace):
    CN_PUNCTUATION = {"。", "，", "？", "\n"}  # Added fullwidth ? (？)

    text = simplify_whitespace(text)

    if level == "char":
        tokens = list(text)
        cleaned = []

        for char in tokens:
            if keep_punctuation:
                # Priority 1: Han + Space + Punctuation
                if is_han(char) or char in CN_PUNCTUATION or char == " ":
                    cleaned.append(char)
                else:
                    cleaned.append(" ")  # Replace unknowns with space

            elif keep_whitespace:
                # Priority 2: Han + Space + \n (No Punc)
                if is_han(char) or char in {" ", "\n"}:
                    cleaned.append(char)
                else:
                    cleaned.append(" ")

            else:
                # Priority 3: Han Only
                if is_han(char):
                    cleaned.append(char)

        # Post-processing for 1 and 2
        if keep_punctuation or keep_whitespace:
            res = "".join(cleaned)
            return list(res.strip())
        else:
            return cleaned

    elif level == "word":
        tokens = jieba.lcut(text)
        cleaned = []

        for token in tokens:
            has_han = any(is_han(c) for c in token)

            if keep_punctuation:
                if has_han or token in CN_PUNCTUATION:
                    cleaned.append(token)

            elif keep_whitespace:
                # Keep words and newlines only
                if has_han or "\n" in token:
                    # Clean punctuation out of mixed tokens if necessary,
                    # but usually jieba separates them.
                    cleaned.append(token if token == "\n" else token.strip())

            else:
                # Keep words only
                if has_han:
                    cleaned.append(token)

        return cleaned

    return []


def clean_text(
    text, level="char", keep_punctuation=False, keep_whitespace=True, lang="en"
):
    """
    Unified preprocessing function.

    Priority Logic:
    1. keep_punctuation=True -> Keeps words + space + {.,?\n}
    2. keep_whitespace=True -> Keeps words + space + {\n} (No punctuation)
    3. Both False          -> Keeps words/chars only (No space, no \n, no punctuation)
    """
    if lang == "cn":
        return _clean_text_cn(text, level, keep_punctuation, keep_whitespace)
    else:
        return _clean_text_en(text, level, keep_punctuation, keep_whitespace)


# N-GRAM MODELING FUNCTIONS
def build_ngram_counts(tokens, order, show_pb=False, print_stats=False):
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
    start_time = time.time()
    for i in (
        tqdm(range(len(tokens) - order)) if show_pb else range(len(tokens) - order)
    ):
        context = tuple(tokens[i : i + order])
        next_token = tokens[i + order]
        counts[context][next_token] += 1
    elapsed = time.time() - start_time

    if print_stats:
        print_model_statistics(counts, order, elapsed)

    return counts


def normalize_to_probs(counts):
    """
    Converts counts to probabilities.
    Returns: dict { context_tuple: { next_token: prob } }
    """
    model = {}
    for context, counter in counts.items():
        total = sum(counter.values())
        model[context] = {token: count / total for token, count in counter.items()}
    return model


def print_model_statistics(counts, order, elapsed=None):
    """
    Calculates and prints key statistics from the N-gram counts dictionary.

    Args:
        counts (dict): The dictionary returned by build_ngram_counts.
        order (int): The order of the N-gram model (e.g., 2 for bigram).
    """
    if not counts:
        print("\n--- Model Statistics ---")
        print("The model is empty.")
        return

    # Total number of unique contexts (keys in the top level dictionary)
    num_contexts = len(counts)

    # Total number of observed N-grams (observations/transitions)
    # This is the sum of all counts in all inner counters.
    total_observations = sum(sum(counter.values()) for counter in counts.values())

    # Vocabulary size (V): total number of unique next tokens found across all contexts
    # This is often done globally, but here we estimate the unique next tokens.
    # To get the full vocabulary (all unique tokens used anywhere in the text),
    # we would need the original 'tokens' list.
    # Since we only have 'counts', we calculate the size of the *conditional* vocabulary.

    # Calculate the size of the conditional vocabulary (V_cond)
    unique_tokens = set()
    for counter in counts.values():
        unique_tokens.update(counter.keys())

    conditional_vocab_size = len(unique_tokens)

    # Number of unique N-grams (V_N): the number of unique (context, next_token) pairs
    unique_n_grams = sum(len(counter) for counter in counts.values())

    max_len = 40
    print("\n----- Model Statistics -----")
    print(f"{'Order of N-gram Model (N)':<{max_len}}: {order:,}")
    if elapsed is not None:
        print(f"{'Computation Time (s)':<{max_len}}: {elapsed:.4f}")
    print(f"{'Number of Unique Contexts:':<{max_len}}: {num_contexts:,}")
    print(
        f"{'Total Observed N-grams (Transitions)':<{max_len}}: {total_observations:,}"
    )
    print(f"{'Unique (Context, Token) N-grams':<{max_len}}: {unique_n_grams:,}")
    print(
        f"{'Conditional Vocab Size (Next Tokens)':<{max_len}}: {conditional_vocab_size:,}"
    )


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

    # Q = maybe i can use external word frequencies obtained from smwhere in the internet
    # and then weight the contexts by that ?

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


def calculate_nll(sequence, model, order, eps=1e-10):
    """
    Calculates Negative Log Likelihood (per symbol) on a sequence using the model.
    H ≈ - (1/N) * Sum( log2 P(x_i | context_i) )
    """
    n = 0
    log_prob_sum = 0.0

    # If transition is missing (zero prob), log(0) blows up to infinity
    # Usually we smooth or take epsilon
    # If evaluating on training set, 0 prob shouldn't happen

    for i in range(len(sequence) - order):
        if order > 0:
            context = tuple(sequence[i : i + order])
        else:
            context = ()

        next_token = sequence[i + order]

        if context in model and next_token in model[context]:
            prob = model[context][next_token]
            if prob <= 0:
                prob = eps
            log_prob_sum += math.log2(prob)
            n += 1

    if n == 0:
        return 0.0

    return -log_prob_sum / n


# OTHERS
def calculate_gzip_ratio(original_text):
    """
    Returns compressed_size / original_size
    """
    if not original_text:
        return 0.0

    original_bytes = original_text.encode("utf-8")
    compressed_bytes = zlib.compress(original_bytes)

    return len(compressed_bytes) / len(original_bytes)


def get_vocabulary_size(counts):
    """
    Calculates the size of the full vocabulary (V) based on the N-gram counts.
    V includes all unique tokens that appear as a context token or a next token.
    """
    vocabulary = set()

    # Add tokens from contexts (if order > 0)
    for context in counts.keys():
        if isinstance(context, tuple):  # For order > 0
            vocabulary.update(context)
        elif (
            context == ()
        ):  # For order 0 (unigram), context is empty tuple, only next tokens matter
            pass

    # Add tokens from next_token counters
    for counter in counts.values():
        vocabulary.update(counter.keys())

    return len(vocabulary)


def calculate_redundancy_model(counts, entropy_rate):
    """
    Calculates the redundancy of the source based on the N-gram model's entropy.

    Args:
        counts (dict): The dictionary returned by build_ngram_counts.
        entropy_rate (float): The entropy rate (H) calculated by a function
                              like calculate_entropy_from_counts.

    Returns:
        float: The redundancy (R) in bits/token.
    """
    vocab_size = get_vocabulary_size(counts)

    if vocab_size <= 1:
        # Cannot calculate H_max if vocab is 0 or 1
        return 0.0

    # Max Entropy H_max = log2(V)
    H_max = math.log2(vocab_size)

    # Redundancy R = H_max - H
    redundancy = H_max - entropy_rate

    # Redundancy is often expressed as a percentage of H_max: R/H_max
    # Redundancy_percent = (redundancy / H_max) * 100

    print("\n----- Redundancy Calculation (Based on N-gram Model) -----")
    print(f"Vocabulary Size (V): {vocab_size:,}")
    print(f"Maximum Entropy (H_max = log2 V): {H_max:.4f} bits/token")
    print(f"Model Entropy Rate (H): {entropy_rate:.4f} bits/token")
    print(f"Redundancy (R = H_max - H): {redundancy:.4f} bits/token")
    print(f"Redundancy Ratio (R / H_max): {redundancy / H_max:.4f}")

    return redundancy
