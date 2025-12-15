import math
import collections
import random
import re
import zlib
import string


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# REVISED to keep linebreaks
def clean_text(text, level="char", keep_punctuation=False):
    """
    Preprocesses text.
    level: 'char' or 'word'
    keep_punctuation: bool
    Returns: list of tokens
    """
    text = text.lower()

    # Define allowed characters
    if keep_punctuation:
        # Allow letters, numbers, and standard punctuation, plus \n and \t
        allowed = set(
            string.ascii_lowercase + string.digits + string.punctuation + " \n\t"
        )
    else:
        # Strict: only a-z and space
        allowed = set(string.ascii_lowercase + " ")
        text = text.replace("\n", " ").replace(
            "\t", " "
        )  # normalize whitespace for strict mode

    if level == "char":
        cleaned = [c for c in text if c in allowed]

        if keep_punctuation:
            # FIX: Preserve all characters as tokens, including \n and \t, by returning the list directly
            return cleaned
        else:
            # For strict char mode, join and then normalize spaces
            cleaned_str = "".join(cleaned)
            cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
            return list(cleaned_str)

    elif level == "word":
        if keep_punctuation:
            # FIX: Explicitly include \n and \t as distinct tokens in the regex capture group
            text = text.replace(
                "\t", "\n"
            )  # Treat tabs like newlines for simplicity in modeling breaks

            # The pattern is: [Words] OR [\n] OR [Other Punctuation]
            tokens = re.findall(r"[\w']+|\n|[.,!?;:\"()]", text)

            return tokens
        else:
            # Strict word mode: remove punctuation, split by whitespace
            cleaned_str = re.sub(r"[^a-z\s]", "", text)
            tokens = cleaned_str.split()
            return tokens

    return []


# Helper function for simplification
def simplify_newlines(text):
    """
    Replaces sequences of one or more newline characters (\n) with a single newline.
    """
    # This regex substitutes two or more newlines with a single newline.
    return re.sub(r"\n+", "\n", text)


# REVISED to only keep simple punctuation . , \n
# def clean_text(text, level="char", keep_punctuation=False):
#     """
#     Preprocesses text.
#     level: 'char' or 'word'
#     keep_punctuation: bool
#     Returns: list of tokens
#     """
#     text = text.lower()

#     # Define the simple punctuation set: . , \n (and space, always)
#     SIMPLE_PUNCTUATION = {".", ",", "\n"}

#     # --- Character Level ---
#     if level == "char":
#         if keep_punctuation:
#             # 1. Filtering and cleaning
#             allowed = set(string.ascii_lowercase + string.digits + " \n\t").union(
#                 {".", ","}
#             )
#             # Consolidate tabs into newlines
#             text = text.replace("\t", "\n")

#             # Filter all characters to only keep allowed ones
#             cleaned = [
#                 c
#                 for c in text
#                 if c in allowed
#                 and c not in set(string.punctuation) - SIMPLE_PUNCTUATION
#             ]

#             # 2. Simplification of multiple newlines
#             cleaned_str = "".join(cleaned)
#             cleaned_str = simplify_newlines(
#                 cleaned_str
#             )  # <<< NEWLINE SIMPLIFICATION APPLIED HERE

#             # Return the list of characters directly, preserving '\n' as a token
#             return list(cleaned_str)  # Convert back to list of character tokens

#         else:
#             # Strict: only a-z and space
#             allowed = set(string.ascii_lowercase + " ")
#             text = text.replace("\n", " ").replace("\t", " ")
#             cleaned = [c for c in text if c in allowed]
#             cleaned_str = "".join(cleaned)
#             cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
#             return list(cleaned_str)

#     # --- Word Level ---
#     elif level == "word":
#         if keep_punctuation:
#             text = text.lower()

#             # B. Remove all punctuation *except* . and ,
#             all_punc_to_remove = set(string.punctuation) - {".", ","}
#             punc_pattern = r"[" + re.escape("".join(all_punc_to_remove)) + r"]"
#             text = re.sub(punc_pattern, " ", text)

#             # C. Replace all tabs with a single space (or \n if you want to model \t as \n)
#             text = text.replace("\t", " ")

#             # 1. Simplification of multiple newlines
#             text = simplify_newlines(text)  # <<< NEWLINE SIMPLIFICATION APPLIED HERE

#             # D. Use findall to explicitly capture our desired tokens.
#             tokens = re.findall(r"[\w']+|\n|\.|\,", text)

#             # E. Filter out any remaining single space tokens or empty strings that might result from the regex
#             return [t for t in tokens if t.strip() or t == "\n"]

#         else:
#             # Strict word mode: remove punctuation, split by whitespace
#             cleaned_str = re.sub(r"[^a-z\s]", "", text)
#             tokens = cleaned_str.split()
#             return tokens

#     return []


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
        context = tuple(tokens[i : i + order])
        next_token = tokens[i + order]
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
        model[context] = {
            token: round(count / total, 4) for token, count in counter.items()
        }
    return model


def print_model_statistics(counts, order):
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

    # Prepare statistics for printing
    stats = [
        ("Order of N-gram Model (N)", order),
        ("Number of Unique Contexts", num_contexts),
        ("Total Observed N-grams (Transitions)", total_observations),
        ("Unique (Context, Token) N-grams", unique_n_grams),
        ("Conditional Vocab Size (Next Tokens)", conditional_vocab_size),
    ]

    # Find the max len of the label strings for alignment
    max_label_len = max(len(label) for label, _ in stats) + 1

    print("\n----- Model Statistics -----")
    # 3. Print the statistics using f-string padding and formatting
    for label, value in stats:
        # Use < for left alignment of the label, padded to max_label_len
        # Use :, for comma separated thousands for the value
        # Using LaTeX for labels as requested in your profile
        # Use an f-string for the whole output
        print(f"{label:<{max_label_len}}: {value:,}")


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
            context = tuple(sequence[i : i + order])
        else:
            context = ()

        next_token = sequence[i + order]

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

    original_bytes = original_text.encode("utf-8")
    compressed_bytes = zlib.compress(original_bytes)

    return len(compressed_bytes) / len(original_bytes)
