import utils
import pandas as pd
import os

def run_experiments():
    corpus_path = 'data/corpus.txt'
    raw_text = utils.load_text(corpus_path)

    # Configurations
    configs = [
        {'level': 'char', 'keep_punct': False, 'orders': [0, 1, 2, 3, 4]},
        {'level': 'char', 'keep_punct': True,  'orders': [0, 1, 2, 3, 4]},
        {'level': 'word', 'keep_punct': False, 'orders': [0, 1, 2]},
        {'level': 'word', 'keep_punct': True,  'orders': [0, 1, 2]}
    ]

    results = []

    # Output file for samples
    samples_file = open('results/generated_samples.txt', 'w', encoding='utf-8')

    print("Starting experiments...")

    for config in configs:
        level = config['level']
        keep_punct = config['keep_punct']
        orders = config['orders']

        print(f"Processing: Level={level}, Punct={keep_punct}")

        # Preprocess
        tokens = utils.clean_text(raw_text, level=level, keep_punctuation=keep_punct)

        # Calculate GZIP ratio of the CLEANED text (to compare apples to apples with entropy of that text)
        # We need to reconstruct the string from tokens for GZIP
        if level == 'char':
            reconstructed_text = "".join(tokens)
        else:
            reconstructed_text = " ".join(tokens)

        gzip_ratio = utils.calculate_gzip_ratio(reconstructed_text)

        for k in orders:
            print(f"  Order k={k}")

            # Build Model
            counts = utils.build_ngram_counts(tokens, k)
            model = utils.normalize_to_probs(counts)

            # Calculate Entropy (Model based)
            h_model = utils.calculate_entropy_from_counts(counts)

            # Calculate Entropy (Sequence based - NLL)
            h_seq = utils.calculate_nll(tokens, model, k)

            # Generate Text
            gen_len = 200 if level == 'char' else 50
            generated_tokens = utils.generate_text(model, k, gen_len)

            if level == 'char':
                gen_text = "".join(generated_tokens)
            else:
                gen_text = " ".join(generated_tokens)

            # Write sample
            header = f"--- Level: {level}, Punct: {keep_punct}, Order: {k} ---\n"
            samples_file.write(header)
            samples_file.write(gen_text + "\n\n")

            # Record results
            results.append({
                'level': level,
                'keep_punct': keep_punct,
                'order': k,
                'h_model': h_model,
                'h_seq': h_seq,
                'gzip_ratio': gzip_ratio
            })

    samples_file.close()

    # Save metrics
    df = pd.DataFrame(results)
    df.to_csv('results/metrics.csv', index=False)
    print("Metrics saved to results/metrics.csv")

    # --- Chapter Analysis (Extension E3) ---
    print("Running Chapter Analysis...")
    chapters = utils.split_chapters(raw_text)
    print(f"Found {len(chapters)} chapters.")

    chapter_results = []
    # Analyze entropy per chapter for a fixed config (e.g., Char, Punct, k=2)
    # Ideally we train on the whole text and evaluate NLL on chapters,
    # OR train on each chapter?
    # README says "Evaluate non-stationarity: entropy per chapter."
    # Usually this means measuring the entropy OF that chapter.
    # We can measure it by:
    # 1. Training a model on that chapter and finding its entropy.
    # 2. Or using a global model and finding NLL of that chapter (surprisal).
    # "Non-stationarity" implies the statistics change.
    # Let's do both? Or just method 1 (simpler interpretation: "This chapter is more complex").
    # Method 2 (Cross-entropy) shows how much the chapter deviates from the global average.
    # Let's do Method 1 (Self-Entropy) for k=1 (simple)

    target_config = {'level': 'char', 'keep_punct': True, 'order': 1}

    for i, chapter_text in enumerate(chapters):
        tokens = utils.clean_text(chapter_text, level=target_config['level'], keep_punctuation=target_config['keep_punct'])
        if not tokens:
            continue

        k = target_config['order']
        counts = utils.build_ngram_counts(tokens, k)
        # model = utils.normalize_to_probs(counts) # Not needed for entropy_from_counts

        h_chapter = utils.calculate_entropy_from_counts(counts)

        chapter_results.append({
            'chapter': i+1,
            'entropy': h_chapter,
            'length': len(tokens)
        })

    df_chap = pd.DataFrame(chapter_results)
    df_chap.to_csv('results/chapter_entropy.csv', index=False)
    print("Chapter entropy saved to results/chapter_entropy.csv")

if __name__ == "__main__":
    run_experiments()
