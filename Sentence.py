import pandas as pd
import numpy as np
import glob
from category import get_sentence_embedding, build_category_dictionary
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import BertTokenizerFast
from nltk.tokenize import sent_tokenize
import os
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

def doc2sent(text, alephbert_tokenizer):
    max_token_limit = 512  # Maximum tokens allowed by the model
    tokens = alephbert_tokenizer.tokenize(text)
    filtered_tokens = [
        token for token in tokens
        if token not in {'br', '-', '<', '>'}]
    chunks = [filtered_tokens[i:i + max_token_limit] for i in range(0, len(filtered_tokens), max_token_limit)]
    sentences = [alephbert_tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
    processed_sentences = []
    for sentence in sentences:
        split_sentences = re.split(r'[.!?]', sentence)
        processed_sentences.extend([s.strip() for s in split_sentences if s.strip()])
    return processed_sentences

def compute_category_similarity(row, sentence_embeddings, index_category):
    try:
        category = index_category[int(row.category_index)]
        word = row.word
        category_embedding = row.embedding.reshape(1, -1)
        col_name = f"{category}_{word}"
        similarities = np.round(cosine_similarity(category_embedding, sentence_embeddings).flatten(), 3)
        return col_name, similarities
    except Exception as e:
        print(f"Error Processing row {row}: {e}")
        return None, None

def merge_similarity_scores(df, df_similarity, category_dict):
    print("start merge_similarity_scores")
    max_similarities = {category: [] for category in category_dict}
    for _, row in df_similarity.iterrows():
        for category in category_dict:
            category_cols = [col for col in df_similarity.columns if col.startswith(category + '_')]
            max_value = row[category_cols].max()
            max_similarities[category].append(max_value)

    for category, values in max_similarities.items():
        df[category] = values

    df.drop(['sentence','embedding'], axis=1, inplace=True)
    print("merge_similarity_scores ends")
    return df

def compute_sentence_similarity(lang, batch_size=100000):
    print("Starting compute_sentence_similarity")
    # Load metadata and embeddings from the output directory
    category_metadata_path = os.path.join(OUTPUT_DIR, 'category_metadata.parquet')
    sentences_metadata_path = os.path.join(OUTPUT_DIR, 'sentences_metadata.parquet')
    category_embeddings_path = os.path.join(OUTPUT_DIR, 'category_embeddings.npy')
    sentences_embeddings_path = os.path.join(OUTPUT_DIR, 'sentences_embeddings.npy')
    df_category = pd.read_parquet(category_metadata_path)
    df_sentences = pd.read_parquet(sentences_metadata_path)
    category_embeddings = np.load(category_embeddings_path)
    sentence_embeddings = np.load(sentences_embeddings_path)
    df_category['embedding'] = list(category_embeddings)
    df_sentences['embedding'] = list(sentence_embeddings)
    category_dict = build_category_dictionary(lang)
    index_category = {idx: cat for idx, cat in enumerate(category_dict)}
    try:
        df_similarity = pd.read_parquet(os.path.join(OUTPUT_DIR, 'sentence_similarity.parquet'))
        print("Load sentence_similarity.parquet")
    except Exception:
        try:
            df_similarity = pd.read_parquet(os.path.join(OUTPUT_DIR, 'similarity.parquet'))
            print("Load similarity.parquet")
        except Exception:
            # Multithreaded computation
            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        compute_category_similarity,
                        row, sentence_embeddings, index_category
                    )
                    for row in df_category.itertuples(index=False)
                ]
                for future in futures:
                    col_name, similarities = future.result()
                    if col_name and similarities is not None:
                        results[col_name] = similarities
            df_similarity = pd.DataFrame(results)
            df_similarity.to_parquet(os.path.join(OUTPUT_DIR, 'similarity.parquet'), index=False, compression='snappy')
            print("Saved similarity results as Parquet.")

        num_batches = len(df_sentences) // batch_size + (1 if len(df_sentences) % batch_size != 0 else 0)
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df_sentences))
            df_sentences_batch = df_sentences.iloc[start_idx:end_idx]
            df_similarity_batch = df_similarity.iloc[start_idx:end_idx]
            result_df = merge_similarity_scores(df_sentences_batch, df_similarity_batch, category_dict)
            result_file = os.path.join(OUTPUT_DIR, f"sentence_similarity_batch_{batch_num + 1}.parquet")
            result_df.to_parquet(result_file)
            print(f"Batch {batch_num + 1} saved to {result_file}")
        # After processing batches, merge them
        df_similarity = merge_similarity_batches('sentence_similarity.parquet')
        df_similarity = add_top_emotions_to_dataset(df_similarity)
        df_similarity.to_parquet(os.path.join(OUTPUT_DIR, 'sentence_similarity.parquet'))
    print("compute_sentence_similarity ends")
    return df_similarity

def generate_sentence_embeddings(df, model=None, tokenizer=None, device='cpu'):
    sentences_metadata_path = os.path.join(OUTPUT_DIR, 'sentences_metadata.parquet')
    sentences_embeddings_path = os.path.join(OUTPUT_DIR, 'sentences_embeddings.npy')
    try:
        df_sentences = pd.read_parquet(sentences_metadata_path)
        embeddings = np.load(sentences_embeddings_path)
        print("Load sentences metadata and embeddings")
    except Exception:
        print("starting generate_sentence_embeddings")
        if model == 'twitter-xlm-roberta-base-sentiment':
            lang='english'
        else:
            lang='hebrew'
            alephbert_tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")

        rows = []
        if lang == 'hebrew':
            # For Hebrew documents we rely on the AlephBERT tokenizer to split sentences.
            # Each sentence gets a unique sentence_id within its document.
            for _, document in df.iterrows():
                for j, sentence in enumerate(doc2sent(document['text'], alephbert_tokenizer)):
                    rows.append({
                        'document_id': document['document_id'],
                        'sentence_id': j,
                        'sentence': sentence
                    })
        else:
            # For English documents use NLTK's sent_tokenize to split on punctuation.
            # Use the 'document_id' column to ensure consistency with import_dataset.
            for _, document in df.iterrows():
                for j, sentence in enumerate(sent_tokenize(document['text'])):
                    rows.append({
                        'document_id': document['document_id'],
                        'sentence_id': j,
                        'sentence': sentence
                    })
        df_sentences = pd.DataFrame(rows)
        print(f'num of sentences:', len(df_sentences))
        df_sentences.to_parquet(sentences_metadata_path, index=False, compression='snappy')
        embeddings = []
        for sentence in df_sentences['sentence']:
            embeddings.append(get_sentence_embedding(sentence, model, tokenizer, device))
        np.save(sentences_embeddings_path, np.array(embeddings))
        print("Saved sentence metadata as Parquet and embeddings as Numpy array.\n")
        print("generate_sentence_embeddings ends")
    return df_sentences, embeddings

def merge_similarity_batches(output_filename="sentence_similarity.parquet"):
    """Merge all batch files in the output directory into a single parquet file."""
    batch_files = glob.glob(os.path.join(OUTPUT_DIR, "sentence_similarity_batch_*.parquet"))
    if not batch_files:
        print("No batch files found!")
        return
    combined_df = pd.concat([pd.read_parquet(file) for file in batch_files])
    if os.path.isabs(output_filename) or os.path.dirname(output_filename):
        output_path = output_filename
    else:
        output_path = os.path.join(OUTPUT_DIR, output_filename)
    combined_df.to_parquet(output_path)
    print(f"Combined results saved to {output_path}")
    return combined_df

def add_top_emotions_to_dataset(df):
    emotion_columns = [col for col in df.columns if col not in ['document_id', 'sentence_id', 'embedding']]

    # Function to extract top 5 emotions for a row
    def get_top_emotions(row):
        emotions = row[emotion_columns].to_dict()
        top_emotions = []
        top_scores = []
        for _ in range(5):
            max_emotion = max(emotions, key=emotions.get)
            max_score = emotions[max_emotion]
            top_emotions.append(max_emotion)
            top_scores.append(max_score)
            emotions.pop(max_emotion)
        return top_emotions, top_scores

    # Apply the function to each row and store the results in new columns
    results = df.apply(lambda row: get_top_emotions(row), axis=1)
    df['top_emotions'] = results.apply(lambda x: x[0])
    df['top_emotion_scores'] = results.apply(lambda x: x[1])
    return df