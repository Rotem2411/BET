import pandas as pd
import numpy as np
import pyarrow, fastparquet
import glob
from category import get_sentence_embedding, build_category_dictionary
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import BertTokenizerFast


def doc2sent(text, alephbert_tokenizer):
    """
    Tokenizes a document into sentences, ensuring each chunk fits within the model's token limit.

    Args:
        text (str): The input document.
        alephbert_tokenizer (BertTokenizerFast): Tokenizer for Hebrew text.

    Returns:
        list: List of tokenized sentences.
    """
    max_token_limit = 512  # Maximum tokens allowed by the model

    # Tokenize the text
    tokens = alephbert_tokenizer.tokenize(text)

    # Filter tokens: Allow Hebrew characters, numbers, and exclude certain special cases
    filtered_tokens = [
        token for token in tokens
        if token not in {'br', '-', '<', '>'}
    ]

    # Split tokens into chunks of max_token_limit
    chunks = [filtered_tokens[i:i + max_token_limit] for i in range(0, len(filtered_tokens), max_token_limit)]

    # Convert each chunk back into a string
    sentences = [alephbert_tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    # Further split sentences using punctuation marks
    processed_sentences = []
    for sentence in sentences:
        split_sentences = re.split(r'[.!?]', sentence)
        processed_sentences.extend([s.strip() for s in split_sentences if s.strip()])

    return processed_sentences

def compute_category_similarity(row, sentence_embeddings, index_category):
    """
    Computes similarity between a sentence and predefined category embeddings.

    Args:
        row (pd.Series): A row from the category dataframe.
        sentence_embeddings (numpy.ndarray): Array of sentence embeddings.
        index_category (dict): Dictionary mapping category indices to names.

    Returns:
        tuple: Column name and similarity scores.
    """
    try:
        category = index_category[int(row.category_index)]
        word = row.word
        category_embedding = row.embedding.reshape(1, -1)
        col_name = f"{category}_{word}"

        # Calculate similarities
        similarities = np.round(cosine_similarity(category_embedding, sentence_embeddings).flatten(), 3)
        return col_name, similarities
    except Exception as e:
        print(f"Error Processing row {row}: {e}")
        return None, None

def merge_similarity_scores(df, df_similarity, category_dict):
    """
    Merges similarity scores by computing the highest similarity per category.

    Args:
        df (pd.DataFrame): The input dataset.
        df_similarity (pd.DataFrame): The dataset containing similarity scores.
        category_dict (dict): Dictionary of categories.

    Returns:
        pd.DataFrame: Updated dataframe with max similarity values.
    """
    print("start merge_similarity_scores")
    # Initialize a dictionary to store max similarity values for each main category
    max_similarities = {category: [] for category in category_dict}

    # Iterate over each sentence to compute max similarity per main category
    for _, row in df_similarity.iterrows():
        for category in category_dict:
            # Filter columns corresponding to the current main category
            category_cols = [col for col in df_similarity.columns if col.startswith(category + '_')]
            # Compute max similarity for the current main category
            max_value = row[category_cols].max()
            max_similarities[category].append(max_value)

    # Add the max similarity values as new columns to the DataFrame
    for category, values in max_similarities.items():
        df[category] = values

    # Concatenate the new columns to the existing DataFrame
    df.drop(['sentence','embedding'], axis=1, inplace=True)
    print("merge_similarity_scores ends")
    return df

def compute_sentence_similarity(batch_size=100000):
    """
    Computes sentence similarity in batches and saves results to a file.

    Args:
        batch_size (int, optional): Number of sentences per batch. Defaults to 100000.

    Returns:
        pd.DataFrame: The computed sentence similarity dataset.
    """
    print("Starting compute_sentence_similarity")

    # Load metadata
    df_category = pd.read_parquet('category_metadata.parquet')
    df_sentences = pd.read_parquet('sentences_metadata.parquet')

    # Load embeddings
    category_embeddings = np.load('category_embeddings.npy')
    sentence_embeddings = np.load('sentences_embeddings.npy')

    # Attach embeddings back to DataFrames
    df_category['embedding'] = list(category_embeddings)
    df_sentences['embedding'] = list(sentence_embeddings)

    # Create the category dictionary
    category_dict = build_category_dictionary()
    index_category = {idx: cat for idx, cat in enumerate(category_dict)}

    try:
        df_similarity = pd.read_parquet('sentence_similarity.parquet')

    except:
        try:
            df_similarity = pd.read_parquet('similarity.parquet')

        except:
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

            # Create a new DataFrame from results
            df_similarity = pd.DataFrame(results)

            # Save similarity results as Parquet
            df_similarity.to_parquet('similarity.parquet', index=False, compression='snappy')

            print("Saved similarity results as Parquet.")

        # Calculate the number of batches needed
        num_batches = len(df_sentences) // batch_size + (1 if len(df_sentences) % batch_size != 0 else 0)

        for batch_num in range(num_batches):
            # Determine the start and end indices for the batch
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df_sentences))

            # Get the batch of sentences
            df_sentences_batch = df_sentences.iloc[start_idx:end_idx]
            df_similarity_batch = df_similarity.iloc[start_idx:end_idx]

            result_df = merge_similarity_scores(df_sentences_batch, df_similarity_batch, category_dict)

            # Save each batch result to a different file, you can name the files with the batch number
            result_file = f"sentence_similarity_batch_{batch_num + 1}.parquet"
            result_df.to_parquet(result_file)

            print(f"Batch {batch_num + 1} saved to {result_file}")
        df_similarity = merge_similarity_batches("sentence_similarity.parquet")

    df_similarity = add_top_emotions_to_dataset(df_similarity)
    df_similarity.to_parquet('sentence_similarity.parquet')
    print("compute_sentence_similarity ends")
    return df_similarity

def generate_sentence_embeddings(df, model, tokenizer, device):
    """
    Generates sentence embeddings for a dataset.

    Args:
        df (pd.DataFrame): The input dataset.
        model (object): Pre-trained embedding model.
        tokenizer (object): Tokenizer for text.
        device (str): Device to run computations on (CPU/GPU).

    Returns:
        tuple: Dataframe of sentences and their embeddings.
    """
    print("starting generate_sentence_embeddings")
    alephbert_tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")
    # Create a DataFrame with specified documents
    df_docs = df[['id', 'story1']].rename(columns={'story1': 'content'})
    rows = []
    # Populate rows by iterating through sampled documents
    for _, document in df_docs.iterrows():
        for j, sentence in enumerate(doc2sent(document['content'], alephbert_tokenizer)):
            rows.append({'document_id': document['id'], 'sentence_id': j, 'sentence': sentence})

    df_sentences = pd.DataFrame(rows)
    print(f'num of sentences:', len(df_sentences))
    df_sentences.to_parquet('sentences_metadata.parquet', index=False, compression='snappy')

    # Add embeddings column and save them separately
    embeddings = []
    for sentence in df_sentences['sentence']:
        embeddings.append(get_sentence_embedding(sentence, model, tokenizer, device))
    np.save('sentences_embeddings.npy', np.array(embeddings))

    print("Saved sentence metadata as Parquet and embeddings as Numpy array.\n")
    print("generate_sentence_embeddings ends")
    return df_sentences, embeddings

def merge_similarity_batches(output_filename="sentence_similarity.parquet"):
    # Get a list of all batch files
    batch_files = glob.glob("sentence_similarity_batch_*.parquet")

    if not batch_files:
        print("No batch files found!")
        return

    # Read all batch files and concatenate them into a single DataFrame
    combined_df = pd.concat([pd.read_parquet(file) for file in batch_files])

    # Save the combined DataFrame to a single file
    combined_df.to_parquet(output_filename)

    print(f"Combined results saved to {output_filename}")
    return combined_df

def add_top_emotions_to_dataset(df):
    # Drop non-emotion columns to work only with emotion scores
    emotion_columns = [col for col in df.columns if col not in ['document_id', 'sentence_id', 'embedding']]

    # Function to extract top 5 emotions for a row
    def get_top_emotions(row):
        emotions = row[emotion_columns].to_dict()
        top_emotions = []
        top_scores = []
        for _ in range(5):
            # Find the max emotion and its score
            max_emotion = max(emotions, key=emotions.get)
            max_score = emotions[max_emotion]

            # Append to the results
            top_emotions.append(max_emotion)
            top_scores.append(max_score)

            # Remove the max emotion to find the next max in the next iteration
            emotions.pop(max_emotion)

        return top_emotions, top_scores

    # Apply the function to each row and store the results in new columns

    results = df.apply(lambda row: get_top_emotions(row), axis=1)
    df['top_emotions'] = results.apply(lambda x: x[0])
    df['top_emotion_scores'] = results.apply(lambda x: x[1])

    return df