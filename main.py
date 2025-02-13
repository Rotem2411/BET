from category import generate_category_embeddings
from Sentence import compute_sentence_similarity, generate_sentence_embeddings
from Topics import analyze_topics_in_data
from figures import generate_visual_insights
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast

def run_pipeline():
    """
    Executes the full pipeline: loading data, computing embeddings, extracting topics, and generating insights.
    """
    try:
        df_merged = pd.read_parquet('merged_data.parquet')
        # Check for GPU availability
        if torch.cuda.is_available():
            print("Using GPU for computation")
            #torch.cuda.set_device(1)  # Use GPU 1
            device = "cuda:1" # Use GPU 1
        else:
            print("Using CPU for computation")
            device = "cpu"

        """
        model_name = "onlplab/alephbert-base"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
        """
        tokenizer = None
        sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        origin_dataset_name = 'finaid_applications_2011_2024.xlsx'
        df_original = import_dataset(origin_dataset_name)
        df_category, category_embedding = generate_category_embeddings(sentence_model, tokenizer, device)
        df_sentences, sentences_embedding = generate_sentence_embeddings(df_original, sentence_model, tokenizer, device)
        df_similarity = compute_sentence_similarity(batch_size=100000)
        df_topic = analyze_topics_in_data(df_original)
        df_merged = integrate_topic_and_similarity_data(df_similarity, df_topic)
        df_docs, emotion_columns = aggregate_document_emotions(df_merged)

        generate_visual_insights(df_docs)

    except Exception as e:
        print(f"Error during main: {e}")

def import_dataset(dataset_name):
    """
    Loads a dataset from an Excel file.

    Args:
        dataset_name (str): Name of the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # Load the data
    df_original = pd.read_excel(dataset_name)
    documents = df_original.story1.tolist()
    print(f"num of documents:", len(documents))
    return df_original

def integrate_topic_and_similarity_data(df_similarity, df_topic):
    """
    Merges topic and sentence similarity data into a unified dataset.

    Args:
        df_similarity (pd.DataFrame): Sentence similarity dataset.
        df_topic (pd.DataFrame): Topic model dataset.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    merged_df = pd.merge(df_similarity, df_topic, on='document_id', how='inner')
    # Reordering the columns
    columns_order = ['document_id', 'sentence_id', 'Topic_number', 'year'] + [col for col in merged_df.columns if
                                                                              col not in ['year', 'Topic_number',
                                                                                          'document_id',
                                                                                          'sentence_id']]
    merged_df = merged_df[columns_order]
    merged_df.to_csv('merged_data.csv', index=False)
    merged_df.to_parquet('merged_data.parquet', index=False)
    return merged_df

def aggregate_document_emotions(df_merged):
    """
    Computes aggregate emotional scores for documents.

    Args:
        df_merged (pd.DataFrame): Merged dataset containing emotion scores.

    Returns:
        tuple: Dataframe with document emotion scores and list of emotion columns.
    """
    print('start aggregate_document_emotions')
    # Drop non-emotion columns to work only with emotion scores
    emotion_columns = [col for col in df_merged.columns if
                       col not in ['document_id', 'sentence_id', 'Topic_number', 'year', 'top_emotions',
                                   'top_emotion_scores']]
    # Group by 'document_id' and find the max value of each emotion
    max_emotion_per_doc = df_merged.groupby('document_id')[emotion_columns].max().reset_index()
    # Add a new column for the average emotion score per document
    max_emotion_per_doc['Average_Emotions_Score'] = max_emotion_per_doc[emotion_columns].mean(axis=1)
    # Extract the year and topic_number columns from the merged data
    df_docs = df_merged[['document_id', 'year', 'Topic_number']].drop_duplicates(subset=['document_id'])
    # Merge with the max_emotion_per_doc dataframe
    df_docs = pd.merge(df_docs, max_emotion_per_doc, on='document_id', how='left')
    df_docs.to_csv('documents.csv', index=False)
    print('aggregate_document_emotions ends')
    return df_docs, emotion_columns

if __name__ == '__main__':
    run_pipeline()