from category import generate_category_embeddings
from Sentence import compute_sentence_similarity, generate_sentence_embeddings
from Topics import analyze_topics_in_data, run_bertopic_experiments, run_lda_experiments, evaluate_bertopic_experiments_results
from figures import generate_visual_insights, visualize_performance_models, visualize_embedding_clusters, plot_metrics
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast

def run_pipeline(topic_model="bertopic_hdbscan", sentence_model=None):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sentence_model == 'paraphrase-multilingual-MiniLM-L12-v2':
            tokenizer = None
            sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            df_original = import_dataset('finaid_applications_2011_2024.xlsx', text_col='story1', id_col='id', year_col='year')
            lang='hebrew'
        elif sentence_model == 'alephbert':
            model_name = "onlplab/alephbert-base"
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            sentence_model = BertModel.from_pretrained(model_name).to(device)
            df_original = import_dataset('finaid_applications_2011_2024.xlsx', text_col='story1', id_col='id', year_col='year')
            lang = 'hebrew'
        elif sentence_model == 'all-MiniLM-L6-v2':
            tokenizer = None
            sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            df_original = import_dataset('df_docs_eng.csv', text_col='sentence', id_col='document_id')
            lang = 'english'
        elif sentence_model == None:
            tokenizer = None
            sentence_model=None
            lang = 'hebrew'
        else:
            print("sentence_model not recognized")
            return

        df_category, category_embedding = generate_category_embeddings(sentence_model, tokenizer, device)
        visualize_embedding_clusters(df_category, category_embedding, method='umap', n_components=3)
        df_sentences, sentences_embedding = generate_sentence_embeddings(df_original, sentence_model, tokenizer, device)
        df_similarity = compute_sentence_similarity(lang, batch_size=100000)

        if topic_model.lower() == "lda":
            metrics_df = run_lda_experiments(df_original, lang)
            plot_metrics(metrics_df,
                         x_col="num_topics",
                         y_cols_labels_colors=[
                             ("coherence_score", "Coherence Score", "blue"),
                             ("topic_diversity", "Topic Diversity", "green")],
                         title="LDA Topic Modeling Metrics")
            df_topic = analyze_topics_in_data(df_original,lang, None, topic_model)
        else:
            run_bertopic_experiments(df_original, sentence_model, topic_model, lang)
            if topic_model == "bertopic_hdbscan":
                param_name = "nr_topics"
                param_values = [10, 25, 50, 75, 100, 125, 150, 200]
            elif topic_model == "bertopic_Agglomerative":
                param_name = "distance_threshold"
                param_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            else:
                param_name = None
                param_values = []

            if param_name:
                metrics_df = evaluate_bertopic_experiments_results(df_original, topic_model, lang, param_values)
            df_topic = analyze_topics_in_data(df_original, lang, sentence_model, topic_model)

        df_merged = integrate_topic_and_similarity_data(df_similarity, df_topic)
        df_docs, emotion_columns = aggregate_document_emotions(df_merged)
        generate_visual_insights(df_docs)

    except Exception as e:
        print(f"Error during main: {e}")

def import_dataset(dataset_name, text_col, id_col=None, year_col=None):
    if dataset_name.endswith('.csv'):
        df = pd.read_csv(dataset_name)
    else:
        df = pd.read_excel(dataset_name)
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in the dataset.")
    df = df.rename(columns={text_col: 'text'})
    if id_col in df.columns:
        df = df.rename(columns={id_col: 'document_id'})
    elif 'document_id' not in df.columns:
        df['document_id'] = range(1, len(df) + 1)
    if year_col in df.columns:
        df = df.rename(columns={year_col: 'year'})
    elif 'year' not in df.columns:
        df['year'] = None
    df_out = df[['document_id', 'year', 'text']].copy()
    print(f"num of documents: {len(df_out)}")
    return df_out

def integrate_topic_and_similarity_data(df_similarity, df_topic):
    print("Start integrate_topic_and_similarity_data")
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
    # input sentence_model: all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, alephbert or None
    sentence_model = "all-MiniLM-L6-v2"
    # input topic_modeling_method: lda, bertopic_hdbscan, bertopic_Agglomerative
    run_pipeline(topic_model="bertopic_hdbscan", sentence_model=sentence_model)
    visualize_performance_models()
