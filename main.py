from category import generate_category_embeddings
from Sentence import compute_sentence_similarity, generate_sentence_embeddings
from Topics import analyze_topics_in_data, run_bertopic_experiments, run_lda_experiments, evaluate_bertopic_experiments_results
from figures import generate_visual_insights, visualize_performance_models, visualize_embedding_clusters, plot_metrics
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast

def run_pipeline(
    topic_model: str = "bertopic_hdbscan",
    sentence_model: str | None = None,
    dataset_name: str | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
    year_col: str | None = None,
    lang: str | None = None
) -> None:
    """
    Execute the full BET pipeline.

    Parameters
    ----------
    topic_model : str
        Topic modeling method to use. Options include "lda", "bertopic_hdbscan", or
        "bertopic_Agglomerative". Defaults to "bertopic_hdbscan".
    sentence_model : str or None
        The identifier for the sentence embedding model to use. If None, a default
        Hebrew model is assumed. Recognised values are:
        - "paraphrase-multilingual-MiniLM-L12-v2"
        - "alephbert"
        - "all-MiniLM-L6-v2"
        - None (Hebrew default)
    dataset_name : str or None
        Path to a custom dataset to process. When provided, this overrides the
        default dataset selection based on the sentence_model. Should be either
        a CSV or Excel file. If omitted, the built‑in defaults are used.
    text_col : str or None
        Name of the column containing the raw text in the dataset. Only required
        when specifying a custom dataset via ``dataset_name``. Ignored when using
        the built‑in datasets.
    id_col : str or None
        Name of the column containing a unique document identifier in the dataset.
        Optional; if not provided, a sequential ID is assigned.
    year_col : str or None
        Name of the column containing a year associated with each document.
        Optional; defaults to None if not present.
    lang : str or None
        Explicitly specify the language of the dataset ("hebrew" or "english").
        This parameter is mainly used with custom datasets. If omitted, the
        language is inferred from the sentence_model selection.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = None
        # Determine which model and dataset to use. If a custom dataset is provided,
        # the language must either be specified explicitly or inferred from the
        # sentence_model name.
        if dataset_name is not None:
            # Load a user‑provided dataset. Use specified column names or fall back
            # to sensible defaults. The language must be provided or inferred.
            if lang is None:
                # Infer language based on sentence_model when possible.
                if sentence_model in {"all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"}:
                    lang = "english"
                elif sentence_model in {"alephbert", None}:
                    lang = "hebrew"
                else:
                    # If model name is unrecognised, default to English.
                    lang = "english"
            # When using a custom dataset, the user should specify the text column.
            # If not provided, assume the column is called "text".
            df_original = import_dataset(
                dataset_name,
                text_col=text_col or "text",
                id_col=id_col,
                year_col=year_col,
            )
            # Load the appropriate sentence model based on user selection or language.
            if sentence_model == "paraphrase-multilingual-MiniLM-L12-v2":
                tokenizer = None
                sentence_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            elif sentence_model == "alephbert":
                model_name = "onlplab/alephbert-base"
                tokenizer = BertTokenizerFast.from_pretrained(model_name)
                sentence_model = BertModel.from_pretrained(model_name).to(device)
            elif sentence_model == "all-MiniLM-L6-v2":
                tokenizer = None
                sentence_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            elif sentence_model is None:
                # Default to a None model; this triggers the Hebrew pipeline with
                # AlephBERT inside generate_sentence_embeddings.
                tokenizer = None
                sentence_model = None
            else:
                print("sentence_model not recognised")
                return
        else:
            # Use built‑in dataset selection logic based on the sentence_model.
            if sentence_model == "paraphrase-multilingual-MiniLM-L12-v2":
                tokenizer = None
                sentence_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                df_original = import_dataset(
                    "finaid_applications_2011_2024.xlsx",
                    text_col="story1",
                    id_col="id",
                    year_col="year",
                )
                lang = "hebrew"
            elif sentence_model == "alephbert":
                model_name = "onlplab/alephbert-base"
                tokenizer = BertTokenizerFast.from_pretrained(model_name)
                sentence_model = BertModel.from_pretrained(model_name).to(device)
                df_original = import_dataset(
                    "finaid_applications_2011_2024.xlsx",
                    text_col="story1",
                    id_col="id",
                    year_col="year",
                )
                lang = "hebrew"
            elif sentence_model == "all-MiniLM-L6-v2":
                tokenizer = None
                sentence_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                df_original = import_dataset(
                    "df_docs_eng.csv",
                    text_col="sentence",
                    id_col="document_id",
                )
                lang = "english"
            elif sentence_model is None:
                tokenizer = None
                sentence_model = None
                df_original = import_dataset(
                    "finaid_applications_2011_2024.xlsx",
                    text_col="story1",
                    id_col="id",
                    year_col="year",
                )
                lang = "hebrew"
            else:
                print("sentence_model not recognised")
                return

        # Generate embeddings for categories and sentences
        df_category, category_embedding = generate_category_embeddings(
            sentence_model, tokenizer, device
        )
        # Visualise embedding clusters for debugging/analysis
        visualize_embedding_clusters(
            df_category, category_embedding, method="umap", n_components=3
        )
        df_sentences, sentences_embedding = generate_sentence_embeddings(
            df_original, sentence_model, tokenizer, device
        )
        df_similarity = compute_sentence_similarity(lang, batch_size=100000)

        # Topic modelling and evaluation
        if topic_model.lower() == "lda":
            metrics_df = run_lda_experiments(df_original, lang)
            plot_metrics(
                metrics_df,
                x_col="num_topics",
                y_cols_labels_colors=[
                    ("coherence_score", "Coherence Score", "blue"),
                    ("topic_diversity", "Topic Diversity", "green"),
                ],
                title="LDA Topic Modeling Metrics",
            )
            df_topic = analyze_topics_in_data(
                df_original, lang, None, topic_model
            )
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
                metrics_df = evaluate_bertopic_experiments_results(
                    df_original, topic_model, lang, param_values
                )
            df_topic = analyze_topics_in_data(
                df_original, lang, sentence_model, topic_model
            )

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
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the BET pipeline on a Hebrew or English corpus. By default "
            "the pipeline uses the Hebrew financial aid applications dataset or "
            "a provided English dataset when a sentence model is set to a "
            "multilingual/English transformer."
        )
    )
    parser.add_argument(
        "--sentence_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help=(
            "Sentence embedding model to use. Choose from: "
            "'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2', 'alephbert', or leave empty for the Hebrew default."
        ),
    )
    parser.add_argument(
        "--topic_model",
        type=str,
        default="bertopic_hdbscan",
        help=(
            "Topic modeling algorithm to use. Options include 'lda', "
            "'bertopic_hdbscan', or 'bertopic_Agglomerative'."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to a custom dataset file. If provided, this overrides the "
            "default dataset selection. The file can be CSV or Excel."
        ),
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default=None,
        help=(
            "Name of the column containing raw text in the provided dataset. "
            "If omitted, defaults to 'text'."
        ),
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default=None,
        help=(
            "Name of the column containing a unique document identifier in the "
            "provided dataset. Optional; if omitted a sequential ID will be assigned."
        ),
    )
    parser.add_argument(
        "--year_col",
        type=str,
        default=None,
        help=(
            "Name of the column containing a year associated with each document. "
            "Optional; if omitted the year will be set to None."
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=(
            "Explicitly specify the dataset language ('hebrew' or 'english'). "
            "This parameter is only required when passing a custom dataset and "
            "using a sentence model that does not imply the language."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        topic_model=args.topic_model,
        sentence_model=args.sentence_model,
        dataset_name=args.dataset,
        text_col=args.text_col,
        id_col=args.id_col,
        year_col=args.year_col,
        lang=args.lang,
    )

    # If running BERTopic experiments, visualise performance metrics if available.
    if args.topic_model.lower().startswith("bertopic"):
        try:
            visualize_performance_models()
        except Exception:
            # The visualisation depends on files generated during experiments; ignore if unavailable.
            pass
