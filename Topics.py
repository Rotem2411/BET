from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
#nltk.download('stopwords')
#nltk.download('punkt_tab')
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def clean(text):
    """
    Cleans the given text by removing stopwords, non-alphabetic characters, and short words.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
      return None  # Return None if text is not a string
    stopword = nltk.corpus.stopwords.words('hebrew')
    additional_stopwords = {'לאחר', 'הייתי', 'מאוד', 'במהלך', 'מכן', 'שנים', 'במשך', 'ולאחר', 'כדי', 'לאחר מכן', 'הייתה'}  # Additional stopwords to remove
    stopword.extend(additional_stopwords)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopword]
    tokens = [word for word in tokens if re.match(r'^[א-ת0-9]+$', word)]
    tokens = [word for word in tokens if len(word) > 1]
    return " ".join(tokens)

def preprocess_data(original_df, required_columns):
    """
    Prepares the data by cleaning specified text columns and dropping empty rows.

    Args:
        original_df (pd.DataFrame): The original dataset.
        required_columns (list): List of column names to be processed.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    print('start preprocess_data for topics')
    # Drop rows where all specified columns are empty
    cleaned_df = original_df.dropna(subset=required_columns, how='all').copy()

    # Apply the clean function to each column and add as a new column
    for column in required_columns:
        new_column_name = f"{column}_cleaned"
        cleaned_df[new_column_name] = cleaned_df[column].apply(clean)
    return cleaned_df

def identify_document_topics(documents):
    """
    Extracts topics from a list of documents using BERTopic and HDBSCAN.

    Args:
        documents (list of str): The input documents.

    Returns:
        BERTopic: The trained BERTopic model.
    """
    print('start identify_document_topics')
    try:
        topic_model = BERTopic.load('Bertopic_model')
        print('Load Bertopic_model successfully')
    except:

        # Dynamically adjust min_df based on dataset size
        num_docs = len(documents)
        max_df = 0.9
        min_df = 5 / num_docs
        print(f"Dynamically adjusted min_df: {min_df}, max_df:{max_df}")
        # Initialize BERTopic with a custom vectorizer
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)

        # HDBSCAN clustering model setup
        print('Setup HDBSCAN model')
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        # Initialize the BERTopic model
        print('Initialize BERTopic model')
        topic_model = BERTopic(
            language="multilingual",
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            min_topic_size=10,
            nr_topics=100,
            n_gram_range=(1, 2),
            calculate_probabilities=True,
            verbose=True
        )

        # Fit the documents and get topics
        topic_model.save('Bertopic_model')
        print('BERTopic saved')
    topics, probs = topic_model.fit_transform(documents)
    print('Fit BERTopic with docs')

    # updated topics and probabilities
    topics = topic_model.topics_
    probs = topic_model.probabilities_

    # Get topic details
    topics_info = topic_model.get_topic_info()
    probs_df = pd.DataFrame(probs)
    probs_df['Main Topic Score'] = pd.DataFrame({'max': probs_df.max(axis=1)})
    probs_df.rename(columns={probs_df.columns[0]: 'document_id'}, inplace=True)
    probs_df.to_excel('topics_probabilities_each_document.xlsx')
    topics_info.to_excel(f"Topics_by_Docs_Info.xlsx", index=False)
    return topic_model

def display_topic_analysis(topic_model, df):
    """
    Generates visualizations for topic analysis including bar charts, intertopic distances, and hierarchies.

    Args:
        topic_model (BERTopic): The trained topic model.
        df (pd.DataFrame): Dataframe containing topic-related data.
    """
    print('start display_topic_analysis')
    # Visualize topic distributions
    topic_model.visualize_barchart(top_n_topics=50, n_words=15, title=f"Topic Word Scores").write_html(f"Topics_by_Docs_Barchart.html")
    topic_model.visualize_topics(title=f"Intertopic Distance Map").write_html(f"Topics_by_Docs_Intertopics_Distance_Map.html")
    topic_model.visualize_hierarchy(title=f"Hierarchical Clustering").write_html(f"Topics_by_Docs_Hierarchical_Clustering.html")
    topic_model.visualize_heatmap(n_clusters=20).write_html('Topics_Similarity_Matrix.html')
    topics_over_time = topic_model.topics_over_time(df['story1_cleaned'].tolist(), df['year'],)
    topic_model.visualize_topics_over_time(topics_over_time).write_html('Topics_Over_Time.html')


def analyze_topics_in_data(df):
    """
    Processes a dataset to extract topics and visualize them.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset with topic numbers assigned.
    """
    print("start analyze_topics_in_data")
    df = df.drop(columns=['sig1', 'know1', 'asp1'])
    df_docs = preprocess_data(df, ['story1'])
    TopicModel = identify_document_topics(df_docs['story1_cleaned'])
    df['Topic_number'] = TopicModel.transform(df_docs['story1_cleaned'].tolist())[0]
    df = df.drop(columns='story1')
    df.rename(columns={'id': 'document_id'}, inplace=True)
    df.to_csv('finaid_applications_Topics.csv', index=False)
    display_topic_analysis(TopicModel, df)
    print("analyze_topics_in_data complete")
    return df
