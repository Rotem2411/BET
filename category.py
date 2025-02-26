import plotly.express as px
import pandas as pd
import torch
import numpy as np
import pyarrow, fastparquet
from sklearn.manifold import TSNE
import umap
from hepsylex import Lexicons

def generate_category_embeddings(model, tokenizer, device):
    print("starting category_embeddings")
    category_dict = build_category_dictionary()
    embeddings_categories = []
    index_category = {}
    for cat_index, cat in enumerate(category_dict):
        index_category[cat_index] = cat
        for key, word in enumerate(category_dict[cat]):
            embedding = get_sentence_embedding(word, model, tokenizer, device)
            embeddings_categories += [(cat_index, key, word, embedding)]
    df_category = pd.DataFrame(embeddings_categories,
                                          columns=['category_index', 'word_index', 'word', 'embedding'])

    visualize_embedding_clusters(df_category, method='umap', n_components=3)
    # Save embeddings separately as a Numpy array
    embeddings = np.stack(df_category['embedding'].values)
    np.save('category_embeddings.npy', embeddings)

    # Drop embeddings column and save the rest as Parquet
    df_category.drop('embedding', axis=1, inplace=True)
    df_category.to_parquet('category_metadata.parquet', index=False, compression='snappy')

    print("Saved category metadata as Parquet and embeddings as Numpy array.\n")

    return df_category, embeddings

def build_category_dictionary():
    """
    Constructs a dictionary of categories using Hebrew psychological lexicons.

    Returns:
        dict: Dictionary of categories mapped to word lists.
    """
    lexicons = Lexicons()  # Initialize Hebrew psychological lexicons
    category_dict = {}

    # Dynamically populate category_dict with all EmotionalVariety lexicons
    for attr in dir(lexicons):
        if attr.startswith("EmotionalVariety"):
            category_name = attr.split("_")[-1]  # Use the part after "EmotionalVariety_" as the category name
            category_dict[category_name] = getattr(lexicons, attr)

    return category_dict

def visualize_embedding_clusters(df, method='umap', n_components=3):
    """
    Visualizes embeddings using dimensionality reduction techniques.

    Args:
        df (pd.DataFrame): Dataframe with embeddings.
        method (str, optional): Dimensionality reduction method ('umap' or 'tsne'). Defaults to 'umap'.
        n_components (int, optional): Number of dimensions. Defaults to 3.
    """
    print('start visualize_embedding_clusters')
    # Extract embeddings
    embeddings = list(df['embedding'])

    # Perform dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_reduced = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
        embeddings_reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'.")

    # Create a new DataFrame with reduced dimensions
    df_reduced = pd.DataFrame({
        'category_index': df['category_index'],
        'word_index': df['word_index'],
        'word': df['word'],
        'x': embeddings_reduced[:, 0],
        'y': embeddings_reduced[:, 1],
        'z': embeddings_reduced[:, 2] if n_components == 3 else None
    })

    # Plot the embeddings using Plotly Express
    if n_components == 3:
        fig = px.scatter_3d(df_reduced, x='x', y='y', z='z',
                            color='category_index', hover_data=['word'],
                            title=f'Word Embeddings ({method.upper()} 3D)',
                            width=1200, height=900)
    else:
        fig = px.scatter(df_reduced, x='x', y='y',
                         color='category_index', hover_data=['word'],
                         title=f'Word Embeddings ({method.upper()} 2D)')

    # Save the plot as an HTML file
    output_file = f"word_embeddings_{method}_{n_components}D.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

def get_sentence_embedding(sentence, model, tokenizer, device):
    """
    Generates a BERT embedding for a given sentence using a specified SentenceTransformer model.

    Args:
        sentence (str): The sentence for which the embedding is to be generated.
        model_name (str): The name of the SentenceTransformer model to use. Defaults to 'all-MiniLM-L6-v2'.
        model (SentenceTransformer, optional): Preloaded SentenceTransformer model. If provided, this model will be used instead of loading a new one.

    Returns:
        numpy.ndarray: The BERT embedding for the input sentence.
    """
    # Load the SentenceTransformer model if not provided
    if tokenizer is None:
        # Generate embedding for the sentence
        embedding = model.encode(sentence)
        return embedding
    else:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

