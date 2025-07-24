import pandas as pd
import torch
import numpy as np
from hepsylex import Lexicons
import os
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

def generate_category_embeddings(model, tokenizer=None, device='cpu'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings_path = os.path.join(OUTPUT_DIR, 'category_embeddings.npy')
    metadata_path = os.path.join(OUTPUT_DIR, 'category_metadata.parquet')
    try:
        embeddings = np.load(embeddings_path)
        df_category = pd.read_parquet(metadata_path)
        print("Load category metadata and embeddings")
    except Exception:
        print("starting category_embeddings")
        # Determine language based on the model name when possible
        try:
            if model and hasattr(model, 'tokenizer') and getattr(model.tokenizer, 'name_or_path', '') == 'sentence-transformers/all-MiniLM-L6-v2':
                lang = 'english'
            else:
                lang = 'hebrew'
        except Exception:
            lang = 'hebrew'
        category_dict = build_category_dictionary(lang)
        embeddings_categories = []
        index_category = {}
        for cat_index, cat in enumerate(category_dict):
            index_category[cat_index] = cat
            for key, word in enumerate(category_dict[cat]):
                embedding = get_sentence_embedding(word, model, tokenizer, device)
                embeddings_categories += [(cat_index, key, word, embedding)]
        df_category = pd.DataFrame(embeddings_categories,
                                   columns=['category_index', 'word_index', 'word', 'embedding'])
        embeddings = np.stack(df_category['embedding'].values)
        # Save to the output directory
        np.save(embeddings_path, embeddings)
        df_category.drop('embedding', axis=1, inplace=True)
        df_category.to_parquet(metadata_path, index=False, compression='snappy')
        print("Saved category metadata as Parquet and embeddings as Numpy array")
    return df_category, embeddings

def build_category_dictionary(lang):
    print("Starting build_category_dictionary")
    if lang == 'hebrew':
        lexicons = Lexicons()  # Initialize Hebrew psychological lexicons
        category_dict = {}
        # Dynamically populate category_dict with all EmotionalVariety lexicons
        for attr in dir(lexicons):
            if attr.startswith("EmotionalVariety"):
                category_name = attr.split("_")[-1]  # Use the part after "EmotionalVariety_" as the category name
                category_dict[category_name] = getattr(lexicons, attr)
        return category_dict
    else:
        category_dict = {}
        category_dict['Drives'] = ['affiliation', 'achieve', 'power']
        category_dict['States'] = ['need', 'want', 'acquire', 'lack', 'fulfil', 'fatigue']
        category_dict['Motive'] = ['reward', 'risk', 'curiosity', 'allure']
        category_dict['Time_Orientation'] = ['focus_past', 'focus_present', 'focus_future']
        category_dict['Culture'] = ['politic', 'ethnicity', 'tech']
        category_dict['Lifestyle'] = ['leisure', 'home', 'work', 'money', 'religious']
        category_dict['Physical'] = ['health', 'illness', 'wellness', 'mental', 'substances', 'sexual', 'food', 'death']
        category_dict['Social'] = ['prosocial', 'polite', 'conflict', 'moral', 'communication']
        category_dict['Emotions_pos'] = ['Amused', 'Anticipation', 'Calm', 'Contentment', 'Enthusiastic', 'Interested',
                                         'Joy', 'Proud', 'Surprise', 'Trust', 'Vigor']
        category_dict['Emotions_neg'] = ['Anger', 'Anxiety', 'Ashamed', 'Confusion', 'Disgust', 'Fatigue', 'Guilt',
                                         'Hostile', 'Nervous', 'Sad', 'Crying', 'Sarcasm', 'Smirk', 'Swear']
        return category_dict

def get_sentence_embedding(sentence, model, tokenizer, device):
    if tokenizer is None:
        embedding = model.encode(sentence)
        return embedding
    else:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()