from hepsylex import Lexicons
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertModel, BertTokenizerFast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_embedding(sentence, model, tokenizer=None, device="cpu"):
    """Generates a BERT embedding for a given sentence."""
    if tokenizer is None:
        return model.encode(sentence)
    else:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def build_category_dictionary_eng():
    """Constructs an English category dictionary."""
    return {
        'Drives': ['affiliation', 'achieve', 'power'],
        'States': ['need', 'want', 'acquire', 'lack', 'fulfil', 'fatigue'],
        'Motive': ['reward', 'risk', 'curiosity', 'allure'],
        'Time_Orientation': ['focus_past', 'focus_present', 'focus_future'],
        'Culture': ['politic', 'ethnicity', 'tech'],
        'Lifestyle': ['leisure', 'home', 'work', 'money', 'religious'],
        'Physical': ['health', 'illness', 'wellness', 'mental', 'substances', 'sexual', 'food', 'death'],
        'Social': ['prosocial', 'polite', 'conflict', 'moral', 'communication'],
        'Emotions_pos': ['Amused', 'Anticipation', 'Calm', 'Contentment', 'Enthusiastic', 'Interested', 'Joy', 'Proud', 'Surprise', 'Trust', 'Vigor'],
        'Emotions_neg': ['Anger', 'Anxiety', 'Ashamed', 'Confusion', 'Disgust', 'Fatigue', 'Guilt', 'Hostile', 'Nervous', 'Sad', 'Crying', 'Sarcasm', 'Smirk', 'Swear']
    }

def build_category_dictionary_heb():
    """Constructs a dictionary of Hebrew psychological lexicons."""
    lexicons = Lexicons()
    return {attr.split("_")[-1]: getattr(lexicons, attr) for attr in dir(lexicons) if attr.startswith("EmotionalVariety")}

def generate_category_embeddings(model, category_dict, tokenizer=None, device="cpu"):
    """Generates category embeddings."""
    embeddings_data = []
    for cat_index, (cat, words) in enumerate(category_dict.items()):
        for key, word in enumerate(words):
            embedding = get_sentence_embedding(word, model, tokenizer, device)
            embeddings_data.append((cat_index, key, word, embedding))
    return pd.DataFrame(embeddings_data, columns=['category_index', 'word_index', 'word', 'embedding'])

def compute_similarity(sentences, sentences_embedding, category_embeddings, category_dict):
    """
    Compute the maximum similarity between each sentence and each category.

    Args:
        sentences (list): List of sentences.
        sentences_embedding (list): Corresponding sentence embeddings.
        category_embeddings (DataFrame): Category embeddings with columns ['category_index', 'word_index', 'word', 'embedding'].
        category_dict (dict): Dictionary mapping category indices to category names.

    Returns:
        DataFrame: Sentence-category similarity DataFrame with max similarity per category.
    """
    similarity_results = []

    for i, sentence in enumerate(sentences):
        sentence_embedding = sentences_embedding[i].reshape(1, -1)  # Reshape for cosine similarity
        category_max_similarities = {}  # Store max similarity per category

        for _, row in category_embeddings.iterrows():
            category_name = category_dict[row["category_index"]]
            similarity = cosine_similarity(sentence_embedding, row["embedding"].reshape(1, -1))[0, 0]  # Compute similarity
            if category_name in ['Emotions_pos', 'Emotions_neg']:
                category_name = row["word"]

            # Store the max similarity per category
            if category_name not in category_max_similarities or similarity > category_max_similarities[category_name]:
                category_max_similarities[category_name] = similarity

        # Add max similarity for each category
        for category, max_similarity in category_max_similarities.items():
            similarity_results.append((sentence, category, max_similarity))

    return pd.DataFrame(similarity_results, columns=["Sentence", "Category", "Max_Similarity"])

def run():
    """Executes the model comparison pipeline."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Sentence Transformer models
    model_heb = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model_eng = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load AlephBERT model and tokenizer
    alephbert_model = BertModel.from_pretrained("onlplab/alephbert-base", output_hidden_states=True).to(device)
    alephbert_tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")

    category_dicts = {"eng": build_category_dictionary_eng(), "heb": build_category_dictionary_heb()}
    category_embeddings = {
        "eng": generate_category_embeddings(model_eng, category_dicts["eng"]),
        "heb": generate_category_embeddings(model_heb, category_dicts["heb"]),
        "alephbert": generate_category_embeddings(alephbert_model, category_dicts["heb"], alephbert_tokenizer, device)
    }

    sentences = {
        "eng": ["I love my family",
                   "I love chocolate cake",
                   "The dog runs fast",
                   "She speaks three languages fluently",
                   "They are playing soccer outside",
                   "He studies hard every night to improve his math skills",
                   "My best friend and I love watching movies every weekend",
                   "The teacher gave us homework that is due next Monday",
                   "I like drinking hot tea while reading a good book"],
        "heb": ["אני אוהב את משפחתי",
                     "אני אוהבת עוגת שוקולד",
                     "הכלב רץ ממש מהר",
                     "היא מדברת שלוש שפות שוטף",
                     "הם משחקים כדורגל בחוץ עכשיו",
                     "הוא לומד קשה כל לילה כדי לשפר את יכולותיו במתמטיקה",
                     "החבר הכי טוב שלי ואני אוהבים לראות סרטים בסופי שבוע",
                     "המורה נתן לנו שיעורי בית שצריך להגיש ביום שני הקרוב",
                     "אני אוהב לשתות תה חם תוך כדי קריאת ספר טוב"]
    }

    sentence_embeddings = {
        "eng": [get_sentence_embedding(sent, model_eng) for sent in sentences["eng"]],
        "heb": [get_sentence_embedding(sent, model_heb) for sent in sentences["heb"]],
        "alephbert": [get_sentence_embedding(sent, alephbert_model, alephbert_tokenizer, device) for sent in sentences["heb"]]
    }
    # Ensure alephbert key exists before using it
    if "alephbert" in category_embeddings:
        similarity_data = {lang: compute_similarity(sentences[lang], sentence_embeddings[lang], category_embeddings[lang], {idx: cat for idx, cat in enumerate(category_dicts["heb"].keys())}) for lang in ["heb", "alephbert"]}
    else:
        similarity_data = {lang: compute_similarity(sentences[lang], sentence_embeddings[lang], category_embeddings[lang], {idx: cat for idx, cat in enumerate(category_dicts[lang].keys())}) for lang in ["eng", "heb"]}


    """
    similarity_data = {lang: compute_similarity(sentences[lang], sentence_embeddings[lang], category_embeddings[lang], {idx: cat for idx, cat in enumerate(category_dicts[lang].keys())}) for lang in ["eng", "heb"]}
    df_compare = pd.DataFrame({"sentences_english": sentences["eng"], "sentences_hebrew": sentences["heb"]})
    df_compare["num_words_eng"] = df_compare["sentences_english"].apply(lambda x: len(x.split()))
    df_compare["num_words_heb"] = df_compare["sentences_hebrew"].apply(lambda x: len(x.split()))

    df_pivot = {lang: sim.pivot(index="Sentence", columns="Category", values="Max_Similarity").reset_index().rename(columns={"Sentence": f"sentences_{lang}"}) for lang, sim in similarity_data.items()}
    df_pivot["eng"].columns = ['sentences_english'] + [f"{col}_eng" for col in df_pivot["eng"].columns[1:]]
    df_pivot["heb"].columns = ['sentences_hebrew'] + [f"{col}_heb" for col in df_pivot["heb"].columns[1:]]

    df_compare = df_compare.merge(df_pivot["eng"], on="sentences_english", how="left").merge(df_pivot["heb"], on="sentences_hebrew", how="left")

    common_categories = set(col.replace("_eng", "") for col in df_pivot["eng"].columns if "_eng" in col) & set(col.replace("_heb", "") for col in df_pivot["heb"].columns if "_heb" in col)
    for category in common_categories:
        df_compare[f"{category}_diff"] = df_compare[f"{category}_eng"] - df_compare[f"{category}_heb"]

    df_compare.to_csv("sentence_category_comparison.csv", index=False)
    """

    df_compare = pd.DataFrame({"sentences_hebrew": sentences["heb"]})
    df_compare["num_words_heb"] = df_compare["sentences_hebrew"].apply(lambda x: len(x.split()))

    df_pivot = {lang: sim.pivot(index="Sentence", columns="Category", values="Max_Similarity").reset_index()
                for lang, sim in similarity_data.items()}
    df_pivot["heb"].columns = ['sentences_hebrew'] + [f"{col}_heb" for col in df_pivot["heb"].columns[1:]]
    df_pivot["alephbert"].columns = ['sentences_hebrew'] + [f"{col}_alephbert" for col in df_pivot["alephbert"].columns[1:]]

    df_compare = df_compare.merge(df_pivot["heb"], on="sentences_hebrew", how="left")
    df_compare = df_compare.merge(df_pivot["alephbert"], on="sentences_hebrew", how="left")

    common_categories = set(col.replace("_heb", "") for col in df_pivot["heb"].columns if "_heb" in col) & set(col.replace("_alephbert", "") for col in df_pivot["alephbert"].columns if "_alephbert" in col)
    for category in common_categories:
        df_compare[f"{category}_diff"] = df_compare[f"{category}_alephbert"] - df_compare[f"{category}_heb"]

    df_compare.to_csv("sentence_category_comparison_with_alephbert.csv", index=False)