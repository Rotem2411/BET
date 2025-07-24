from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from collections import Counter
import os
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

def clean_text(text, lang="hebrew", return_tokens=False):
    if not isinstance(text, str):
        return None if not return_tokens else []
    if lang == "hebrew":
        STOPWORDS = set(nltk.corpus.stopwords.words('hebrew')).union({'לאחר', 'הייתי', 'מאוד', 'במהלך', 'מכן', 'שנים', 'במשך', 'ולאחר', 'כדי', 'לאחר מכן', 'הייתה'})
        TOKEN_PATTERN = r'^[א-ת0-9]+$'
    else:
        STOPWORDS = set(nltk.corpus.stopwords.words('english'))
        TOKEN_PATTERN = r'^[a-zA-Z0-9]+$'

    tokens = [w for w in nltk.word_tokenize(text)
              if w not in STOPWORDS and re.match(TOKEN_PATTERN, w) and len(w) > 1]
    return tokens if return_tokens else " ".join(tokens)

def preprocess_data(df, columns, lang="hebrew"):
    df = df.dropna(subset=columns, how='all').copy()
    for col in columns:
        df[f"{col}_cleaned"] = df[col].apply(lambda x: clean_text(x, lang=lang))
    return df

def identify_document_topics(documents, sentence_model, cluster_model):
    print('start identify_document_topics')
    try:
        model_path = os.path.join(OUTPUT_DIR, f'{cluster_model}_model')
        topic_model = BERTopic.load(model_path)
        print('Load Bertopic_model successfully')
    except Exception:
        bertopic_args = get_bertopic_init_args(sentence_model)
        if cluster_model=='bertopic_hdbscan':
            print('Setup HDBSCAN model')
            hdbscan_model = HDBSCAN(
                min_cluster_size=5,
                cluster_selection_method="eom",
                prediction_data=True,
                cluster_selection_epsilon=0.2
            )
            print('Initialize BERTopic model')
            topic_model = BERTopic(
                **bertopic_args,
                hdbscan_model=hdbscan_model,
                vectorizer_model= CountVectorizer(max_df = 0.9, min_df = 10 / len(documents)),
                #min_topic_size=5,
                nr_topics=50,
                n_gram_range=(1,2),
                calculate_probabilities=True,
                verbose=True
            )
        if cluster_model == 'bertopic_Agglomerative':
            # Setup ClassTfidfTransformer
            vectorizer_model = CountVectorizer(ngram_range=(1, 1), min_df=3)
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
            agg_cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=4)
            topic_model = BERTopic(
                **bertopic_args,
                vectorizer_model=vectorizer_model,
                hdbscan_model=agg_cluster_model,
                ctfidf_model=ctfidf_model,
                calculate_probabilities=True, verbose=True)

    # Fit the documents and get topics
    topics, probs = topic_model.fit_transform(documents)
    print(f'Fit {cluster_model} with docs')

    # Get topic details
    if cluster_model=='hdbscan':
        probs_df = pd.DataFrame(probs)
        probs_df['Main Topic Score'] = pd.DataFrame({'max': probs_df.max(axis=1)})
        probs_df.rename(columns={probs_df.columns[0]: 'document_id'}, inplace=True)
        probs_df.to_excel('topics_probabilities_each_document.xlsx')

    topics_info = topic_model.get_topic_info()
    topics_info.to_excel(os.path.join(OUTPUT_DIR, "Topics_by_Docs_Info.xlsx"), index=False)
    model_path = os.path.join(OUTPUT_DIR, f'{cluster_model}_model')
    topic_model.save(model_path)
    print('BERTopic saved')
    return topic_model

def evaluate_topic_quality(topic_model, documents):
    topics = topic_model.get_topics()
    topic_words = [[word[0] for word in words] for _, words in topics.items() if words]
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    coherence_model = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    unique_words = set(word for topic in topic_words for word in topic)
    topic_diversity = len(unique_words) / (len(topic_words) * len(topic_words[0])) if topic_words else 0
    return coherence_score, topic_diversity

def evaluate_bertopic_experiments_results(df, cluster_model, lang, param_values):
    results = []
    df_clean = preprocess_data(df, ['text'], lang)
    documents = df_clean['text_cleaned'].tolist()
    for value in param_values:
        if cluster_model == 'bertopic_hdbscan':
            model_file = f"Bertopic_model_{value}"
            try:
                topic_model = BERTopic.load(model_file)
            except Exception as e:
                print(f"Could not load {model_file}: {e}")
                continue
            max_retries = 5
            for attempt in range(max_retries + 1):
                try:
                    coherence, diversity = evaluate_topic_quality(topic_model, documents)
                    break
                except Exception as e:
                    if attempt < max_retries:
                        print(
                            f"Attempt {attempt + 1}: Could not get coherence score for nr_topics: {value}. Error: {e}")
                    else:
                        print(f"Failed all retries: Could not get coherence score for nr_topics: {value}")
                        coherence, diversity = None, None
            if coherence is None or diversity is None:
                continue
            topics = topic_model.topics_
            topics_info = topic_model.get_topic_info()
            num_outliers = topics.count(-1)
            results.append({
                'nr_topics': value,
                'coherence_score': coherence,
                'topic_diversity': diversity,
                'num_outliers': num_outliers
            })
            print(f"HDBSCAN | nr_topics={value} | coherence={coherence:.4f} | diversity={diversity:.4f} | num_outliers={num_outliers}")

        elif cluster_model == 'bertopic_Agglomerative':
            model_file = f"Bertopic_model_{value}"
            try:
                topic_model = BERTopic.load(model_file)
            except Exception as e:
                print(f"Could not load {model_file}: {e}")
                continue
            coherence, diversity = evaluate_topic_quality(topic_model, documents)
            topics_info = topic_model.get_topic_info()
            nr_topics = len(topics_info)
            results.append({
                'distance_threshold': value,
                'coherence_score': coherence,
                'topic_diversity': diversity,
                'nr_topics': nr_topics
            })
            print(f"Agglomerative | distance_threshold={value} | coherence={coherence:.4f} | diversity={diversity:.4f} | nr_topics={nr_topics}")

    metrics_df = pd.DataFrame(results)
    output_name = f"{cluster_model}_evaluation_metrics.csv"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved evaluation metrics to '{output_path}'")
    return metrics_df

def run_bertopic_experiments(df, sentence_model, cluster_model, lang):
    print('start run_bertopic_experiments')
    df_clean = preprocess_data(df, ['text'], lang)
    documents = df_clean['text_cleaned']
    bertopic_args = get_bertopic_init_args(sentence_model)
    if cluster_model == 'bertopic_hdbscan':
        topic_numbers = [10, 25, 50, 75, 100, 125, 150, 200]
        results = []
        for nr in topic_numbers:
            print(f"Training BERTopic with nr_topics={nr}")
            hdbscan_model = HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                cluster_selection_method="eom",
                cluster_selection_epsilon=0.2,
                prediction_data=True
            )
            topic_model = BERTopic(
                **bertopic_args,
                hdbscan_model=hdbscan_model,
                vectorizer_model=CountVectorizer(max_df = 0.9, min_df = 10 / len(documents)),
                min_topic_size=5,
                nr_topics=nr,
                n_gram_range=(1, 2),
                calculate_probabilities=True,
                verbose=True
            )
            topics, probs = topic_model.fit_transform(documents)
            topics_info = topic_model.get_topic_info()
            topics_info_path = os.path.join(OUTPUT_DIR, f"Topics_by_Docs_Info_{cluster_model}_nr_topics{nr}.xlsx")
            topics_info.to_excel(topics_info_path, index=False)
            model_path = os.path.join(OUTPUT_DIR, f"Bertopic_model_{nr}")
            topic_model.save(model_path)

    if cluster_model == 'bertopic_Agglomerative':
        distance_thresholds = [3,4,5,6,7,8,9,10,11,12]
        results = []
        for distance_threshold in distance_thresholds:
            print(f"Training BERTopic with distance_threshold={distance_threshold}")
            # Setup ClassTfidfTransformer
            vectorizer_model = CountVectorizer(ngram_range=(1, 1), min_df=3)
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
            agg_cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
            topic_model = BERTopic(
                **bertopic_args,
                vectorizer_model=vectorizer_model,
                hdbscan_model=agg_cluster_model,
                ctfidf_model=ctfidf_model,
                calculate_probabilities=True, verbose=True
            )
            topics, _ = topic_model.fit_transform(documents)
            topics_info = topic_model.get_topic_info()
            topics_info_path = os.path.join(OUTPUT_DIR, f"Topics_by_Docs_Info_{cluster_model}_distance_threshold{distance_threshold}.xlsx")
            topics_info.to_excel(topics_info_path, index=False)
            model_path = os.path.join(OUTPUT_DIR, f"Bertopic_model_{distance_threshold}")
            topic_model.save(model_path)

    print("run_bertopic_experiments completed and all models/info files saved.")
    return

def run_lda_experiments(df, lang):
    print('start run_lda_experiments')
    df_clean = preprocess_data(df, ['text'], lang)
    tokenized_docs = [doc.split() for doc in df_clean['text_cleaned']]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    num_topics_list = [10, 25, 50, 75, 100, 150, 200]
    results = []
    for num_topics in num_topics_list:
        print(f"Training LDA with num_topics={num_topics}")
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10, alpha='auto', per_word_topics=True)
        perplexity = lda_model.log_perplexity(corpus)
        topic_words = [[w for w, _ in lda_model.show_topic(i, topn=10)] for i in range(num_topics)]
        coherence = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v').get_coherence()
        unique_words = set(word for topic in topic_words for word in topic)
        topic_diversity = len(unique_words) / (len(topic_words) * len(topic_words[0])) if topic_words else 0
        topics = get_main_topics(corpus, lda_model)
        df_topic_info = get_lda_doc_topic_info(topics, lda_model, num_words=10)
        df_topic_info.to_excel(f"Topics_by_Docs_Info_LDA_{num_topics}.xlsx", index=False)
        results.append({'num_topics': num_topics, 'coherence_score': coherence, 'topic_diversity': topic_diversity, 'perplexity': perplexity})
        print(f"LDA with num_topics={num_topics}: coherence_score={coherence:.4f}, topic_diversity={topic_diversity:.4f}")
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("lda_multiple_metrics.csv", index=False)
    return metrics_df

def run_lda_topic_modeling(tokenized_docs, num_topics=10):
    print("start run_lda_topic_modeling")
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=42,
                                passes=10,
                                alpha='auto',
                                per_word_topics=True)
    return lda_model, corpus, dictionary

def analyze_topics_in_data(df, lang, sentence_model, cluster_model):
    if cluster_model.lower() == "lda":
        try:
            lda_csv_path = os.path.join(OUTPUT_DIR, 'Topics_by_Docs_Info_LDA.csv')
            df_topic = pd.read_csv(lda_csv_path)
            print(f'Load {lda_csv_path}')
        except Exception:
            print("start analyze_topics_in_data (LDA)")
            df_docs = preprocess_data(df, ['text'], lang)
            tokenized_docs = [doc.split() for doc in df_docs['text_cleaned']]
            lda_model, corpus, dictionary = run_lda_topic_modeling(tokenized_docs, num_topics=10)
            topics = get_main_topics(corpus, lda_model)
            df_topic = df_docs.drop(columns=['text', 'text_cleaned'])
            df_topic['Topic_number'] = topics
            df_topic.rename(columns={'id': 'document_id'}, inplace=True)
            df_topic.to_csv(lda_csv_path, index=False)
            df_topic_info = get_lda_doc_topic_info(topics, lda_model, num_words=10)
            df_topic_info.to_excel(os.path.join(OUTPUT_DIR, "Topics_by_Docs_Info_LDA.xlsx"), index=False)
            lda_model.save(os.path.join(OUTPUT_DIR, 'lda_model_topics.model'))
            dictionary.save(os.path.join(OUTPUT_DIR, 'lda_dictionary.dict'))
            print("analyze_topics_in_data (LDA) complete")
        return df_topic
    else:
        try:
            bert_csv_path = os.path.join(OUTPUT_DIR, 'Topics_by_Docs_Info_Bertopic.csv')
            df_topic = pd.read_csv(bert_csv_path)
            print(f'Load {bert_csv_path}')
        except Exception:
            print("start analyze_topics_in_data (BERTopic)")
            df_docs = preprocess_data(df, ['text'], lang)
            TopicModel = identify_document_topics(df_docs['text_cleaned'], sentence_model, cluster_model)
            df_topic = df_docs.drop(columns=['text', 'text_cleaned'])
            df_topic['Topic_number'] = TopicModel.topics_
            df_topic.rename(columns={'id': 'document_id'}, inplace=True)
            df_topic.to_csv(bert_csv_path, index=False)
            print("analyze_topics_in_data (BERTopic) complete")
        return df_topic

def get_bertopic_init_args(sentence_model):
    if sentence_model.tokenizer.name_or_path == 'sentence-transformers/all-MiniLM-L6-v2':
        return {"language": "english"}
    elif sentence_model.tokenizer.name_or_path == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        return {"language": "multilingual"}
    else:
        return {"embedding_model": sentence_model}

def get_main_topics(corpus, lda_model):
    return [max(lda_model.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1])[0] for bow in corpus]

def get_lda_doc_topic_info(topics, lda_model, num_words=10):
    topic_counts = Counter(topics)
    topic_data = []
    for topic_num in range(lda_model.num_topics):
        count = topic_counts.get(topic_num, 0)
        keywords = ", ".join([w for w, _ in lda_model.show_topic(topic_num, topn=num_words)])
        topic_data.append({
            "Topic": topic_num,
            "Count": count,
            "topic_keywords": keywords
        })
    df_topic_info = pd.DataFrame(topic_data)
    return df_topic_info