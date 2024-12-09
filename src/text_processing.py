import numpy as np
import pandas as pd
import re
from string import punctuation

import src.models as models

from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, ENGLISH_STOP_WORDS

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


TWO_LETTERS_NUMBERS = r"(?u)\b\w\w+\b"
THREE_LETTERS = r"(?u)\b[a-zA-Z]{3,}\b"

class Lemmatizer:
    def __init__(self, token_pattern=TWO_LETTERS_NUMBERS):
        self.tokenize = re.compile(token_pattern).findall
        self.lemmatizer = WordNetLemmatizer()

    def _wordnet_pos(self, tag):
        """Map POS tag to WordNet POS tag"""
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)

    def lemmatize(self, token, tag):
        return self.lemmatizer.lemmatize(token, self._wordnet_pos(tag))

    def __call__(self, text):
        tokens = self.tokenize(text)
        pos_tags = pos_tag(tokens)
        return [self.lemmatize(token, tag) for token, tag in pos_tags]

class Stemmer:
    def __init__(self, token_pattern=TWO_LETTERS_NUMBERS):
        self.tokenize = re.compile(token_pattern).findall
        self.stem = SnowballStemmer("english").stem # or PorterStemmer

    def __call__(self, text):
        tokens = self.tokenize(text)
        return [self.stem(token) for token in tokens]

def get_stop_words(corpus, tokenizer=None, token_pattern=THREE_LETTERS, max_df=0.9):
    """Get english stop words and frequent words (those with document frequency higher than *max_df*)."""
    if tokenizer is not None:
        tokenizer = tokenizer(token_pattern=token_pattern)

    stop_words = set(ENGLISH_STOP_WORDS)

    doc_freq_vectorizer = CountVectorizer(
        binary=True, lowercase=True, strip_accents="ascii", tokenizer=tokenizer,
        token_pattern=token_pattern if tokenizer is None else None
    )

    doc_freqs = np.array(doc_freq_vectorizer.fit_transform(corpus).todense()).sum(axis=0) / len(corpus)

    words = doc_freq_vectorizer.get_feature_names_out()
    frequent_words = words[doc_freqs >= max_df]
    stop_words.update(frequent_words)

    stop_words = list(stop_words)
    stop_words.extend(["pours"])
    stop_words.extend([
        'argentina', 'thailand', 'turkey', 'united states', 'us', 'trinidad', 'tobago', 'sri lanka',
        'australia', 'aussie', 'australian', 'spain', 'austria', 'belgium', 'brazil', 'brazilian',
        'canada', 'china', 'czech', 'denmark', 'england', 'finland', 'france', 'germany', 'greece',
        'india', 'ireland', 'italy', 'jamaica', 'japan', 'kenya', 'mexico', 'netherlands', 'norway',
        'poland', 'russia', 'scotland', 'singapore', 'chinese', 'french', 'irish', 'italian',
        'jamaican', 'japanese', 'africa', 'african', 'mexican', 'alaska', 'alaskan'
    ])
    #stop_words.extend(["aaaagghh", "aaagh", "aaah", "meh"])

    if tokenizer is not None:
        stop_words = tokenizer(" ".join(stop_words))

    return stop_words

def get_word_counts(corpus, beer_names, tokenizer=None, token_pattern=THREE_LETTERS, **vectorizer_kwargs):
    """Get the word counts in the corpus. When using a tokenizer, this takes some time.

    This is separate from computing the tf-idf scores because we may want to do something simple
    like subtracting frequencies when looking for negative words."""
    if tokenizer is not None:
        tokenizer = tokenizer(token_pattern=token_pattern)

    vectorizer = CountVectorizer(
        lowercase=True, strip_accents="ascii", tokenizer=tokenizer,
        token_pattern=token_pattern if tokenizer is None else None,
        **vectorizer_kwargs
    )

    counts = vectorizer.fit_transform(corpus)

    vocabulary = vectorizer_kwargs.get("vocabulary")
    if vocabulary is None:
        vocabulary = vectorizer.get_feature_names_out()

    counts = pd.DataFrame(counts.toarray(), columns=vocabulary, index=beer_names)
    counts.index.name = "beer_name"
    return counts

def get_tfidf_scores(counts):
    """Get the tf-idf scores from the raw word counts."""
    tfidf = TfidfTransformer().fit_transform(counts)

    tfidf = pd.DataFrame(tfidf.toarray(), columns=counts.columns, index=counts.index)
    tfidf.index.name = "beer_name"

    return tfidf

def get_top_attributes(scores, top_attributes=10, column_prefix="attr_"):
    attributes = {}
    for idx, row in scores.iterrows():
        top_attr = row[1:].sort_values(ascending=False).head(top_attributes)
        attributes[idx] = list(top_attr.index)

    attributes = pd.DataFrame.from_dict(
        attributes, orient="index", columns=[f"{column_prefix}{i+1}" for i in range(top_attributes)])
    attributes.index.name = "beer_name"

    return attributes

def split_worst_and_best_reviews(reviews, percent=10):
    percentiles = reviews.groupby("beer_name")["rating"].quantile(percent / 100).rename("percentile")
    reviews = reviews.merge(percentiles, on="beer_name")
    worst_reviews = reviews[reviews["rating"] <= reviews["percentile"]]
    best_reviews = reviews[reviews["rating"] > reviews["percentile"]]
    return worst_reviews, best_reviews


def get_complaints_by_topic(topic_by_beer, tokens_feeling, topic_id):
    tokens_feeling = tokens_feeling[tokens_feeling["max_feel"] == "sadness"]
    topic_by_beer['dominant_topic'] = topic_by_beer['topics'].apply(lambda x: max(x, key=lambda e : e[1])[0])
    topic_by_beer = topic_by_beer[topic_by_beer["dominant_topic"] == topic_id]
    tokens_feeling = tokens_feeling[tokens_feeling["beer_name"].isin(topic_by_beer["beer_name"])]
    tfidf_df = models.summarize(tokens_feeling.iloc[:5000])

    return tfidf_df

def get_complaints_by_beer_name(data, beer_name):
    data = data[data["beer_name"] == beer_name]
    data = data[data["max_feel"] == "sadness"]
    complaints = models.summarize(data)

    return complaints

def lda_by_beers(reviews, nb_topics):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        tokens = text.lower().split(" ")
        tokens = [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]
        return tokens

    grouped_data = reviews.groupby('beer_name')['review'].apply(' '.join).reset_index()
    grouped_data['processed'] = grouped_data['review'].apply(preprocess)
    dictionary = corpora.Dictionary(grouped_data['processed'])
    dictionary.filter_extremes(no_above=0.4)
    corpus = [dictionary.doc2bow(text) for text in grouped_data['processed']]

    num_topics = nb_topics
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)

    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=5):
        print(f"Topic {idx}: {topic}")

    grouped_data['topics'] = [lda_model.get_document_topics(doc) for doc in corpus]

    # lda_model.save("lda_by_beer/LDA_model.pkl")
    # pd.to_pickle(grouped_data, "lda_by_beer/results_lda.pkl")

    return grouped_data

def split_review(review):
    return re.split('\\;|\\,|\\.', review) # tokenize on sentences (by . , or ;)