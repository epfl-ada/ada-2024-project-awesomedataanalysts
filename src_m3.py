import numpy as np
import pandas as pd

import re

def process_document(document):
    # some apostrophes have been replaced by â\x80\x99 (â) this is due to an encoding issue beyond our control
    document = re.sub(r"â\x80\x99", "’", document)

    # the word ipa is often missspelled (eg ipaa) and the lemmatizer doesn't remove the s from the plural ipas
    document = re.sub(r"\b\w*ipa\w*\b", "ipa", document, flags=re.IGNORECASE)

    return document

def build_corpus_from_reviews(reviews, expert_ids=None, expert_weight=2):
    
    reviews_by_beer = reviews.groupby("beer_name")
    #beer_names = reviews_by_beer.groups.keys()
    beer_names = list(reviews_by_beer.size().sample(len(reviews_by_beer), random_state=23).index)

    corpus = []
    for beer_name in beer_names:
        beer_reviews = reviews_by_beer.get_group(beer_name)

        if expert_ids is not None:
            expert_reviews = beer_reviews[beer_reviews["user_id"].isin(expert_ids)]
            beer_reviews = pd.concat([beer_reviews] + (expert_weight - 1) * [expert_reviews], ignore_index=True)

        document = "\n".join(beer_reviews["text"])

        document = process_document(document)

        corpus.append(document)

    return corpus, beer_names

def get_quantile_split(reviews, q=10, column="rating"):
    """Split reviews into positive and negative, at the qth quantile of the given column."""
    percentiles = reviews.groupby("beer_name")[column].quantile(q / 100).rename("percentile")
    reviews = reviews.merge(percentiles, on="beer_name")
    worst_reviews = reviews[reviews[column] <= reviews["percentile"]]
    best_reviews = reviews[reviews[column] > reviews["percentile"]]
    return best_reviews, worst_reviews

def build_split_corpus_from_emotion_analysis(reviews, expert_ids=None, expert_weight=2):
    beer_names = reviews["beer_name"].unique()

    # TODO what about experts and process_document ? emotion analysis model has its own pipeline, can't do this
    # TODO make pickle or (better) upload csv. tokens_feeling = pd.read_pickle("./data/review_with_tokens_emotions.pkl")
    tokens_feeling = pd.read_csv("./data/reviews_with_tokens_emotions.csv")

    neg_corpus = []
    pos_corpus = []
    for beer_name in beer_names:
        #if reviews[reviews['beer_name'] == beer_name]['rating'] >=4:
        beer_token = tokens_feeling[tokens_feeling["beer_name"] == beer_name]
        beer_nagative_tokens_sadness = beer_token[beer_token["max_feel"] == "sadness"]
        beer_nagative_tokens_disgust = beer_token[beer_token["max_feel"] == "disgust"]
        sadness_text = " ".join(beer_nagative_tokens_sadness["review"].apply(str).tolist())
        disgust_text = " ".join(beer_nagative_tokens_disgust["review"].apply(str).tolist())
        negative_text = disgust_text + sadness_text
        beer_positive_tokens_nuetral = beer_token[beer_token["max_feel"] == "neutral"]
        beer_positive_tokens_joy = beer_token[beer_token["max_feel"] == "joy"]
        neutral_text = " ".join(beer_positive_tokens_nuetral["review"].apply(str).tolist())
        joy_text = " ".join(beer_positive_tokens_joy["review"].apply(str).tolist())
        positive_text = neutral_text + joy_text
        neg_corpus.append(negative_text)
        pos_corpus.append(positive_text)

    return pos_corpus, neg_corpus, beer_names

def build_corpus(reviews, expert_ids=None, expert_weight=2, quantile_split=False, emotion_split=False):
    if quantile_split and emotion_split:
        raise ValueError("choose either quantile split or emotion split, not both")

    if emotion_split and expert_ids is not None:
        raise ValueError("experts can only be used in the quantile split case")

    if quantile_split:
        pos_reviews, neg_reviews = get_quantile_split(reviews, q=quantile_split)
        pos_corpus, beer_names = build_corpus_from_reviews(pos_reviews, expert_ids=expert_ids, expert_weight=expert_weight)
        neg_corpus, beer_names_neg = build_corpus_from_reviews(neg_reviews, expert_ids=expert_ids, expert_weight=expert_weight)
        assert beer_names == beer_names_neg
        return pos_corpus, neg_corpus, beer_names

    elif emotion_split:
        pos_corpus, neg_corpus, beer_names = build_split_corpus_from_emotion_analysis(
            reviews, expert_ids=expert_ids, expert_weight=expert_weight)
        return pos_corpus, neg_corpus, beer_names

    else:
        corpus, reviews = build_corpus_from_reviews(reviews, expert_ids=expert_ids, expert_weight=expert_weight)
        return corpus, reviews

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, ENGLISH_STOP_WORDS

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
    stop_words.extend(["pours", "pour", "poured"])
    stop_words.extend([
        'argentina', 'thailand', 'turkey', 'united states', 'us', 'trinidad', 'tobago', 'sri lanka',
        'australia', 'aussie', 'australian', 'spain', 'austria', 'belgium', 'brazil', 'brazilian',
        'canada', 'china', 'czech', 'denmark', 'england', 'finland', 'france', 'germany', 'greece',
        'india', 'ireland', 'italy', 'jamaica', 'japan', 'kenya', 'mexico', 'netherlands', 'norway',
        'poland', 'russia', 'scotland', 'singapore', 'chinese', 'french', 'irish', 'italian',
        'jamaican', 'japanese', 'africa', 'african', 'mexican', 'alaska', 'alaskan', 'german'
    ])
    #stop_words.extend(["aaaagghh", "aaagh", "aaah", "meh"])

    if tokenizer is not None:
        stop_words = tokenizer(" ".join(stop_words))

    return stop_words

def get_word_counts(corpus, beer_names, tokenizer=None, token_pattern=THREE_LETTERS, **vectorizer_kwargs):
    """Get the word counts in the corpus. This takes some time, especially when using a tokenizer.

    This is separate from computing the tf-idf scores because we may want to do something simple
    like subtracting frequencies when looking for negative words.
    """
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

def top_attributes_by(beers, by, top_count=5, column_count=10, column_prefix="attr_"):
    attributes = beers.melt(
        id_vars=by, 
        value_vars=[f'{column_prefix}{i+1}' for i in range(column_count)],
        var_name='attribute_type', 
        value_name='attribute'
    )

    attribute_counts = attributes.groupby([by, 'attribute']).size().reset_index(name='count')

    top_attributes = (
        attribute_counts.sort_values(by=[by, 'count'], ascending=[True, False])
        .groupby(by)
        .head(top_count)
    )

    top_attributes_pivoted = (
        top_attributes.assign(rank=top_attributes.groupby(by).cumcount() + 1)
        .pivot(index=by, columns='rank', values='attribute')
        .reset_index()
    )

    top_attributes_pivoted.columns = [by] + [f"top_{column_prefix}{i+1}" for i in range(top_count)]

    return top_attributes_pivoted

from transformers import pipeline
from src.utils import tqdm

def classify_beer_attributes(beers, column_count=10, column_prefix="attr_", device="cuda"):
    top_attributes = beers[["beer_name"] + [f"{column_prefix}{i+1}" for i in range(column_count)]].set_index("beer_name")
    print(beers[f"{column_prefix}11"])

    classified_attributes = pd.DataFrame(columns=["appearance", "aroma", "palate", "taste"], index=top_attributes.index)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    categories = ["appearance", "aroma", "palate", "taste"]

    for i in tqdm(np.arange(top_attributes.shape[0])):
        features = top_attributes.iloc[i].values
        appearance = []
        aroma = []
        palate = []
        taste = []
        for feature in features:
            result = classifier(feature, candidate_labels=categories)
            if result['labels'][0] == "appearance" and result['scores'][0]>0.5:
                appearance.append(feature)
            elif result['labels'][0] == "aroma" and result['scores'][0]>0.5:
                aroma.append(feature)
            elif result['labels'][0] == "palate" and result['scores'][0]>0.5:
                palate.append(feature)
            elif result['labels'][0] == "taste" and result['scores'][0]>0.5:
                taste.append(feature)
            #print(f"Feature: {feature}")
            #print(f"Predicted Category: {result['labels'][0]} (Score: {result['scores'][0]:.4f})\n")
        classified_attributes.loc[top_attributes.index[i], 'appearance'] = appearance
        classified_attributes.loc[top_attributes.index[i], 'aroma'] = aroma
        classified_attributes.loc[top_attributes.index[i], 'palate'] = palate
        classified_attributes.loc[top_attributes.index[i], 'taste'] = taste

    return classified_attributes
