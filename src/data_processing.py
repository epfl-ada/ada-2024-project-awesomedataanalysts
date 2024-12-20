import numpy as np
import pandas as pd
import re

from .data_loading import iter_reviews

def process_document(document):
    """Fix apostrophe encoding, and 'ipa' spelling."""
    # some apostrophes have been replaced by â\x80\x99 (â) this is due to an encoding issue beyond our control
    document = re.sub(r"â\x80\x99", "’", document)

    # the word ipa is often missspelled (eg ipaa) and the lemmatizer doesn't remove the s from the plural ipas
    document = re.sub(r"\b\w*ipa\w*\b", "ipa", document, flags=re.IGNORECASE)

    return document

def build_corpus_from_reviews(reviews, expert_ids=None, expert_weight=2):
    """Build the corpus from the reviews, giving higher weight to expert users."""
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
    """Split reviews into positive and negative, at the q-th quantile of the given column."""
    percentiles = reviews.groupby("beer_name")[column].quantile(q / 100).rename("percentile")
    reviews = reviews.merge(percentiles, on="beer_name")
    worst_reviews = reviews[reviews[column] <= reviews["percentile"]]
    best_reviews = reviews[reviews[column] > reviews["percentile"]]
    return best_reviews, worst_reviews

def build_split_corpus_from_emotion_analysis(reviews, expert_ids=None, expert_weight=2):
    """Build positive-negative corpus using the emotion analysis results, giving higher weight to expert users."""
    beer_names = reviews["beer_name"].unique()

    # TODO what about experts and process_document ? emotion analysis model has its own pipeline, can't do this
    # TODO make pickle or (better) upload csv. tokens_feeling = pd.read_pickle("./data/review_with_tokens_emotions.pkl")
    tokens_feeling = pd.read_csv("./data/RateBeer_processed/reviews_with_tokens_emotions.csv")

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
    """Build the corpus as specified. Options include using experts and splitting by quantile or by emotion.
    This is the function the user should use."""
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

def review_counts_by(review_file, by, **iter_reviews_kwargs):
    """Count the number of reviews by column *by*.

    >>> review_counts_by(review_file, "beer_id") # number of reviews for each beer"""
    review_counts = {}

    for review in iter_reviews(review_file, **iter_reviews_kwargs):
        current_count = review_counts.setdefault(review[by], 0)
        review_counts[review[by]] = current_count + 1

    return review_counts

def review_avg_by(review_file, by, on, **iter_reviews_kwargs):
    """Compute the average value of reviews[on] by column *by*.
    
    >>> review_avg_by(review_file, "beer_id", "overall") # average `overall` value for each beer
    """
    review_items = {}

    for review in iter_reviews(review_file, **iter_reviews_kwargs):
        review_items.setdefault(review[by], []).append(review[on])

    return {k: np.mean(review_items[k]) for k in review_items}

def add_review_columns(review_file, df, by):
    """Add review counts and average overall rating for each beer."""
    review_counts = pd.DataFrame(review_counts_by(review_file, by).items(), columns=[by, "review_count"])
    df = df.merge(review_counts, on=by, how="inner")

    review_overall = pd.DataFrame(review_avg_by(review_file, by, "overall").items(), columns=[by, "avg_overall"])
    df = df.merge(review_overall, on=by, how="inner")

    return df
