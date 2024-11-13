import numpy as np
import pandas as pd

from .data_loading import iter_reviews


# eg review_counts_by(review_file, "beer_id") for the number of reviews for each beer
def review_counts_by(review_file, by, **iter_reviews_kwargs):
    review_counts = {}

    for review in iter_reviews(review_file, **iter_reviews_kwargs):
        current_count = review_counts.setdefault(review[by], 0)
        review_counts[review[by]] = current_count + 1

    return review_counts

# eg review_avg_by(review_file, "beer_id", "overall") for the average `overall` value for each beer
def review_avg_by(review_file, by, on, **iter_reviews_kwargs):
    review_items = {}

    for review in iter_reviews(review_file, **iter_reviews_kwargs):
        review_items.setdefault(review[by], []).append(review[on])

    return {k: np.mean(review_items[k]) for k in review_items}

def add_review_columns(review_file, df, by):
    review_counts = pd.DataFrame(review_counts_by(review_file, by).items(), columns=[by, "review_count"])
    df = df.merge(review_counts, on=by, how="inner")

    review_overall = pd.DataFrame(review_avg_by(review_file, by, "overall").items(), columns=[by, "avg_overall"])
    df = df.merge(review_overall, on=by, how="inner")

    return df