import pandas as pd

from .utils import tqdm
import re

# note that we load all ids as strings
REVIEW_DATA_CASTS = {
    "date": int, "abv": float, "appearance": float, "aroma": float,
    "palate": float, "taste": float, "overall": float, "rating": float,
    "review": lambda x: True if x == "True" else False
}

FILE_RATINGS_COUNT = {
    "BeerAdvocate/reviews.txt": 2589586,
    "BeerAdvocate/ratings.txt": 8393032,
    "RateBeer/reviews.txt": 7122074,
    "RateBeer/ratings.txt": 7122074,
}

def iter_reviews(file_path, max_reviews=None, verbose=True, do_cast=True, **tqdm_args):
    def inner():
        with open(file_path, encoding="utf-8") as file:
            review = {}
            review_count = 0
            for line in file:
                if max_reviews is not None and review_count >= max_reviews:
                    break

                line = line.strip("\n")
                if not line: # all newlines stripped so line is empty
                    if review:
                        yield review
                        review_count += 1
                        review = {}
                else:
                    key_sep = line.index(": ")
                    key, val = line[:key_sep], line[key_sep+2:]
                    cast = REVIEW_DATA_CASTS.get(key) if do_cast else None
                    review[key] = cast(val) if cast is not None else val
    
            if review:
                yield review

    if verbose:
        if "total" not in tqdm_args:
            if max_reviews is not None:
                tqdm_args["total"] = max_reviews
            else:
                *_, dataset, file = file_path.split("/")
                total = FILE_RATINGS_COUNT.get(f"{dataset}/{file}")
                tqdm_args["total"] = total
                
        return tqdm(inner(), **tqdm_args)
    return inner()

# load all ids as strings, otherwise we get disparities because for example ids are int64 numbers in RateBeer
def load_beers_breweries_users(data_path):
    beers = pd.read_csv(data_path + "/beers.csv")
    beers["beer_id"] = beers["beer_id"].astype("string")
    beers = beers.drop(columns=["brewery_name"])

    breweries = pd.read_csv(data_path + "/breweries.csv")
    beers = beers.merge(breweries[["id", "location"]], left_on="brewery_id", right_on="id", how="left")
    beers = beers.drop(columns=["id"])

    users = pd.read_csv(data_path + "/users.csv")
    users["user_id"] = users["user_id"].astype("string")
    users["joined"] = pd.to_datetime(users["joined"], unit="s")

    return beers, breweries, users

def data_load(path_to_rating, nb_reviews):
    """
    Loads nb_reviews reviews (or all) wtih score and beer_name
    """
    with open(path_to_rating, encoding="utf-8") as input_file:
        if nb_reviews == "all":
            rb_ratings = input_file.read().splitlines()
        else:
            rb_ratings = [next(input_file) for _ in range(nb_reviews * 17)]  # 17 is the number of lines for each review

    rb_ratings_text = [x.replace("text: ", "")
                  .replace("\n", "")
                  .replace("â\x80\x99", "'")
                  for x in rb_ratings 
                  if x.startswith("text:")]

    rb_ratings_num = [round(float(x.replace("rating: ", "")
                    .replace("\n", "")))
                    for x in rb_ratings 
                    if x.startswith("rating:")]

    rb_beer_name = [x.replace("beer_name: ", "")
                    .replace("\n", "")
                    for x in rb_ratings 
                    if x.startswith("beer_name:")] 
    
    rb_user_id = [int(x.replace("user_id: ", "")
                    .replace("\n", ""))
                    for x in rb_ratings 
                    if x.startswith("user_id:")] 
    
    data = {'review': rb_ratings_text, 'score': rb_ratings_num, "beer_name" : rb_beer_name, "user_id" : rb_user_id} 

    return pd.DataFrame(data)

def build_reviews_corpus(reviews, sample_beers=10, expert_ids=None, expert_weight=2, random_state=23):
    reviews_by_beer = reviews.groupby("beer_name")

    unique_beers = len(reviews_by_beer)

    if sample_beers == "all" or sample_beers > unique_beers:
        sample_beers = unique_beers

    beer_names = list(reviews_by_beer.size().sample(sample_beers, random_state=random_state).index)

    corpus = []
    for beer_name in beer_names:
        beer_reviews = reviews_by_beer.get_group(beer_name)

        if expert_ids is not None:
            expert_reviews = beer_reviews[beer_reviews["user_id"].isin(expert_ids)]
            beer_reviews = pd.concat([beer_reviews] + (expert_weight - 1) * [expert_reviews], ignore_index=True)

        document = "\n".join(beer_reviews["text"])

        # some apostrophes have been replaced by â\x80\x99 (â) this is due to an encoding issue beyond our control
        document = re.sub(r"â\x80\x99", "’", document)

        # the word ipa is often missspelled (eg ipaa) and the lemmatizer doesn't remove the s from the plural ipas
        document = re.sub(r"\b\w*ipa\w*\b", "ipa", document, flags=re.IGNORECASE)

        corpus.append(document)

    return corpus, beer_names