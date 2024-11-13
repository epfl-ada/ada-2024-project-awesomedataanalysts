import pandas as pd

from .utils import tqdm


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
        with open(file_path) as file:
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
    beers = beers.merge(breweries, left_on="brewery_id", right_on="id", how="left")

    users = pd.read_csv(data_path + "/users.csv")
    users["user_id"] = users["user_id"].astype("string")
    users["joined"] = pd.to_datetime(users["joined"], unit="s")

    return beers, breweries, users
