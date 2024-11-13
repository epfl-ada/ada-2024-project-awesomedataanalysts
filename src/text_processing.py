import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from string import punctuation

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# inspired by https://gist.github.com/4OH4/f727af7dfc0e6bb0f26d2ea41d89ee55
class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def _wordnet_pos(self, word):
        """Map POS (part of speech) tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def __call__(self, text):
        words = word_tokenize(text)
        return [self.lemmatizer.lemmatize(w, self._wordnet_pos(w)) for w in words]


LEMMATIZER = Lemmatizer()

STOP_WORDS = stopwords.words("english") + list(punctuation)
STOP_WORDS = [sw for sw in STOP_WORDS if sw not in ["no", "not"]] # don't remove negatives
STOP_WORDS_LEMMATIZED = LEMMATIZER(" ".join(STOP_WORDS))

def word_freq(corpus, lemmatize=False, ngrams=(1, 1)):
    if lemmatize:
        vectorizer = CountVectorizer(
            stop_words=STOP_WORDS_LEMMATIZED, ngram_range=ngrams,
            tokenizer=LEMMATIZER, token_pattern=None)
    else:
        vectorizer = CountVectorizer(stop_words=STOP_WORDS, ngram_range=ngrams)

    X = vectorizer.fit_transform(corpus)

    word_counts = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "freq": np.array(X.sum(axis=0))[0]
    })

    word_counts = word_counts[~word_counts["word"].isin(STOP_WORDS)]

    return word_counts

def top_negative_words(corpus_neg, corpus_pos, use_tfidf=False, lemmatize=False, ngrams=(1, 1)):
    if use_tfidf:
        all_reviews = corpus_neg + corpus_pos
        labels = ["negative"] * len(corpus_neg) + ["positive"] * len(corpus_pos)

        if lemmatize:
            tfidf_vectorizer = TfidfVectorizer(
                stop_words=STOP_WORDS_LEMMATIZED, ngram_range=ngrams,
                tokenizer=LEMMATIZER, token_pattern=None)
        else:
            tfidf_vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=ngrams)

        tfidf_matrix = tfidf_vectorizer.fit_transform(all_reviews)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_df["label"] = labels

        positive_tfidf = tfidf_df[tfidf_df["label"] == "positive"].drop(columns=["label"]).mean()
        negative_tfidf = tfidf_df[tfidf_df["label"] == "negative"].drop(columns=["label"]).mean()

        top_negative_words = pd.DataFrame({
            "avg_pos_score": positive_tfidf,
            "avg_neg_score": negative_tfidf,
            "score_diff": negative_tfidf - positive_tfidf
        }).sort_values(by="score_diff", ascending=False)

        return top_negative_words

    else: # use word frequencies
        freq_neg = word_freq(corpus_neg, lemmatize=lemmatize, ngrams=ngrams)
        freq_pos = word_freq(corpus_pos, lemmatize=lemmatize, ngrams=ngrams)

        combined = freq_neg.merge(freq_pos, on="word", how="outer", suffixes=("_neg", "_pos"))
        combined = combined.fillna(0)
        combined["freq_diff"] = combined["freq_neg"] - combined["freq_pos"]

        top_negative_words = combined.sort_values(by="freq_diff", ascending=False)
        return top_negative_words.set_index("word")
