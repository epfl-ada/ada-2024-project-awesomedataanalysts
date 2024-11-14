import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import regexp_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def preprocess(text):
    text = text + "."	
    custom_pattern = r'[^.!?;]+(?:\s*-\s*)?[^.!?;]+[.!?;]*'
    sentences = regexp_tokenize(text, custom_pattern)
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    final_toks = []
    for i in sentences:
        text = i.translate(translator)
        tokens1 = word_tokenize(text)
        tokens = [word.lower() for word in tokens1 if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in tokens]
        tagged_tokens = pos_tag(tokens)        
        filtered_tokens1 = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB')]
        if len(filtered_tokens1) > 0:
            filtered_tokens = [token for token in filtered_tokens1 if token.isalpha()]
            final_toks.extend(filtered_tokens)
            
    return final_toks

def plot_wordcloud(data):
    tokenized_texts = [preprocess(data[i]) for i in range(len(data))]
    all_tokens = [token for text in tokenized_texts for token in text]
    bigrams = ngrams(all_tokens, 1)
    bigram_freq = Counter(bigrams)
    threshold = bigram_freq.most_common(1)[0][1]*1.0
    filtered_counter = Counter({word: freq for word, freq in bigram_freq.items() if freq <= threshold})
    bigram_strings = [' '.join(b) for b in filtered_counter.keys()]
    
    
    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(' '.join(bigram_strings))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def tf_idf(data, ngrams=(1, 2)):
    tokenized_texts = [preprocess(data[i]) for i in range(len(data))]
    tokenized_texts_joined = [" ".join(tokens) for tokens in tokenized_texts]
    vectorizer = TfidfVectorizer(ngram_range=ngrams)
    tfidf_matrix = vectorizer.fit_transform(tokenized_texts_joined)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()
    term_frequencies = np.sum(tfidf_array, axis=0)
    word_frequency_df = pd.DataFrame(list(zip(feature_names, term_frequencies)), columns=["Word", "TF"])
    f_words = word_frequency_df.sort_values(by="TF", ascending=False)
    return f_words
