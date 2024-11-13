import string
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tag import PerceptronTagger
from nltk.tokenize import regexp_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

from .data_loading import iter_reviews


def preprocess(text):
    text = text + "."	
    custom_pattern = r'[^.!?;]+(?:\s*-\s*)?[^.!?;]+[.!?;]*'
    sentences = regexp_tokenize(text, custom_pattern)
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    s = []
    final_toks = []
    final_toks2 = []
    final_adjs = []
    final_adjs2 = []
    tagger = PerceptronTagger()
    for i in sentences:
        adjs = []
        text = i.translate(translator)
        tokens1 = word_tokenize(text)
        tokens = [word.lower() for word in tokens1 if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in tokens]
        tagged_tokens = pos_tag(tokens)
        new_tag = [(word, tag) for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB')]
        
        stack = []
        for j in np.arange(len(new_tag)-1, -1, -1):
            stack.append(new_tag[j])
        a = ""	
        while len(stack) > 0:
            if len(stack) == 1:
                a = a + " " + stack[-1][0]
                adjs.append(a.strip())
                stack.pop()
            elif stack[-1][1].startswith('NN'):
                a = a + " " + stack[-1][0]
                adjs.append(a.strip())
                stack.pop()
                a = ""
            elif stack[-1][1].startswith('JJ') or stack[-1][1].startswith('RB'):
                a = a + " " + stack[-1][0]
                stack.pop()
            else:
                stack.pop()
                
        filtered_tokens1 = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB')]
    
        if len(filtered_tokens1) > 0:
            filtered_tokens = [token for token in filtered_tokens1 if token.isalpha()]
            final_toks.append(filtered_tokens)
            final_toks2.extend(filtered_tokens)
            final_adjs.append(adjs)
            final_adjs2.extend(adjs)

    return final_toks, final_toks2, final_adjs, final_adjs2

def plot_wordcloud():
    n = 1000
    tokenized_texts = [preprocess(review['text'])[1] for review in iter_reviews("./data/RateBeer/reviews.txt", max_reviews=n)]
    all_tokens = [token for text in tokenized_texts for token in text]
    bigrams = ngrams(all_tokens, 1)
    bigram_freq = Counter(bigrams)
    threshold = bigram_freq.most_common(1)[0][1]*1.0
    filtered_counter = Counter({word: freq for word, freq in bigram_freq.items() if freq <= threshold})
    bigram_strings = [' '.join(b) for b in filtered_counter.keys()]
    
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(bigram_strings))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()