from transformers import pipeline
import random 

def emotion_sentiment(data, cuda=False):
    """Compute the emotion scores for anger, disgust, fear, joy, neutral, sadness and surprise."""
    if cuda:
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device="cuda")
    else:
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        
    emotions = {"anger" : [], "disgust": [], 'fear' : [], 'joy' : [], 'neutral' : [], 'sadness' : [], 'surprise' : []}
    total = len(data)

    for x in data["review"]:
        if len(x) < 500: # The length constraint is because the model has a length constraint
            result = classifier(x)[0]
            for emotion in result:
                emotions[emotion["label"]].append(emotion["score"])
            if len(emotions["anger"]) % 100 == 0:
                print(f"{(len(emotions['anger'])/total * 100):.2f} % done", end='\r') 
        else:
            for emotion in result:
                emotions[emotion["label"]].append(0)

    for label, values in emotions.items():
        data[label] = values

    # Create a additionnal column representing the feeling with the maximum score according to the model
    data["max_feel"] = data[["anger", "disgust", 'fear', 'joy', 'neutral', 'sadness', 'surprise']].idxmax(axis=1)
    
    return data


def polarity_sentiment(data, cuda=False):
    """We consider negative review to be those with scores 0, 1 or 2, and we will try to find those that have an extreme
    vocabulary using a sentiment analysis model."""
    if cuda:     
        sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device="cuda")
    else:    
        sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    negative_reviews = data[(data["score"] == 0) | (data["score"] == 1) | (data["score"] == 2)].copy()

    total = len(negative_reviews)
    polarity = []
    for x in negative_reviews["review"]:
        if len(x) < 500:
            result = sentiment_task(x)[0]
            if result["label"] != "negative":
                polarity.append(0)
            else:
                polarity.append(result["score"])
            if len(polarity) % 100 == 0:
                print(f"{(len(polarity)/total * 100):.2f} % done", end='\r') 
        else:
            polarity.append(0)
        
    negative_reviews["polarity"] = polarity

    return negative_reviews

def summarize(df):
    """Summarize reviews."""
    summarizer = pipeline("summarization", device="cuda")
    all_complaints = ' '.join(df['review'])

    def split_text(text, chunk_size=750):
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])

    chunks = list(split_text(all_complaints))
    summaries = [summarizer(chunk[:1000], max_length=100, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    final_summaries = []

    for i in range(100):
        final_summaries.append(summarizer(' '.join(random.choices(summaries, k=200))[:800], max_length=100, min_length=50, do_sample=False)[0]['summary_text'])
    
    final_summary = summarizer(' '.join(random.choices(final_summaries, k=50))[:800], max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    final_summary = summarizer(final_summary, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    
    print("Final Summary of All Complaints:")
    print(final_summary)
    return final_summary

from transformers import pipeline
from src.utils import tqdm

def classify_beer_attributes(criticisms, device="cuda", by = 'location'):
    """Classify beer attributes into appearance, aroma, palate or taste."""
    top_attributes = criticisms
    classified_criticisms = pd.DataFrame(columns=["appearance", "aroma", "palate", "taste"], index=top_attributes.index)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    categories = ["appearance", "aroma", "palate", "taste"]

    for i in tqdm(np.arange(top_attributes.shape[0])):
    
        features = top_attributes.iloc[i, 1::].dropna().values
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
            
        classified_criticisms.loc[top_attributes.iloc[i, 0], 'appearance'] = appearance
        classified_criticisms.loc[top_attributes.iloc[i, 0], 'aroma'] = aroma
        classified_criticisms.loc[top_attributes.iloc[i, 0], 'palate'] = palate
        classified_criticisms.loc[top_attributes.iloc[i, 0], 'taste'] = taste

    return classified_criticisms
