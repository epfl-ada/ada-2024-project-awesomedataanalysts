from transformers import pipeline

def emotion_sentiment(data, cuda=False):
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
    # We consider negative review to be those with scores 0, 1 or 2, and we will try to find those that have an extreme vocabulary using a 
    # sentiment analysis model
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