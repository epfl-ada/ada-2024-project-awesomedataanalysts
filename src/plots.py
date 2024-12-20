import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import scipy.stats as stats
import seaborn as sns

def classify_location(location):
    if pd.isnull(location):
        return None 
    elif "United States" in location:
        return "United States"
    else:
        return "Other Countries"

def plot_review_count_comparison(rb_beers, ba_beers, q=90):
    rb_reviews_sum = rb_beers.groupby('beer_id')['review_count'].sum()
    ba_reviews_sum = ba_beers.groupby('beer_id')['review_count'].sum()

    rb_reviews_sum_df = rb_reviews_sum.reset_index()
    rb_reviews_sum_df.columns = ['beer_id', 'total_reviews']

    ba_reviews_sum_df = ba_reviews_sum.reset_index()
    ba_reviews_sum_df.columns = ['beer_id', 'total_reviews']

    rb_reviews = rb_reviews_sum_df['total_reviews']
    ba_reviews = ba_reviews_sum_df['total_reviews']

    plt.figure(figsize = (10, 4))

    plt.hist(rb_reviews, bins = 20, alpha = 0.5, label = f'RateBeer ({q}-th quantile = {rb_reviews.quantile(q / 100)})', color = 'blue', edgecolor = 'black')
    plt.hist(ba_reviews, bins = 20, alpha = 0.5, label = f'BeerAdvocate ({q}-th quantile = {ba_reviews.quantile(q / 100)})', color = 'pink', edgecolor = 'black')

    plt.yscale('log')

    plt.title('Comparison of Total Reviews per Beer', fontsize = 14)
    plt.xlabel('# Reviews', fontsize = 12)
    plt.ylabel('Frequency (Log Scale)', fontsize = 12)
    plt.legend(loc='upper right', fontsize = 10)

    plt.tight_layout()


def plot_location_comparison(rb_breweries, ba_breweries):
    rb_breweries['location_category'] = rb_breweries['location'].apply(classify_location)
    ba_breweries['location_category'] = ba_breweries['location'].apply(classify_location)

    ba_beer_totals = ba_breweries.groupby('location_category')['nbr_beers'].sum().reset_index()
    ba_beer_totals.columns = ['Location', 'Total Beers']
    ba_beer_totals = ba_beer_totals[~ba_beer_totals['Location'].isna()]

    rb_beer_totals = rb_breweries.groupby('location_category')['nbr_beers'].sum().reset_index()
    rb_beer_totals.columns = ['Location', 'Total Beers']
    rb_beer_totals = rb_beer_totals[~rb_beer_totals['Location'].isna()]

    max_y = max(
        ba_beer_totals['Total Beers'].max()+50000,
        rb_beer_totals['Total Beers'].max()
    )

    color_map = {
        "United States": "#2B65EC",  # Blue
        "Other Countries": "pink"  # pink
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].bar(
        ba_beer_totals['Location'],
        ba_beer_totals['Total Beers'],
        color = [color_map[loc] for loc in ba_beer_totals['Location']]
    )
    axes[0].set_title("BeerAdvocate")
    axes[0].set_xlabel("Location")
    axes[0].set_ylabel("# Beers")
    axes[0].set_ylim(0, max_y)

    axes[1].bar(
        rb_beer_totals['Location'],
        rb_beer_totals['Total Beers'],
        color = [color_map[loc] for loc in rb_beer_totals['Location']]
    )
    axes[1].set_title("Ratebeer")
    axes[1].set_xlabel("Location")

    fig.suptitle("Comparison of Beers by Location", fontsize = 16)
    fig.tight_layout(rect = [0, 0, 1, 0.95])

def plot_histogram_with_percentile(data, title, q=50, **hist_kwargs):
    percentile = np.percentile(data, q)
    n = np.sum(np.array(data) > percentile)

    plt.hist(data, **hist_kwargs)
    plt.axvline(percentile, color="r")

    plt.title(f"""{title}\n{q:.1f}th percentile (n={n:.0f}) in red corresponds to values >= {percentile:.3f} """)

    return percentile

def plot_avg_overall_map(df, title):
    # compute mean `avg_overall` by location
    location_data = df[["avg_overall", "location"]].copy()
    location_data = location_data.dropna()
    location_data["location"] = location_data["location"].apply(
        lambda x: "United States of America" if x.startswith("United States") else x
    )
    location_data = location_data.groupby("location")["avg_overall"].mean()

    # get map shapes data and combine with `avg_overall` values
    # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    world = gpd.read_file("./data/geopandas/ne_110m_admin_0_countries")
    world = world.merge(location_data, left_on='SOVEREIGNT', right_on='location', how='left')

    # plot map
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    world.boundary.plot(ax=ax)
    world.plot(column="avg_overall", ax=ax, legend=True,
               legend_kwds={"label": "average rating",
                            "orientation": "vertical"})

    # fix colorbar size
    cbar = ax.get_figure().axes[-1]
    cbar.set_position([0.77, 0.3, 0.03, 0.4]) # [left, bottom, width, height]

    ax.set_title(title)

def plot_emotion_by_score_spearman(data):
    # Compute counts for each score and emotion
    emotion_counts = data.groupby(['score'])["max_feel"].value_counts().unstack(fill_value=0)
    # Then their percentage in each score class
    emotion_percentages = emotion_counts.div(emotion_counts.sum(axis=1), axis=0) * 100

    emotion_percentages.plot(kind='bar', stacked=True, colormap="Set2", figsize=(10, 6))
    plt.title("Percentage of found emotions in each score class")
    plt.xlabel("Score")
    plt.ylabel("Emotion percentage")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.show()

    emotion_percentages = emotion_percentages.reset_index()

    # Compute the spearman correlation for each emotion between its percentage and its score
    spearman_results = {}
    for emotion in emotion_counts.columns:
        spearman_corr, p_value = stats.spearmanr(emotion_percentages['score'], emotion_percentages[emotion])
        spearman_results[emotion] = {'Spearman correlation': spearman_corr, 'p-value': p_value}

    spearman_df = pd.DataFrame(spearman_results).T
    print("Spearman correlation for emotions by score: ")
    print("(if p value < 0.05 we can say that a change in score, introducted a predictable increase or decrease in the percentage for that emotion) ")
    print(spearman_df) 

def plot_avg_score_by_topic(data_score_topic):
    topic_scores = defaultdict(float)
    topic_weights = defaultdict(float)

    for topics, score in zip(data_score_topic['topics'], data_score_topic['score']):
        for topic_num, prob in topics:
            topic_scores[topic_num] += score * prob
            topic_weights[topic_num] += prob

    avg_scores = {k: topic_scores[k] / topic_weights[k] for k in topic_scores}

    topics = list(avg_scores.keys())
    average_scores = list(avg_scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(topics, average_scores, color='green', edgecolor='black')

    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{avg_score:.2f}', ha='center', fontsize=10)

    plt.xlabel('Topic Number', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.title('Average Score for Each Topic', fontsize=14)
    plt.xticks(topics)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_topic_distrib(data_topic):
    topic_counter = Counter()
    topic_counts = Counter()

    for topics in data_topic['topics']:
        for topic_num, prob in topics:
            topic_counter[topic_num] += prob
            topic_counts[topic_num] += 1

    total_probability = sum(topic_counter.values())
    topic_percentages = {k: (v / total_probability) * 100 for k, v in topic_counter.items()}
    topic_avg_probabilities = {k: topic_counter[k] / topic_counts[k] for k in topic_counter}

    topics = list(topic_percentages.keys())
    percentages = list(topic_percentages.values())
    avg_probabilities = [topic_avg_probabilities[topic] for topic in topics]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(topics, percentages, color='skyblue', edgecolor='black')

    for bar, avg_prob in zip(bars, avg_probabilities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f'{avg_prob:.2f}', ha='center', fontsize=10)

    plt.xlabel('Topic Number', fontsize=12)
    plt.ylabel('Percentage of Total Probability', fontsize=12)
    plt.title('Distribution of Topics with Average Probability', fontsize=14)
    plt.xticks(topics)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_topic_emotions_distrib(data_emotion_topic):
    emotions = data_emotion_topic['max_feel'].unique()
    emotion_indices = {emotion: idx for idx, emotion in enumerate(emotions)}
    topic_distribution = {}

    for topics, emotion in zip(data_emotion_topic['topics'], data_emotion_topic['max_feel']):
        for topic_num, prob in topics:
            if topic_num not in topic_distribution:
                topic_distribution[topic_num] = np.zeros(len(emotions))
            topic_distribution[topic_num][emotion_indices[emotion]] += prob

    def adjust_to_shape(distribution):
        n = len(distribution)
        for i in range(n):
            distribution[i] = (distribution[i] + distribution[(i+1) % n]) / 2
        return distribution / distribution.sum()
    for topic_num in topic_distribution:
        topic_distribution[topic_num] = adjust_to_shape(topic_distribution[topic_num]) 

    def plot_radar_chart(topic_num, values, emotions):
        angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, values, color='skyblue', alpha=0.4)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(emotions, fontsize=10)
        ax.set_title(f'Topic {topic_num} Distribution Across Emotions', fontsize=14, pad=20)
        plt.show()

    for topic_num, distribution in topic_distribution.items():
        plot_radar_chart(topic_num, distribution, emotions)

def plot_score_emotions_distrib(data_score_emotions):
    scores = data_score_emotions['score'].unique()
    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    average_emotion_probabilities = {}
    for score in scores:
        group = data_score_emotions[data_score_emotions['score'] == score]
        average_emotion_probabilities[score] = group[emotions].mean().values

    def flatten_distribution(distribution):
        n = len(distribution)
        flattened = np.zeros_like(distribution)
        for i in range(n):
            flattened[i] = (distribution[i - 1] + distribution[i] + distribution[(i + 1) % n]) / 3
        return flattened / flattened.max()

    for score in average_emotion_probabilities:
        probabilities = average_emotion_probabilities[score]
        average_emotion_probabilities[score] = flatten_distribution(probabilities)

    def plot_radar_chart(score, distribution, emotions):
        angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
        values = np.concatenate((distribution, [distribution[0]])) 
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, values, color='skyblue', alpha=0.4)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_ylim(0, 1) 
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(emotions, fontsize=10)
        ax.set_title(f'Score {score} Emotion Distribution', fontsize=14, pad=20)
        plt.show()

    for score, distribution in average_emotion_probabilities.items():
        plot_radar_chart(score, distribution, emotions)

def plot_emotion_loc_distrib(data_location_emotion):
    top_locations = data_location_emotion['location'].value_counts().head(10).index
    filtered_df = data_location_emotion[data_location_emotion['location'].isin(top_locations)]
    distribution = filtered_df.groupby(['location', 'max_feel']).size().unstack(fill_value=0)
    distribution = distribution.loc[distribution.sum(axis=1).sort_values(ascending=False).index]
    distribution.plot(kind='bar', stacked=True, figsize=(12, 6))

    plt.title("Distribution of max_feel by Top 10 Locations (Ordered by Total Count)")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.legend(title="Max Feel")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_loc_emotion(data_emotion_location, threshold, emotion_name, top_n):
    location_counts = data_emotion_location['location'].value_counts()
    valid_locations = location_counts[location_counts > threshold].index
    counts = data_emotion_location[data_emotion_location['max_feel'] == emotion_name]['location'].value_counts()
    percentage = (counts / location_counts * 100).dropna()
    percentage = percentage[valid_locations]
    top_loc = percentage.sort_values(ascending=False).head(top_n)
    top_loc.plot(kind='bar', color='skyblue', figsize=(12, 6))

    plt.title(f'Top {top_n} Locations with Highest Percentage of "{emotion_name}" in max_feel (Count > {threshold})')
    plt.xlabel('Location')
    plt.ylabel(f'Percentage of "{emotion_name}" (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()