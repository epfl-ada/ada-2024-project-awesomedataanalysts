import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_nbr_reviews_per_user(data):
    beeradvocate_user_reviews = data['beeradvocate_users'].groupby('user_id')['nbr_reviews'].sum()
    ratebeer_user_reviews = data['ratebeer_users'].groupby('user_id')['nbr_reviews'].sum()

    beeradvocate_user_reviews_df = beeradvocate_user_reviews.reset_index()
    beeradvocate_user_reviews_df.columns = ['user_id', 'total_reviews']

    ratebeer_user_reviews_df = ratebeer_user_reviews.reset_index()
    ratebeer_user_reviews_df.columns = ['user_id', 'total_reviews']

    # Histograms
    fig = go.Figure()

    fig.add_trace(go.Histogram(
    x=beeradvocate_user_reviews_df['total_reviews'],
    nbinsx=20,  
    name='BeerAdvocate User Reviews',
    opacity=0.7
    ))

    fig.add_trace(go.Histogram(
    x=ratebeer_user_reviews_df['total_reviews'],
    nbinsx=20, 
    name='Ratebeer User Reviews', 
    opacity=0.7
    ))


    fig.update_layout(
    title=dict(
        text='Comparison Of Total Reviews Per User',
        x=0.5, 
        xanchor='center' 
    ),
    xaxis=dict(
        title='Sum Of Reviews Per User',
        range=[0, 25000],
    ),
    yaxis=dict(
        title='Frequency (Log Scale)',
        type='log',  
        range=[0, 5],  
        showgrid=True  
    ),
    barmode='overlay',  
    template='plotly_white'
)

    fig.update_traces(opacity=0.5)

    fig.show()


def classify_location(location):
    if pd.isnull(location):
        return None 
    elif "United States" in location:
        return "United States"
    else:
        return "Other Countries"


def plot_n_brewery_location(data):

    # Adding location category
    data['./data/beeradvocate_breweries']['location_category'] = data['./data/beeradvocate_breweries']['location'].apply(classify_location)
    data['./data/ratebeer_breweries']['location_category'] = data['./data/ratebeer_breweries']['location'].apply(classify_location)
    # Calculating total number of beers per location
    beeradvocate_beer_totals = data['./data/beeradvocate_breweries'].groupby('location_category')['nbr_beers'].sum().reset_index()
    beeradvocate_beer_totals.columns = ['Location', 'Total Beers']
    # Filtering out NaN rows
    beeradvocate_beer_totals = beeradvocate_beer_totals[~beeradvocate_beer_totals['Location'].isna()]
    # Calculating total number of beers per location
    ratebeer_beer_totals = data['./data/ratebeer_breweries'].groupby('location_category')['nbr_beers'].sum().reset_index()
    ratebeer_beer_totals.columns = ['Location', 'Total Beers']
    # Filtering out NaN rows
    ratebeer_beer_totals = ratebeer_beer_totals[~ratebeer_beer_totals['Location'].isna()]
    max_y = max(beeradvocate_beer_totals['Total Beers'].max(), ratebeer_beer_totals['Total Beers'].max())
    # Define custom colors
    color_map = {"United States": "#87CEFA",  "Other Countries": "pink"}
    fig = make_subplots(rows=1, cols=2,  subplot_titles=("BeerAdvocate Breweries (Total Beers)", "Ratebeer Breweries (Total Beers)"))

    fig.add_trace(go.Bar(
        x = beeradvocate_beer_totals['Location'],
        y = beeradvocate_beer_totals['Total Beers'],
        name = "BeerAdvocate Breweries",
        marker_color = [color_map[loc] for loc in beeradvocate_beer_totals['Location']]),row = 1, col = 1)

    fig.add_trace(go.Bar(
        x = ratebeer_beer_totals['Location'],
        y = ratebeer_beer_totals['Total Beers'],
        name = "Ratebeer Breweries",
        marker_color = [color_map[loc] for loc in ratebeer_beer_totals['Location']]  ),row = 1, col = 2)
    # axes
    fig.update_yaxes(range = [0, max_y], title_text = "Total Number of Beers", row = 1, col = 1)
    fig.update_yaxes(range = [0, max_y], title_text = "Total Number of Beers", row = 1, col = 2)
    fig.update_xaxes(title_text = "Location Category", row = 1, col = 1)
    fig.update_xaxes(title_text = "Location Category", row = 1, col = 2)
    fig.update_layout(title = dict(text = "Comparison of Total Beers by Location in Different Sites",x = 0.5, xanchor = "center" ),
    showlegend = False,
    template = "plotly_white",
    height = 500,  
    width = 1000)

    fig.show()