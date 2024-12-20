import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Calculating percentages for attributes
def calculate_percentages(row):
    attributes = ['appearance', 'aroma', 'taste', 'palate']
    word_counts = {attr: len(eval(row[attr])) for attr in attributes}
    total_words = sum(word_counts.values())
    return {attr.capitalize(): (count / total_words) * 100 for attr, count in word_counts.items()}

def plot_pie_chart_by_style(datapath):
    data = pd.read_csv(datapath)
    # Sample selected styles
    style_sample = [6, 22, 30, 9, 73, 53]
    selected_style = data.iloc[style_sample]

    # Using Six Beer Style Randomly to show the plot, notable that since we wanted to show exact number of Criticism, we extract them manually.
    styles_data = {
    'American Pale Ale': {
        'Appearance': ['Rotten', 'Puffy', 'Cardboard', 'Plain', 'Chalk'],
        'Aroma': ['Ripe', 'Roasted', 'Dank', 'Skunk', 'Oily'],
        'Taste': ['Pale', 'Bland', 'Stale', 'Cheap', 'Sap'],
        'Palate': ['Salty', 'Pepper', 'Citric', 'Bitter', 'Resiny']
    },
    'Dortmunder/Helles': {
        'Appearance': ['Rotten', 'Faint', 'Cardboard', 'Metal', 'Ordinary'],
        'Aroma': ['Skunk', 'Pungent', 'Oily', 'Smell', 'Ripe'],
        'Taste': ['Bland', 'Boring', 'Stale', 'Cheap', 'Crap'],
        'Palate': ['Bitter', 'Sour', 'Salty', 'Aftertaste', 'Cheese']
    },
    'Foreign Stout': {
        'Appearance': ['Rotten', 'Ashy', 'Cardboard', 'Scorch', 'Blackish'],
        'Aroma': ['Pungent', 'Oily', 'Acrid', 'Ripe', 'Vinous'],
        'Taste': ['Infection', 'Boring', 'Pale', 'Carbonated', 'Alcohol'],
        'Palate': ['Vinegar', 'Sourness', 'Bitterness', 'Salty', 'Syrup']
    },
    'Barley Wine': {
        'Appearance': ['Rotten', 'Cloy', 'Cardboard', 'Muddy', 'Leather'],
        'Aroma': ['Pungent', 'Ripe', 'Vinous', 'Oily', 'Perfume'],
        'Taste': ['Pale', 'Gross', 'Solvent', 'Numb', 'Alcohol'],
        'Palate': ['Salty', 'Boozy', 'Vinegar', 'Syrup', 'Whiskey']
    },
    'Sweet Stout': {
        'Appearance': ['Otten', 'Artificial', 'Ashy', 'Thinner', 'Cardboard'],
        'Aroma': ['Pungent', 'Ripe', 'Oily', 'Acrid', 'Perfume'],
        'Taste': ['Pale', 'Cheap', 'Disappointment', 'Cellar', 'Pale'],
        'Palate': ['Salty', 'Bitter', 'Sourness', 'Liquorice', 'Sour']
    },
    'Old Ale': {
        'Appearance': ['Rotten', 'Cardboard', 'Fatty', 'Bandaid', 'Ashy'],
        'Aroma': ['Pungent', 'Ripe', 'Vinous', 'Worty', 'Perfume'],
        'Taste': ['Ruin', 'Bland', 'Cough', 'Peaty', 'Boring'],
        'Palate': ['Salty', 'Boozy', 'Sourish', 'Buttery', 'Syrup']
    }
}


    # Generating traces for each style
    traces = []
    buttons = []
    style_names = list(styles_data.keys())
    custom_colors = ['#FFB6C1', '#87CEFA', '#98FB98', '#FFD700']

    for idx, (style_name, style_data) in enumerate(styles_data.items()):
        percentages = calculate_percentages(selected_style.iloc[idx])  
   
        hover_texts = [
        f"Some Features: {', '.join(style_data['Appearance'])}",
        f"Some Features: {', '.join(style_data['Aroma'])}",
        f"Some Features: {', '.join(style_data['Taste'])}",
        f"Some Features: {', '.join(style_data['Palate'])}"
        ]
   
        trace = go.Pie(
            labels = ['Appearance', 'Aroma', 'Taste', 'Palate'],
            values = [
            percentages['Appearance'],
            percentages['Aroma'],
            percentages['Taste'],
            percentages['Palate']
        ],
        textinfo = 'label+percent',
        hoverinfo = 'text',
        marker = dict(colors = custom_colors),
        textfont = dict(family = "Georgia", size = 16, color = "black"),
        hovertext = hover_texts,
        name = style_name,
        visible = (idx == 0)
        )
        traces.append(trace)

        # Creating a button for each style
        button = dict(
        label=style_name,
        method="update",
        args=[
            {"visible": [i == idx for i in range(len(style_names))]},
            {"title": f"Distribution of '{style_name}' Style Criticism"}
        ]
        )
        buttons.append(button)

    # Create the figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title = dict(
        text = f"Distribution of '{style_names[0]}' Style Criticisms",
        font = dict(family="Georgia", size=24, color='darkblue'),
        x = 0.5,
        xanchor = 'center'
    ),
    updatemenus=[dict(
        type = "buttons",
        direction = "up",
        x = 0,
        xanchor = "right",
        bgcolor = '#F5F5DC',
        y = 0.2,
        yanchor = "top",
        buttons = buttons
    )],
    height = 600,
    width = 800,
    showlegend = True
    )

    fig.update_layout(
    legend = dict(
        font = dict(
            family = "Georgia",
            size = 14,          
            color = "darkblue"  
        ),
        title = dict(
            text = "Categories",  
            font = dict(
                family = "Georgia",
                size = 16,
                color = "darkgreen"
                )
            )
     )
    )

    pio.write_html(fig, file = "beer_styles_distribution.html", full_html=True)

    fig.show()