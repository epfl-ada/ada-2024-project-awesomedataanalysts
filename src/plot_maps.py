import pandas as pd
import plotly.graph_objects as go
import ast


def plot_top_attributes_by_location(datapath):
    df = pd.read_csv(datapath)
    state_codes = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

    ## Data Pre-processing
    states_df = df[df['location'].str.startswith('United States')].copy()
    countries_df = df[~df['location'].str.startswith('United States')].copy()
    states_df['state'] = states_df['location'].str.split(',').str[1].str.strip()
    states_df['state_code'] = states_df['state'].map(state_codes)

    def create_hover_text(row, is_state=False):
        location = row['state'] if is_state else row['location']
        attrs = [row[f'top_attr_{i}'] for i in range(1, 13) if pd.notna(row[f'top_attr_{i}'])]
        attr_text = "<br>".join([ 
        f"<span style='font-size:14px; color:teal;'>{attr}</span>"
        for attr in attrs
        ])
        avg_score = f"<br><span style='color:darkred; font-size:14px;'>Average Overall <br>Reviews Rating: {row['avg_overall']:.2f}</span>"
        return f"<b style='color:darkblue; font-size:16px;'>{location}</b>{avg_score}<br>{attr_text}"

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
    locations=countries_df['location'],
    z=countries_df['avg_overall'],
    locationmode='country names',
    colorscale='inferno',
    name='Countries',
    marker_line_color='white',
    colorbar_title="Average Overall Reviews Rating",
    hovertemplate='%{customdata}<extra></extra>',
    customdata=[create_hover_text(row) for _, row in countries_df.iterrows()]
    ))

    fig.add_trace(go.Choropleth(
    locations=states_df['state_code'],
    z=states_df['avg_overall'],
    locationmode='USA-states',
    colorscale='inferno',
    name='US States',
    marker_line_color='white',
    showscale=False,
    hovertemplate='%{customdata}<extra></extra>',
    customdata=[create_hover_text(row, True) for _, row in states_df.iterrows()]
    ))

    fig.update_layout(
    title=dict(
        text='Geographic Distribution of Beer Ratings and Top Attributes',
        font=dict(
            family='Georgia',  
            size=24,
            color='darkblue'
        ),
        x=0.5,  
        xanchor='center'
    ),
    geo=dict(
        showframe=True,
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="LightBlue",
        showlakes=False,
        lakecolor="Blue",
        showcountries=True,
        countrycolor="gray",
        projection_type="orthographic",
        resolution=50,
        center=dict(lon=0, lat=20),
    ),
    width=800,
    height=600,
    showlegend=True
    )

    fig.data[0].colorbar.titlefont = dict(
    family='Georgia', 
    size=16,
    color='darkgreen'
    )

    fig.update_traces(
    hoverlabel=dict(
        bgcolor='rgba(255, 236, 210, 0.9)',
        font=dict(color='darkblue', size=16),
        bordercolor='white',
        align='auto'
        )
    )

    fig.show()



def plot_top_complaints_by_location(datapath):
    # Load the dataset
    df = pd.read_csv(datapath)
    for col in ['appearance', 'aroma', 'palate', 'taste']:
        df[col] = df[col].apply(ast.literal_eval)

    def get_top_5(attr_list):
        return attr_list[:5] if len(attr_list) >= 5 else attr_list

    df['appearance_top5'] = df['appearance'].apply(get_top_5)
    df['aroma_top5'] = df['aroma'].apply(get_top_5)
    df['palate_top5'] = df['palate'].apply(get_top_5)
    df['taste_top5'] = df['taste'].apply(get_top_5)

    state_codes = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

    states_df = df[df['location'].str.startswith('United States')].copy()
    countries_df = df[~df['location'].str.startswith('United States')].copy()
    states_df['state'] = states_df['location'].str.split(',').str[1].str.strip()
    states_df['state_code'] = states_df['state'].map(state_codes)

    def create_hover_text(row, feature_col, is_state=False):
        location = row['state'] if is_state else row['location']
        attrs = row[feature_col]
        attr_text = "<br>".join([f"<span style='font-size:14px; color:teal;'>{attr}</span>" for attr in attrs])
        avg_score = f"<br><span style='color:darkred; font-size:14px;'>Average Overall <br>Rating: {row['avg_overall']:.2f}</span>"
        return f"<b style='color:darkblue; font-size:16px;'>{location}</b>{avg_score}<br>{attr_text}"

    fig = go.Figure()

    features = ['appearance_top5', 'aroma_top5', 'palate_top5', 'taste_top5']
    feature_names = ['Appearance', 'Aroma', 'Palate', 'Taste']

    for feature in features:
        # Countries
        fig.add_trace(go.Choropleth(
        locations=countries_df['location'],
        z=countries_df['avg_overall'],
        locationmode='country names',
        colorscale='inferno',  # Use the inferno color scale
        name='Countries',
        marker_line_color='white',
        colorbar_title=dict(
            text="<span style='font-family:Georgia;'>Average Overall Rating</span>",
            font=dict(family="Georgia", size=16, color='darkgreen')
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=[create_hover_text(row, feature) for _, row in countries_df.iterrows()],
        visible=False
    ))

        # US states
        fig.add_trace(go.Choropleth(
        locations=states_df['state_code'],
        z=states_df['avg_overall'],
        locationmode='USA-states',
        colorscale='inferno',  
        name='US States',
        marker_line_color='white',
        showscale=False,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=[create_hover_text(row, feature, True) for _, row in states_df.iterrows()],
        visible=False
        ))

    fig.data[0].visible = True
    fig.data[1].visible = True

    buttons = []
    for i, feature_name in enumerate(feature_names):
        visibility = [False] * len(fig.data)
        visibility[i*2] = True  # Countries 
        visibility[i*2 + 1] = True  # States
        buttons.append(dict(
        label=feature_name,
        method="update",
        args=[{"visible": visibility}]
    ))

    fig.update_layout(
    title=dict(
        text="<span style='font-family:Georgia;'>Geographic Distribution of Top Negative Beer Attributes by Feature</span>",
        font=dict(family="Georgia", size=24, color='darkblue'),
        x=0.5,
        xanchor='center'
    ),
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.5,  
        xanchor='center',  
        y=-0.1,
        showactive=True,
        buttons=buttons,
        bgcolor='#F5F5DC',
        pad=dict(l=50, r=50),  
        font=dict(
            family="Georgia", 
            size=14, 
            color="black"  
        )
    )],
    geo=dict(
        showframe=True,
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="LightBlue",
        showlakes=False,
        lakecolor="Blue",
        showcountries=True,
        countrycolor="gray",
        projection_type="orthographic",
        resolution=50,
        center=dict(lon=0, lat=20),
    ),
    width=800,
    height=600,
    showlegend=True
)

    fig.update_traces(
    hoverlabel=dict(
        bgcolor='rgba(255, 236, 210, 0.9)',
        font=dict(color='darkblue', size=16),
        bordercolor='white',
        align='auto'
    )
)

    fig.show()

