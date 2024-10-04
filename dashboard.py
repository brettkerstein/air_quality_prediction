import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import requests

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Air Quality Dashboard", className="mt-4 mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("Current AQI"),
            html.Div(id="current-aqi")
        ], width=6),
        dbc.Col([
            html.H2("AQI Prediction"),
            html.Div(id="aqi-prediction")
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("Historical AQI Data"),
            dcc.Graph(id="historical-graph")
        ])
    ])
], fluid=True)

@app.callback(
    Output("current-aqi", "children"),
    Input("interval-component", "n_intervals")
)
def update_current_aqi(_):
    response = requests.get("http://127.0.0.1:5000/api/current")
    data = response.json()
    return f"Current AQI: {data['value']:.2f} (as of {pd.to_datetime(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"

@app.callback(
    Output("aqi-prediction", "children"),
    Input("interval-component", "n_intervals")
)
def update_aqi_prediction(_):
    response = requests.get("http://127.0.0.1:5000/api/predict")
    data = response.json()
    return f"Predicted AQI: {data['value']:.2f} (for {pd.to_datetime(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"

@app.callback(
    Output("historical-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_historical_graph(_):
    response = requests.get("http://127.0.0.1:5000/api/historical")
    data = response.json()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['value'], mode='lines', name='AQI'))
    fig.update_layout(
        title="Historical AQI Data",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode="x unified"
    )
    return fig

def debug_request(url):
    try:
        response = requests.get(url)
        print(f"Request to {url} successful. Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {str(e)}")
        return None
    
# Add an interval component to refresh data
app.layout.children.append(dcc.Interval(
    id='interval-component',
    interval=60*1000,  # in milliseconds, update every 1 minute
    n_intervals=0
))

if __name__ == '__main__':
    data = debug_request("http://127.0.0.1:5000/api/current")
    app.run_server(debug=True, port=8050)