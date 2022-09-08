import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PowerRadar import PowerRadar
import numpy as np
from datetime import datetime
from dash.dependencies import Input, Output


radar = PowerRadar()

# for i in range(0,100):
#     pwr = radar.get_received_power()
#     avg_pwr = np.mean(pwr)
#     print(avg_pwr)


app = dash.Dash()   #initialising dash app
df = px.data.stocks() #reading stock price dataset 
radar_df = pd.DataFrame(columns=["Time", "Power"])

def collect_data():
    global radar, radar_df
    now = datetime.now()
    pwr = radar.get_received_power()
    #avg_pwr = np.mean(pwr)
    df_add = {"Time": str(now), "Power": pwr}

    radar_df = radar_df.append(df_add, ignore_index=True)
    print(radar_df)

@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def radar_power(n):
    # Function for creating line chart showing Google stock prices over time 
    collect_data()
    fig = go.Figure([go.Scatter(x = radar_df['Time'], y = radar_df['Power'],\
                     line = dict(color = 'firebrick', width = 4), name = 'Power')
                     ])
    fig.update_layout(title = 'Simple Radar Demo',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Power'
                      )
    return fig  

 




app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'SDRR Early Demo', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),

        
        dcc.Graph(id = 'live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ]
                     )


if __name__ == '__main__': 
    app.run_server()
    