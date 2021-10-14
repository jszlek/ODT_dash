# Import libraries
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import string
import statistics
import pandas as pd
import numpy as np
# import plotly.graph_objects as go
import run_h2o_server
from run_h2o_server import my_model, open_browser
import h2o
from layouts import single_page, batch_page

import psutil
import random
import math
import csv
import sys
import time
from h2o.automl import H2OAutoML

from dash.dependencies import Input, Output, State

import base64
import datetime
import io

# --------------------------------------
# Load test data
# --------------------------------------
my_data = pd.read_csv('data/ODT_Disintegration_Time_db_expanded_raw_v17_1.csv', sep='\t', index_col=0)

extracted_column_names = ["API_perc", "Mannitol_perc",	"MCC_perc",	"Lactose_perc",	"Calcium_silicate_perc",
                          "HPMC_perc",	"Sodium_bicarbonate_perc",	"SSG_perc",	"CC_Na_perc",
                          "Crospovidone_perc",	"L_HPC_perc",	"Pregelatinized_starch_perc",
                          "Sodium_carboxymethyl_starch_perc",	"Mg_stearate_perc",	"Aerosil_perc",
                          "Sodium_stearyl_fumarate_perc",	"Colloidal_silicon_dioxide_perc",	"Talc_perc",
                          "X2HP_bCD_perc",	"bCD_perc",	"CD_methacrylate_perc",	"Amberlite_IRP_64_69_perc",
                          "Eudragit_EPO_perc",	"Poloxamer_188_perc",	"PVP_perc",	"SLS_perc",	"PVA_perc",
                          "Camphor_perc",	"Hardness_N",	"GATS7i",	"Thickness_mm",	"GGI7",	"MATS4p",
                          "MIC2",	"Punch_mm",	"nT12Ring",	"XLogP",	"GATS7p",	"nF8HeteroRing",
                          "Disintegration_time_sec"]
my_data = my_data[extracted_column_names]

# Get the features' importance and their corresponding label
extraced_varimp = my_model.varimp(use_pandas=True)
df_feature_importances = pd.DataFrame(extraced_varimp[["scaled_importance", "variable"]])


# ------------------------------------------------- #
# -------------------- NAV BAR -------------------- #
# ------------------------------------------------- #
ODT_LOGO = "/assets/ODT_calc_icon.svg"

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search", id="search_bar")),
        dbc.Col(
            dbc.Button("Search", color="light", className="ml-2", id="search_bar_button"),
            width="auto",
        ),
    ],
    style={"margin-right": "3rem"},
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=ODT_LOGO, height="30px")),
                    dbc.Col(html.P()),
                    dbc.Col(dbc.NavbarBrand("ODT calculator", className="ml-2")),
                ], style={"margin-left": "11%"},
                align="center",
                no_gutters=True,
            ),
            href="https://www.uj-cm.krakow.edu.pl",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color='primary',
    sticky="top",
    style={'width': 'calc(100% - 12rem)', 'float': 'right', 'height': '4.5rem'}
)

# ------------------------------------------------- #
# -------------------- SIDEBAR -------------------- #
# ------------------------------------------------- #


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
    'text-align': 'center'
}

links = {
    "Single prediction": ["/single-prediction", "single-prediction-link"],
    "Batch mode": ["/batch-mode", "batch-mode-link"]
}

sidebar = html.Div(
    [
        html.H6("Menu", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [dbc.NavLink(x, href=links[x][0], id=links[x][1]) for x in links.keys()],
            vertical=True,
            pills=True
        ),
    ],
    style=SIDEBAR_STYLE,
)


# ----------------------------------------------- #
# ----------------- CONTENT STYLE --------------- #
# ----------------------------------------------- #
# CSS styling for main-page contents (user page and transaction page)
CONTENT_STYLE = {
    "margin-left": "13rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
                        dcc.Location(id="url", refresh=False),
                        sidebar,
                        navbar,
                        html.Div(id='page-content', style=CONTENT_STYLE)
])


# ------------------------------------- #
# ------------- CALLBACKS ------------- #
# ------------------------------------- #


# Switch pathname for url
@app.callback(
    [Output(f"{links[id][1]}", "active") for id in links.keys()],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/" or pathname == "//":
        # Treat page 1 as the homepage / index
        return True, False
    return [pathname == f"{links[id][0]}" for id in links.keys()]

# Set page for each pathname
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname in ["/", "//", f"{links[list(links.keys())[0]][0]}"]:
        return single_page
    else:
        return batch_page
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# The callback function will provide one "Output" in the form of a string (=children)
@app.callback([
    Output(component_id="prediction_result", component_property="children"),
    Output(component_id="X1_slider", component_property="value"),
    Output(component_id="X2_slider", component_property="value"),
    Output(component_id="X3_slider", component_property="value"),
    Output(component_id='X1_slider_value', component_property='value'),
    Output(component_id='X2_slider_value', component_property='value'),
    Output(component_id='X3_slider_value', component_property='value')
],
    # The values corresponding to the three sliders are obtained by calling their id and value property
    [
        Input("X1_slider", "value"),
        Input("X2_slider", "value"),
        Input("X3_slider", "value"),
        Input('X1_slider_value', 'value'),
        Input('X2_slider_value', 'value'),
        Input('X3_slider_value', 'value')
    ])
# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3, X1_value, X2_value, X3_value):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "X1_slider_value":
        X1 = X1_value
    else:
        X1

    if trigger_id == "X1_slider":
        X1_value = X1
    else:
        X1_value

    if trigger_id == "X2_slider_value":
        X2 = X2_value
    else:
        X2

    if trigger_id == "X2_slider":
        X2_value = X2
    else:
        X2_value

    if trigger_id == "X3_slider_value":
        X3 = X3_value
    else:
        X3

    if trigger_id == "X3_slider":
        X3_value = X3
    else:
        X3_value


    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = my_data.copy()
    input_X = input_X.iloc[[0]]
    input_X.loc[:, 'API_perc'] = X1
    input_X.loc[:, 'Mannitol_perc'] = X2
    input_X.loc[:, 'MCC_perc'] = X3

    input_X = pd.DataFrame(input_X)

    my_data_h2o = h2o.H2OFrame(input_X)

    # Prediction is calculated based on the input_X array
    prediction = my_model.predict(my_data_h2o)
    prediction = prediction.as_data_frame()
    prediction = float(prediction.iloc[0])



    # And retuned to the Output of the callback function
    return [
            "Prediction: {}".format(round(prediction, 1)),
            X1,
            X2,
            X3,
            X1_value,
            X2_value,
            X3_value,
           ]


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    try:
        if df is not None:
            # make predictions
            h2o_df = h2o.H2OFrame(df)
            predicted_values = my_model.predict(h2o_df)
            predicted_values = predicted_values.as_data_frame()
            df['predicted'] = predicted_values

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error during predicting'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),



        dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True),
#            (
#            data=predicted_table.to_dict('records'),
#            columns=[{'name': i, 'id': i} for i in predicted_table.columns]
#        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# open browser
open_browser()


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)