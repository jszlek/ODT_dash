# Import libraries
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import string
import statistics
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psutil
import random
import math
import csv
import sys
from h2o.automl import H2OAutoML
import h2o
from pathlib import Path
from dash.dependencies import Input, Output, State
from configparser import ConfigParser

# Load config.ini file
config = ConfigParser(allow_no_value=True)
config.read('config.ini')

my_threads = int(config['DEFAULT']['my_threads'])
my_max_ram_allowed = int(config['DEFAULT']['my_max_ram_allowed'])


# -------------------------------
# Random key generator - function
# -------------------------------
def random_key_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# get current directory (PosixPath)
# -----------------------
my_current_dir = Path.cwd()

# check runtime mode - either many servers on the machine (server_multicore = F)
# or one server per one machine (server_multicore = T)
# -------------------------------------------
my_cores = my_threads

# check system free mem and apply it to the server
# ------------------------------------------------
memfree = psutil.virtual_memory().total
memfree_g = int(round(memfree / 1024 / 1024 / 1024 / my_cores))

if memfree_g < 2:
    memfree_g = 2

# Create three random strings
# -------------------------
aml_name = 'A' + random_key_generator(15)  # for FS project name
aml2_name = 'A' + random_key_generator(15)  # for classic approach project name
cluster_name = 'A' + random_key_generator(15)  # for h2o cluster name

# -------------------------------------
# run h2o server
# -------------------------------------
h2o.init(nthreads=my_threads,
         min_mem_size=memfree_g,
         max_mem_size=memfree_g,
         port=57750,
         ip='127.0.0.1',
         #         ice_root=str(my_export_dir),
         name=str(cluster_name),
         start_h2o=True)
# -------------------------------------

# --------------------------------------
# Load h2o model
# --------------------------------------
my_model = h2o.load_model('model/DeepLearning_grid__2_AutoML_20210831_185216_model_20')

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


# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances["variable"],
                                         y=df_feature_importances["scaled_importance"],
                                         marker_color='rgb(171,226,251)')
                                  )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

# We record the name, min, mean and max of the three most important features
slider_1_label = my_data.columns[0]
slider_1_min = math.floor(my_data[slider_1_label].min())
slider_1_mean = round(my_data[slider_1_label].mean())
slider_1_max = round(my_data[slider_1_label].max())

slider_2_label = my_data.columns[1]
slider_2_min = math.floor(my_data[slider_2_label].min())
slider_2_mean = round(my_data[slider_2_label].mean())
slider_2_max = round(my_data[slider_2_label].max())

slider_3_label = my_data.columns[2]
slider_3_min = math.floor(my_data[slider_3_label].min())
slider_3_mean = round(my_data[slider_3_label].mean())
slider_3_max = round(my_data[slider_3_label].max())

###############################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

sidebar = html.Div(
    style={'color': "#004f9c"},
    children=[
        html.Div(
            [
                # width: 3rem ensures the logo is the exact width of the
                # collapsed sidebar (accounting for padding)
                html.Img(src='assets/UJCM_logo.svg', style={"width": "3rem"}),
                html.H2("ODT calculator"),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), html.Span("Home")],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-calendar-alt me-2"),
                        html.Span("Calendar"),
                    ],
                    href="/calendar",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-envelope-open-text me-2"),
                        html.Span("Messages"),
                    ],
                    href="/messages",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

content = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                   children=[
                          # Title display
                          html.H1(children="ODT Simulation Tool"),

                          # Dash Graph Component calls the fig_features_importance parameters
                          dcc.Graph(figure=fig_features_importance),

                          # We display the most important feature's name
                          html.H6(
                                  children=[
                                      slider_1_label,
                                      html.Span(children=' ', id='updatemode-output-container1')
                                            ]
                          ),
                          html.Div(),

                          # The Dash Slider is built according to Feature #1 ranges
                          dcc.Slider(
                              id='X1_slider',
                              min=slider_1_min,
                              max=slider_1_max,
                              step=5,
                              value=slider_1_mean,
                              marks={i: '{} perc'.format(i) for i in
                                     np.linspace(slider_1_min, slider_1_max, 1 + (slider_1_max - slider_1_min)*1)}
                          ),

                          # The same logic is applied to the following names / sliders
                          html.H4(children=slider_2_label),

                          dcc.Slider(
                              id='X2_slider',
                              min=slider_2_min,
                              max=slider_2_max,
                              step=5,
                              value=slider_2_mean,
                              marks={i: '{} perc'.format(i) for i in
                                     np.linspace(slider_3_min, slider_2_max, 1 + (slider_2_max - slider_2_min)*1)}
                          ),

                          html.H4(children=slider_3_label),

                          dcc.Slider(
                              id='X3_slider',
                              min=slider_3_min,
                              max=slider_3_max,
                              step=1,
                              value=slider_3_mean,
                              marks={i: '{} perc'.format(i) for i in
                                     np.linspace(slider_3_min, slider_3_max, 1 + (slider_3_max - slider_3_min)*1)},
                          ),

                          # The prediction result will be displayed and updated here
                          html.H2(id="prediction_result"),

                      ])

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# set the content according to the current pathname
@app.callback(Output("content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the home page!")
    elif pathname == "/calendar":
        return html.P("This is your calendar... not much in the diary...")
    elif pathname == "/messages":
        return html.P("Here are all your messages")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# The callback function will provide one "Output" in the form of a string (=children)
@app.callback(  [
                    Output(component_id="prediction_result", component_property="children"),
                    Output(component_id="updatemode-output-container1", component_property="children")
                ],
              # The values correspnding to the three sliders are obtained by calling their id and value property
                [
                    Input("X1_slider", "value"),
                    Input("X2_slider", "value"),
                    Input("X3_slider", "value")
                ])
# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3):
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
    return "Prediction: {}".format(round(prediction, 1)), " {}".format(X1)


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)
