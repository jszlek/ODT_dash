import dash
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from run_h2o_server import my_model

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


# ----------------------------------------------- #
# ------------- SINGLE PAGE --------------------- #
# ----------------------------------------------- #

# Initial data for single prediction page
# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances["variable"],
                                         y=df_feature_importances["scaled_importance"],
                                         marker_color='rgb(171,226,251)')
                                  )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

# We record the name, min, mean and max of the three most important features
sliders_label = my_data.columns
sliders_min = np.floor(my_data.min())
sliders_mean = np.floor(my_data.mean())
sliders_max = np.floor(my_data.max())


# Layout for user info page
# Page wrapped in dcc.Loading widget to add a loading animation when the page is loaded/updated
single_page = dcc.Loading(
                   children=html.Div([
                           # Title display
                           dbc.Row([dbc.Col(html.Div(html.H2(children="ODT Simulation Tool")))],
                                   style={"height": "40px", "width": "100%", "margin-left": "auto", "margin-right": "auto", "margin-bottom": "5px", "margin-top": "5px"}),

                           # Dash Graph Component calls the fig_features_importance parameters
                           # We display the most important feature's name
                           dcc.Graph(figure=fig_features_importance),

                           html.Hr(style={'size': '100%', 'color': 'grey'}),
                           html.Div(
                            [

                               dbc.Row(                                         # First row
                                            [
                                                dbc.Col(html.Div([
                                                    html.P(
                                                        sliders_label[0],
                                                        style={'width': '100%', 'align': 'left'}
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            # The Dash Slider is built according to Feature #1 ranges
                                                            html.Div(
                                                                [
                                                                    dcc.Slider(id='X1_slider', min=sliders_min[0],
                                                                               max=sliders_max[0], step=1,
                                                                               value=sliders_mean[0],
                                                                               )
                                                                ],
                                                                style={'width': '60%', 'justify-content': 'left',
                                                                       'align-items': 'left'}
                                                            ),

                                                            dcc.Dropdown(
                                                                id='X1_slider_value',
                                                                options=[
                                                                    {'label': ' K ', 'value': 0},
                                                                    {'label': ' M ', 'value': 1}
                                                                ],
                                                                value=1,
                                                                style={'width': '30%', 'align': 'right'},
                                                                searchable=False
                                                                ),
                                                        ],
                                                    ),
                                                    html.Hr(style={'size': '33%', 'color': 'grey'}),
                                                    ]),
                                                    width={"size": 4, "order": 1},
                                                ),
                                                dbc.Col(
                                                    html.Div("The second of three columns"),
                                                    width={"size": 4, "order": 2},
                                                ),
                                                dbc.Col(
                                                    html.Div("The third of three columns"),
                                                    width={"size": 4, "order": 3},
                                                ),
                                            ]
                                        ),
                               dbc.Row(                                         # Second row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Third row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Fourth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Fifth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                          # Sixth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                            # Seventh row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Eighth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Ninth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Tenth row
                                    [
                                        dbc.Col(
                                            html.Div("The first of three columns"),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div("The second of three columns"),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div("The third of three columns"),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),
                            ]),

                           html.Div([
                               dbc.Row([


                                       html.P(
                                           sliders_label[1],
                                           style={'width': '100%', 'align': 'left'}
                                        ),

                                       html.Div([
                                           dbc.Col(
                                            [
                                                # The Dash Slider is built according to Feature #2 ranges
                                                html.Div(
                                                    [
                                                        dcc.Slider(id='X2_slider', min=sliders_min[1],
                                                                   max=sliders_max[1], step=0.1,
                                                                   value=sliders_mean[1],
                                                                   )
                                                    ],
                                                    style={'width': '80%', 'justify-content': 'center', 'align-items': 'center'}
                                                ),

                                                dcc.Input(
                                                    id='X2_slider_value',
                                                    type='number',
                                                    min=sliders_min[1],
                                                    max=sliders_max[1],
                                                    value=sliders_mean[1],
                                                    style={'width': '20%', 'align': 'right'}
                                                ),

                                            ]
                                           )
                                       ]),

                                       html.Hr(style={'size': '15%', 'color' : 'grey'}),

                                       html.P(
                                               sliders_label[2],
                                               style={'width': '100%', 'align': 'left'}
                                           ),
                                       html.Div([
                                           dbc.Col(
                                               [
                                                   # The Dash Slider is built according to Feature #3 ranges
                                                   html.Div(
                                                       [
                                                           dcc.Slider(id='X3_slider', min=sliders_min[2],
                                                                      max=sliders_max[2], step=0.1,
                                                                      value=sliders_mean[2],
                                                                      )
                                                       ],
                                                       style={'width': '80%', 'justify-content': 'center',
                                                              'align-items': 'center'}
                                                   ),

                                                   dcc.Input(
                                                       id='X3_slider_value',
                                                       type='number',
                                                       min=sliders_min[2],
                                                       max=sliders_max[2],
                                                       value=sliders_mean[2],
                                                       style={'width': '20%', 'align': 'right'}
                                                   ),

                                               ], width={"size": 4, "order": 3},
                                           )
                                       ]),

                                       html.Hr(style={'size': '15%', 'color': 'grey'}),

                                       ],
                                    style={'width': '33%', 'display': 'inline-block', 'align': 'left'},
                                       ),
                               dbc.Row([
                                   dbc.Col([html.Div(html.P('my next col'))])

                               ],
                                   style={'width': '33%', 'display': 'inline-block', 'margin': 'center'},
                               )
                               ]),
                           html.Div([

                           ],
                            style={'width': '33%', 'display': 'inline-block', 'margin': 'center'},
                           ),
                           # The prediction result will be displayed and updated here
                           html.Div([
                                html.P(),
                                html.H3(id="prediction_result")
                           ],
                               style={'width': '100%', 'display': 'inline-block'},
                           ),

                            ], className='prediction_result'),
                   )

batch_page = dcc.Loading(

    html.Div([
        html.Hr(),
        html.A('Batch mode, please provide comma separated file with header',
               style={
                   'width': '100%',
                   'height': '90px',
                   'lineHeight': '90px',
                   'textAlign': 'center',
               },
               ),
        html.Hr(),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files'),
                html.Hr(),
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Br(),
        html.P(),
        html.P(),
        html.P(),
        html.Div(id='output-data-upload'),
    ])
)