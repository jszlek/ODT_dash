#
# Copyrights, Jakub Szlęk, November 2021
# layouts.py is a part of ODT_dash application
# Repository is available at: https://github.com/jszlek/ODT_dash
# License: GNU GPLv3
#

from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import h2o
import pickle
import shap
import base64
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from run_h2o_server import my_model
from sklearn import preprocessing
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
try:
    if ("StackedEnsemble" in my_model.key) is False:
        extraced_varimp = my_model.varimp(use_pandas=True)
        df_feature_importances = pd.DataFrame(extraced_varimp[["scaled_importance", "variable"]])
    elif ("StackedEnsemble" in my_model.key) is True:

        # get the metalearner model
        meta = h2o.get_model(my_model.metalearner().model_id)

        # get varimp_df from metalearner
        if ('glm' in meta.algo) is True:
            varimp_df = pd.DataFrame.from_dict((meta.coef()), orient='index')
            varimp_df = varimp_df[1:]  # omit Intercept
        else:
            varimp_df = pd.DataFrame(meta.varimp())

        model_list = []

        for model in my_model.params['base_models']['actual']:
            model_list.append(model['name'])

        print(model_list)

        # create a dict for storing variable importance
        var_imp_models = dict([(key, []) for key in model_list])

        # get variable importance from base learners
        for model in model_list:
            tmp_model = h2o.get_model(str(model))

            # check if tmp_model has varimp()
            if tmp_model.varimp() is None:
                print(str(model))
                del var_imp_models[str(model)]
            else:
                # check if tmp_model is glm - it has no varimp() but coef()
                if ('glm' in tmp_model.algo) is True:
                    tmp_var_imp = pd.DataFrame.from_dict(tmp_model.coef(), orient='index').rename(
                        columns={0: 'scaled_importance'})
                    tmp_var_imp = tmp_var_imp[1:]  # omit Intercept
                    tmp_var_imp.insert(loc=0, column='variable',
                                       value=tmp_var_imp.index)  # reset index of rows into column
                else:
                    tmp_var_imp = tmp_model.varimp(use_pandas=True).iloc[:, [0, 2]]

                var_imp_models[str(model)].append(tmp_var_imp)
                meta_scale = varimp_df
                for idx in meta_scale.iterrows():
                    if ('glm' in meta.algo) is True:
                        var_imp_models[str(idx[0])][0]['scaled_importance'] = var_imp_models[str(idx[0])][0].values[0:,
                                                                              1] * float(idx[1])
                    else:
                        var_imp_models[str(idx[1][0])][0]['scaled_importance'] = var_imp_models[str(idx[1][0])][0][
                                                                                     'scaled_importance'] * idx[1][3]

            # new dataframe init
            scaled_var_imp_df = pd.DataFrame()

            for idx in var_imp_models.keys():
                df_tmp = var_imp_models[str(idx)][0]['scaled_importance']
                df_tmp.index = var_imp_models[str(idx)][0]['variable']
                scaled_var_imp_df = pd.concat([scaled_var_imp_df, df_tmp], axis=1, sort=False)

            # sum rows by index, NaNs are considered as zeros
            # Total sum per row:
            scaled_var_imp_df.loc[:, 'Total'] = scaled_var_imp_df.sum(axis=1)

            # scale column 'Total' from 0 to 1
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaled_var_imp_df.loc[:, 'Total'] = min_max_scaler.fit_transform(
                scaled_var_imp_df.loc[:, 'Total'].values.reshape(-1, 1))

            # Sort by 'Total' values
            scaled_var_imp_df_sorted = scaled_var_imp_df.sort_values(by=['Total'], ascending=False)

            # Make additional column with original column indicies
            orig_column_list = list()

            for i in scaled_var_imp_df_sorted.index:
                orig_column_list.append(my_data.columns.get_loc(i) + 1)

            # orig_column_list = [(data.columns.get_loc(i)+1) for i in scaled_var_imp_df_sorted.index]
            scaled_var_imp_df_sorted['Orig column'] = orig_column_list

        df_feature_importances = scaled_var_imp_df_sorted

except Exception as e:
    print(e)
    html.Div([
        'There was an error during feature extraction'
    ])


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


f = open('data/shap_explainer_expected_value.dat', 'rb')
shap_explainer_expected_value = pickle.load(f)
f.close()

f = open('data/shap_values.dat', 'rb')
shap_values = pickle.load(f)
f.close()

f = open('data/data_features.dat', 'rb')
shap_data_features = pickle.load(f)
f.close()

def _force_plot_html(*args):
    force_plot = shap.force_plot(*args, matplotlib=False, figsize=(8, 4))
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"height": "450px", "border": 0, 'width': '100%', 'align': 'center'})

def _summary_plot(*args):
    image_filename = 'assets/summary_plot.png'  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    style={"height": "100%", "border": 0, 'width': '100%'})


# Layout for user info page
# Page wrapped in dcc.Loading widget to add a loading animation when the page is loaded/updated
single_page = dcc.Loading(children=
                   html.Div([
                           # Title display
                           dbc.Row([
                               dbc.Col(html.Div(html.H2("ODT Simulation Tool")), style={'width': '100%'}),
                                ],
                                   style={"height": "40px", "width": "100%", "margin-left": "auto", "margin-right": "auto", "margin-bottom": "5px", "margin-top": "5px"}),


                           # Dash Graph Component calls the fig_features_importance parameters
                           # We display the most important feature's name
                           dbc.Row([
                               dbc.Col(html.Div([dcc.Graph(figure=fig_features_importance)]), width={"size": 4, "order": 1}, className="card"),
                               dbc.Col(html.Div(
                                   [
                                       _force_plot_html(shap_explainer_expected_value, shap_values, shap_data_features)
                                   ]
                               ), width={"size": 4, "order": 2}, className="card"),
                               dbc.Col(html.Div(
                                   [
                                       _summary_plot()
                                   ], style={"height": "100%", "border": 0, 'width': '100%', 'align': 'center'}
                               ), width={"size": 4, "order": 3}, className="card")

                           ], style={'width': '100%', 'align': 'center'}),

                           html.Hr(style={'size': '100%', 'color': 'grey'}),

                           # The prediction result will be displayed and updated here
                           html.Div([
                               html.P(),
                               html.H3(id="prediction_result")
                           ],
                               style={'width': '100%', 'display': 'inline-block'},
                           ),

                           html.Hr(style={'size': '100%', 'color': 'grey'}),

                           html.Div([

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
                                                                    dcc.Slider(id={'type': 'X_slider', 'index': 0},
                                                                               min=sliders_min[0],
                                                                               max=sliders_max[0], step=0.1,
                                                                               value=sliders_mean[0],
                                                                               )
                                                                ],
                                                                style={'width': '70%', 'justify-content': 'left',
                                                                       'align-items': 'left'}
                                                            ),

                                                            dcc.Input(
                                                                id={'type': 'X_slider_value', 'index': 0},
                                                                type='number',
                                                                min=sliders_min[0],
                                                                max=sliders_max[0],
                                                                value=sliders_mean[0],
                                                                style={'width': '20%', 'align': 'right'}
                                                            )
                                                        ],
                                                    ),
                                                    html.Hr(style={'size': '33%', 'color': 'grey'}),
                                                    ]),
                                                    width={"size": 4, "order": 1},
                                                ),
                                                dbc.Col(
                                                    html.Div([
                                                        html.P(
                                                            sliders_label[1],
                                                            style={'width': '100%', 'align': 'left'}
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                # The Dash Slider is built according to Feature #2 ranges
                                                                html.Div(
                                                                    [
                                                                        dcc.Slider(id={'type': 'X_slider', 'index': 1},
                                                                                   min=sliders_min[1],
                                                                                   max=sliders_max[1], step=0.1,
                                                                                   value=sliders_mean[1],
                                                                                   )
                                                                    ],
                                                                    style={'width': '70%', 'justify-content': 'left',
                                                                           'align-items': 'left'}
                                                                ),

                                                                dcc.Input(
                                                                    id={'type': 'X_slider_value', 'index': 1},
                                                                    type='number',
                                                                    min=sliders_min[1],
                                                                    max=sliders_max[1],
                                                                    value=sliders_mean[1],
                                                                    style={'width': '20%', 'align': 'right'}
                                                                ),
                                                            ],
                                                        ),
                                                        html.Hr(style={'size': '33%', 'color': 'grey'}),
                                                    ]),
                                                    width={"size": 4, "order": 2},
                                                ),
                                                dbc.Col(html.Div([
                                                        html.P(
                                                            sliders_label[2],
                                                            style={'width': '100%', 'align': 'left'}
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                # The Dash Slider is built according to Feature #3 ranges
                                                                html.Div(
                                                                    [
                                                                        dcc.Slider(id={'type': 'X_slider', 'index': 2},
                                                                                   min=sliders_min[2],
                                                                                   max=sliders_max[2], step=0.1,
                                                                                   value=sliders_mean[2],
                                                                                   )
                                                                    ],
                                                                    style={'width': '70%', 'justify-content': 'left',
                                                                           'align-items': 'left'}
                                                                ),

                                                                dcc.Input(
                                                                    id={'type': 'X_slider_value', 'index': 2},
                                                                    type='number',
                                                                    min=sliders_min[2],
                                                                    max=sliders_max[2],
                                                                    value=sliders_mean[2],
                                                                    style={'width': '20%', 'align': 'right'}
                                                                ),
                                                            ],
                                                        ),
                                                        html.Hr(style={'size': '33%', 'color': 'grey'}),
                                                    ]),

                                                    width={"size": 4, "order": 3},

                                                ),
                                            ],
                                                className="mb-4"
                                        ),
                               dbc.Row(                                         # Second row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[3],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #4 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 3},
                                                                           min=sliders_min[3],
                                                                           max=sliders_max[3], step=0.1,
                                                                           value=sliders_mean[3],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 3},
                                                            type='number',
                                                            min=sliders_min[3],
                                                            max=sliders_max[3],
                                                            value=sliders_mean[3],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[4],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #5 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 4},
                                                                           min=sliders_min[4],
                                                                           max=sliders_max[4], step=0.1,
                                                                           value=sliders_mean[4],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 4},
                                                            type='number',
                                                            min=sliders_min[4],
                                                            max=sliders_max[4],
                                                            value=sliders_mean[4],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[5],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #1 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 5},
                                                                           min=sliders_min[5],
                                                                           max=sliders_max[5], step=0.1,
                                                                           value=sliders_mean[5],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 5},
                                                            type='number',
                                                            min=sliders_min[5],
                                                            max=sliders_max[5],
                                                            value=sliders_mean[5],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Third row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[6],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #1 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 6},
                                                                           min=sliders_min[6],
                                                                           max=sliders_max[6], step=0.1,
                                                                           value=sliders_mean[6],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 6},
                                                            type='number',
                                                            min=sliders_min[6],
                                                            max=sliders_max[6],
                                                            value=sliders_mean[6],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[7],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #8 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 7},
                                                                           min=sliders_min[7],
                                                                           max=sliders_max[7], step=0.1,
                                                                           value=sliders_mean[7],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 7},
                                                            type='number',
                                                            min=sliders_min[7],
                                                            max=sliders_max[7],
                                                            value=sliders_mean[7],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[8],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #8 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 8},
                                                                           min=sliders_min[8],
                                                                           max=sliders_max[8], step=0.1,
                                                                           value=sliders_mean[8],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 8},
                                                            type='number',
                                                            min=sliders_min[8],
                                                            max=sliders_max[8],
                                                            value=sliders_mean[8],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Fourth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[9],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #10 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 9},
                                                                           min=sliders_min[9],
                                                                           max=sliders_max[9], step=0.1,
                                                                           value=sliders_mean[9],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 9},
                                                            type='number',
                                                            min=sliders_min[9],
                                                            max=sliders_max[9],
                                                            value=sliders_mean[9],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[10],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #11 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 10},
                                                                           min=sliders_min[10],
                                                                           max=sliders_max[10], step=0.1,
                                                                           value=sliders_mean[10],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 10},
                                                            type='number',
                                                            min=sliders_min[10],
                                                            max=sliders_max[10],
                                                            value=sliders_mean[10],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[11],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #12 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 11},
                                                                           min=sliders_min[11],
                                                                           max=sliders_max[11], step=0.1,
                                                                           value=sliders_mean[11],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 11},
                                                            type='number',
                                                            min=sliders_min[11],
                                                            max=sliders_max[11],
                                                            value=sliders_mean[11],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                         # Fifth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[12],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #13 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 12},
                                                                           min=sliders_min[12],
                                                                           max=sliders_max[12], step=0.1,
                                                                           value=sliders_mean[12],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 12},
                                                            type='number',
                                                            min=sliders_min[12],
                                                            max=sliders_max[12],
                                                            value=sliders_mean[12],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[13],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #14 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 13},
                                                                           min=sliders_min[13],
                                                                           max=sliders_max[13], step=0.1,
                                                                           value=sliders_mean[13],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 13},
                                                            type='number',
                                                            min=sliders_min[13],
                                                            max=sliders_max[13],
                                                            value=sliders_mean[13],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[14],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #15 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 14},
                                                                           min=sliders_min[14],
                                                                           max=sliders_max[14], step=0.1,
                                                                           value=sliders_mean[14],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 14},
                                                            type='number',
                                                            min=sliders_min[14],
                                                            max=sliders_max[14],
                                                            value=sliders_mean[14],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                          # Sixth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[15],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #16 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 15},
                                                                           min=sliders_min[15],
                                                                           max=sliders_max[15], step=0.1,
                                                                           value=sliders_mean[15],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 15},
                                                            type='number',
                                                            min=sliders_min[15],
                                                            max=sliders_max[15],
                                                            value=sliders_mean[15],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[16],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #17 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 16},
                                                                           min=sliders_min[16],
                                                                           max=sliders_max[16], step=0.1,
                                                                           value=sliders_mean[16],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 16},
                                                            type='number',
                                                            min=sliders_min[16],
                                                            max=sliders_max[16],
                                                            value=sliders_mean[16],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[17],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #18 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 17},
                                                                           min=sliders_min[17],
                                                                           max=sliders_max[17], step=0.1,
                                                                           value=sliders_mean[17],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 17},
                                                            type='number',
                                                            min=sliders_min[17],
                                                            max=sliders_max[17],
                                                            value=sliders_mean[17],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                            # Seventh row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[18],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 18},
                                                                           min=sliders_min[18],
                                                                           max=sliders_max[18], step=0.1,
                                                                           value=sliders_mean[18],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 18},
                                                            type='number',
                                                            min=sliders_min[18],
                                                            max=sliders_max[18],
                                                            value=sliders_mean[18],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[19],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #20 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 19},
                                                                           min=sliders_min[19],
                                                                           max=sliders_max[19], step=0.1,
                                                                           value=sliders_mean[19],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 19},
                                                            type='number',
                                                            min=sliders_min[19],
                                                            max=sliders_max[19],
                                                            value=sliders_mean[19],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[20],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 20},
                                                                           min=sliders_min[20],
                                                                           max=sliders_max[20], step=0.1,
                                                                           value=sliders_mean[20],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 20},
                                                            type='number',
                                                            min=sliders_min[20],
                                                            max=sliders_max[20],
                                                            value=sliders_mean[20],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Eighth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[21],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 21},
                                                                           min=sliders_min[21],
                                                                           max=sliders_max[21], step=0.1,
                                                                           value=sliders_mean[21],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 21},
                                                            type='number',
                                                            min=sliders_min[21],
                                                            max=sliders_max[21],
                                                            value=sliders_mean[21],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[22],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 22},
                                                                           min=sliders_min[22],
                                                                           max=sliders_max[22], step=0.1,
                                                                           value=sliders_mean[22],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 22},
                                                            type='number',
                                                            min=sliders_min[22],
                                                            max=sliders_max[22],
                                                            value=sliders_mean[22],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[23],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 23},
                                                                           min=sliders_min[23],
                                                                           max=sliders_max[23], step=0.1,
                                                                           value=sliders_mean[23],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 23},
                                                            type='number',
                                                            min=sliders_min[23],
                                                            max=sliders_max[23],
                                                            value=sliders_mean[23],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Ninth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[24],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 24},
                                                                           min=sliders_min[24],
                                                                           max=sliders_max[24], step=0.1,
                                                                           value=sliders_mean[24],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 24},
                                                            type='number',
                                                            min=sliders_min[24],
                                                            max=sliders_max[24],
                                                            value=sliders_mean[24],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[25],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 25},
                                                                           min=sliders_min[25],
                                                                           max=sliders_max[25], step=0.1,
                                                                           value=sliders_mean[25],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 25},
                                                            type='number',
                                                            min=sliders_min[25],
                                                            max=sliders_max[25],
                                                            value=sliders_mean[25],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[26],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 26},
                                                                           min=sliders_min[26],
                                                                           max=sliders_max[26], step=0.1,
                                                                           value=sliders_mean[26],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 26},
                                                            type='number',
                                                            min=sliders_min[26],
                                                            max=sliders_max[26],
                                                            value=sliders_mean[26],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(                                             # Tenth row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[27],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 27},
                                                                           min=sliders_min[27],
                                                                           max=sliders_max[27], step=0.1,
                                                                           value=sliders_mean[27],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 27},
                                                            type='number',
                                                            min=sliders_min[27],
                                                            max=sliders_max[27],
                                                            value=sliders_mean[27],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[28],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 28},
                                                                           min=sliders_min[28],
                                                                           max=sliders_max[28], step=0.1,
                                                                           value=sliders_mean[28],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 30},
                                                            type='number',
                                                            min=sliders_min[28],
                                                            max=sliders_max[28],
                                                            value=sliders_mean[28],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[29],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 29},
                                                                           min=sliders_min[29],
                                                                           max=sliders_max[29], step=0.1,
                                                                           value=sliders_mean[29],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 29},
                                                            type='number',
                                                            min=sliders_min[29],
                                                            max=sliders_max[29],
                                                            value=sliders_mean[29],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(  # Eleventh row
                                    [
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[30],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 30},
                                                                           min=sliders_min[30],
                                                                           max=sliders_max[30], step=0.1,
                                                                           value=sliders_mean[30],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 30},
                                                            type='number',
                                                            min=sliders_min[30],
                                                            max=sliders_max[30],
                                                            value=sliders_mean[30],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 1},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[31],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 31},
                                                                           min=sliders_min[31],
                                                                           max=sliders_max[31], step=0.1,
                                                                           value=sliders_mean[31],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 31},
                                                            type='number',
                                                            min=sliders_min[31],
                                                            max=sliders_max[31],
                                                            value=sliders_mean[31],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 2},
                                        ),
                                        dbc.Col(
                                            html.Div([
                                                html.P(
                                                    sliders_label[32],
                                                    style={'width': '100%', 'align': 'left'}
                                                ),
                                                dbc.Row(
                                                    [
                                                        # The Dash Slider is built according to Feature #19 ranges
                                                        html.Div(
                                                            [
                                                                dcc.Slider(id={'type': 'X_slider', 'index': 32},
                                                                           min=sliders_min[32],
                                                                           max=sliders_max[32], step=0.1,
                                                                           value=sliders_mean[32],
                                                                           )
                                                            ],
                                                            style={'width': '70%', 'justify-content': 'left',
                                                                   'align-items': 'left'}
                                                        ),

                                                        dcc.Input(
                                                            id={'type': 'X_slider_value', 'index': 32},
                                                            type='number',
                                                            min=sliders_min[32],
                                                            max=sliders_max[32],
                                                            value=sliders_mean[32],
                                                            style={'width': '20%', 'align': 'right'}
                                                        ),
                                                    ],
                                                ),
                                                html.Hr(style={'size': '33%', 'color': 'grey'}),
                                            ]),
                                            width={"size": 4, "order": 3},
                                        ),
                                    ]
                                ),

                               dbc.Row(  # Twelveth row
                                   [
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[33],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 33},
                                                                          min=sliders_min[33],
                                                                          max=sliders_max[33], step=0.1,
                                                                          value=sliders_mean[33],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 33},
                                                           type='number',
                                                           min=sliders_min[33],
                                                           max=sliders_max[33],
                                                           value=sliders_mean[33],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 1},
                                       ),
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[34],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 34},
                                                                          min=sliders_min[34],
                                                                          max=sliders_max[34], step=0.1,
                                                                          value=sliders_mean[34],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 34},
                                                           type='number',
                                                           min=sliders_min[34],
                                                           max=sliders_max[34],
                                                           value=sliders_mean[34],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 2},
                                       ),
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[35],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 35},
                                                                          min=sliders_min[35],
                                                                          max=sliders_max[35], step=0.1,
                                                                          value=sliders_mean[35],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 35},
                                                           type='number',
                                                           min=sliders_min[35],
                                                           max=sliders_max[35],
                                                           value=sliders_mean[35],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 3},
                                       ),
                                   ]
                               ),

                               dbc.Row(  # Therteenth row
                                   [
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[36],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 36},
                                                                          min=sliders_min[36],
                                                                          max=sliders_max[36], step=0.1,
                                                                          value=sliders_mean[36],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 36},
                                                           type='number',
                                                           min=sliders_min[36],
                                                           max=sliders_max[36],
                                                           value=sliders_mean[36],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 1},
                                       ),
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[37],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 37},
                                                                          min=sliders_min[37],
                                                                          max=sliders_max[37], step=0.1,
                                                                          value=sliders_mean[37],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 37},
                                                           type='number',
                                                           min=sliders_min[37],
                                                           max=sliders_max[37],
                                                           value=sliders_mean[37],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 2},
                                       ),
                                       dbc.Col(
                                           html.Div([
                                               html.P(
                                                   sliders_label[38],
                                                   style={'width': '100%', 'align': 'left'}
                                               ),
                                               dbc.Row(
                                                   [
                                                       # The Dash Slider is built according to Feature #19 ranges
                                                       html.Div(
                                                           [
                                                               dcc.Slider(id={'type': 'X_slider', 'index': 38},
                                                                          min=sliders_min[38],
                                                                          max=sliders_max[38], step=0.1,
                                                                          value=sliders_mean[38],
                                                                          )
                                                           ],
                                                           style={'width': '70%', 'justify-content': 'left',
                                                                  'align-items': 'left'}
                                                       ),

                                                       dcc.Input(
                                                           id={'type': 'X_slider_value', 'index': 38},
                                                           type='number',
                                                           min=sliders_min[38],
                                                           max=sliders_max[38],
                                                           value=sliders_mean[38],
                                                           style={'width': '20%', 'align': 'right'}
                                                       ),
                                                   ],
                                               ),
                                               html.Hr(style={'size': '33%', 'color': 'grey'}),
                                           ]),
                                           width={"size": 4, "order": 3},
                                       ),
                                   ]
                               ),

                           ]),
                           ])
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

help_page = dcc.Loading(

    html.Div([
        html.Hr(),
        html.A('Help page is landing here',
               style={
                   'width': '100%',
                   'height': '90px',
                   'lineHeight': '90px',
                   'textAlign': 'center',
               },
               ),
        html.Hr(),
        html.Div(id='help-page'),
    ])
)

about_page = dcc.Loading(

    html.Div([
        html.Hr(),
        dcc.Markdown('''
        # About  
        ### License GPLv3  
        ### Copyright: Jakub Szlęk, 2021  
        ## Disclaimer  
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
    
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
        
        '''),
        html.Hr(),
        html.P("License"),
        html.Div(id='about-page'),
    ])
)