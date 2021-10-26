# Import libraries
import dash
import h2o
import base64
import datetime
import io
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from run_h2o_server import RunBeforeDash, my_model, open_browser
from layouts import single_page, batch_page
from sklearn import preprocessing
from dash.dependencies import Input, Output, State, ALL

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
    # Output - prediction
    Output(component_id="prediction_result", component_property="children"),

    Output({
        'type': 'X_slider',
        'id': ALL
    }, 'value'),

    Output({
        'type': 'X_slider_value',
        'id': ALL
    }, 'value'
    )

    # X?? slider outputs
    # Output(component_id="X1_slider", component_property="value"),
    # Output(component_id="X2_slider", component_property="value"),
    # Output(component_id="X3_slider", component_property="value"),
    #
    # # X?? textboxes outputs
    # Output(component_id='X1_slider_value', component_property='value'),
    # Output(component_id='X2_slider_value', component_property='value'),
    # Output(component_id='X3_slider_value', component_property='value')
],
    # The values corresponding to the three sliders are obtained by calling their id and value property

        # X?? Slider inputs
        Input({'type': 'X_slider', 'index': ALL}, 'value'),
        Input({'type': 'X_slider_value', 'index': ALL}, 'value'),
        # Input("X1_slider", "value"),
        # Input("X2_slider", "value"),
        # Input("X3_slider", "value"),
        # Input("X3_slider", "value"),
        # Input("X4_slider", "value"),
        # Input("X5_slider", "value"),
        # Input("X6_slider", "value"),
        # Input("X7_slider", "value"),
        # Input("X8_slider", "value"),
        # Input("X9_slider", "value"),
        # Input("X10_slider", "value"),
        # Input("X11_slider", "value"),
        # Input("X12_slider", "value"),
        # Input("X13_slider", "value"),
        # Input("X14_slider", "value"),
        # Input("X15_slider", "value"),
        # Input("X16_slider", "value"),
        # Input("X17_slider", "value"),
        # Input("X18_slider", "value"),
        # Input("X19_slider", "value"),
        # Input("X20_slider", "value"),
        # Input("X21_slider", "value"),
        # Input("X22_slider", "value"),
        # Input("X23_slider", "value"),
        # Input("X24_slider", "value"),
        # Input("X25_slider", "value"),
        # Input("X26_slider", "value"),
        # Input("X27_slider", "value"),
        # Input("X28_slider", "value"),
        # Input("X29_slider", "value"),
        # Input("X30_slider", "value"),
        # Input("X31_slider", "value"),
        # Input("X32_slider", "value"),
        # Input("X33_slider", "value"),
        # Input("X34_slider", "value"),
        # Input("X35_slider", "value"),
        # Input("X36_slider", "value"),
        # Input("X37_slider", "value"),
        # Input("X38_slider", "value"),
        # Input("X39_slider", "value"),
        #
        # # X?? Inputs in textboxes
        # Input('X1_slider_value', 'value'),
        # Input('X2_slider_value', 'value'),
        # Input('X3_slider_value', 'value'),
        # Input('X4_slider_value', 'value'),
        # Input('X5_slider_value', 'value'),
        # Input('X6_slider_value', 'value'),
        # Input('X7_slider_value', 'value'),
        # Input('X8_slider_value', 'value'),
        # Input('X9_slider_value', 'value'),
        # Input('X10_slider_value', 'value'),
        # Input('X11_slider_value', 'value'),
        # Input('X12_slider_value', 'value'),
        # Input('X13_slider_value', 'value'),
        # Input('X14_slider_value', 'value'),
        # Input('X15_slider_value', 'value'),
        # Input('X16_slider_value', 'value'),
        # Input('X17_slider_value', 'value'),
        # Input('X18_slider_value', 'value'),
        # Input('X19_slider_value', 'value'),
        # Input('X20_slider_value', 'value'),
        # Input('X21_slider_value', 'value'),
        # Input('X22_slider_value', 'value'),
        # Input('X23_slider_value', 'value'),
        # Input('X24_slider_value', 'value'),
        # Input('X25_slider_value', 'value'),
        # Input('X26_slider_value', 'value'),
        # Input('X27_slider_value', 'value'),
        # Input('X28_slider_value', 'value'),
        # Input('X29_slider_value', 'value'),
        # Input('X30_slider_value', 'value'),
        # Input('X31_slider_value', 'value'),
        # Input('X32_slider_value', 'value'),
        # Input('X33_slider_value', 'value'),
        # Input('X34_slider_value', 'value'),
        # Input('X35_slider_value', 'value'),
        # Input('X36_slider_value', 'value'),
        # Input('X37_slider_value', 'value'),
        # Input('X38_slider_value', 'value'),
        # Input('X39_slider_value', 'value'),
        #

    )
# The input variable are set in the same order as the callback Inputs
def update_prediction(X_slider, X_slider_value
        # X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20,
        #               X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39,
        #               X1_value, X2_value, X3_value, X4_value, X5_value, X6_value, X7_value, X8_value, X9_value,
        #               X10_value, X11_value, X12_value, X13_value, X14_value, X15_value, X16_value, X17_value, X18_value,
        #               X19_value, X20_value, X21_value, X22_value, X23_value, X24_value, X25_value, X26_value, X27_value,
        #               X28_value, X29_value, X30_value, X31_value, X32_value, X33_value, X34_value, X35_value, X36_value,
        #               X37_value, X38_value, X39_value
                      ):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "X_slider_value":
        X_slider = X_slider_value
    else:
        X_slider

    if trigger_id == "X_slider":
        X_slider_value = X_slider
    else:
        X_slider_value

    # if trigger_id == "X2_slider_value":
    #     X2 = X2_value
    # else:
    #     X2
    #
    # if trigger_id == "X2_slider":
    #     X2_value = X2
    # else:
    #     X2_value
    #
    # if trigger_id == "X3_slider_value":
    #     X3 = X3_value
    # else:
    #     X3
    #
    # if trigger_id == "X3_slider":
    #     X3_value = X3
    # else:
    #     X3_value

    # We create a NumPy array in the form of the original features
    # ["API_perc","Mannitol_perc",	"MCC_perc",	"Lactose_perc", (...)]
    input_X = my_data.copy()
    input_X = input_X.iloc[[0]]
    input_X.loc[:, 'API_perc'] = X_slider[0]
    input_X.loc[:, 'Mannitol_perc'] = X_slider[1]
    input_X.loc[:, 'MCC_perc'] = X_slider[2]
    input_X.loc[:, 'Lactose_perc'] = X_slider[3]
    input_X.loc[:, 'Calcium_silicate_perc'] = X_slider[4]
    input_X.loc[:, 'HPMC_perc'] = X_slider[5]
    input_X.loc[:, 'Sodium_bicarbonate_perc'] = X_slider[6]
    input_X.loc[:, 'SSG_perc'] = X_slider[7]
    input_X.loc[:, 'CC_Na_perc'] = X_slider[8]
    input_X.loc[:, 'Crospovidone_perc'] = X_slider[9]
    input_X.loc[:, 'L_HPC_perc'] = X_slider[10]
    input_X.loc[:, 'Pregelatinized_starch_perc'] = X_slider[11]
    input_X.loc[:, 'Sodium_carboxymethyl_starch_perc'] = X_slider[12]
    input_X.loc[:, 'Mg_stearate_perc'] = X_slider[13]
    input_X.loc[:, 'Aerosil_perc'] = X_slider[14]
    input_X.loc[:, 'Sodium_stearyl_fumarate_perc'] = X_slider[15]
    input_X.loc[:, 'Colloidal_silicon_dioxide_perc'] = X_slider[16]
    input_X.loc[:, 'Talc_perc'] = X_slider[17]
    input_X.loc[:, 'X2HP_bCD_perc'] = X_slider[18]
    input_X.loc[:, 'bCD_perc'] = X_slider[19]
    input_X.loc[:, 'CD_methacrylate_perc'] = X_slider[20]
    input_X.loc[:, 'Amberlite_IRP_64_69_perc'] = X_slider[21]
    input_X.loc[:, 'Eudragit_EPO_perc'] = X_slider[22]
    input_X.loc[:, 'Poloxamer_188_perc'] = X_slider[23]
    input_X.loc[:, 'PVP_perc'] = X_slider[24]
    input_X.loc[:, 'SLS_perc'] = X_slider[25]
    input_X.loc[:, 'PVA_perc'] = X_slider[26]
    input_X.loc[:, 'Camphor_perc'] = X_slider[27]
    input_X.loc[:, 'Hardness_N'] = X_slider[28]
    input_X.loc[:, 'GATS7i'] = X_slider[29]
    input_X.loc[:, 'Thickness_mm'] = X_slider[30]
    input_X.loc[:, 'GGI7'] = X_slider[31]
    input_X.loc[:, 'MATS4p'] = X_slider[32]
    input_X.loc[:, 'MIC2'] = X_slider[33]
    input_X.loc[:, 'Punch_mm'] = X_slider[34]
    input_X.loc[:, 'nT12Ring'] = X_slider[35]
    input_X.loc[:, 'XLogP'] = X_slider[36]
    input_X.loc[:, 'GATS7p'] = X_slider[37]
    input_X.loc[:, 'nF8HeteroRing'] = X_slider[38]


    input_X = pd.DataFrame(input_X)

    my_data_h2o = h2o.H2OFrame(input_X)

    # Prediction is calculated based on the input_X array
    prediction = my_model.predict(my_data_h2o)
    prediction = prediction.as_data_frame()
    prediction = float(prediction.iloc[0])



    # And retuned to the Output of the callback function
    return [
            "Prediction: {}".format(round(prediction, 1)),
            X_slider, X_slider_value
           ]


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    df = None

    decoded = base64.b64decode(content_string)
    try:
        if ('csv' or 'txt') in filename:
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
            df.insert(loc=0, column='predicted', value=predicted_values)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error during predicting'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True),

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


if __name__ == "__main__":
    # open browser
    open_browser()
    app.run_server(host='0.0.0.0', port=8050, debug=True)
