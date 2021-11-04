# Import libraries
import dash
import h2o
import base64
import datetime
import io
import webbrowser
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from run_h2o_server import my_model
from layouts import single_page, batch_page, my_data
from dash.dependencies import Input, Output, State, ALL

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
@app.callback(
    [
    Output({'type': 'X_slider', 'index': ALL}, 'value'),
    Output({'type': 'X_slider_value', 'index': ALL}, 'value'),
    # Output - prediction
    Output(component_id="prediction_result", component_property="children")
    ],
    Input({'type': 'X_slider', 'index': ALL}, 'value'),
    Input({'type': 'X_slider_value', 'index': ALL}, 'value'),
    )
# The input variable are set in the same order as the callback Inputs
def update_sliders(X_slider, X_slider_value):

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # if trigger_id == "X_slider_value":
    #     X_slider = X_slider_value
    # else:
    #     X_slider_value = X_slider
    #
    # if trigger_id == "X_slider":
    #     X_slider_value = X_slider
    # else:
    #     X_slider = X_slider_value

    value = X_slider_value if "X_slider_value" in trigger_id else X_slider

    # We create a NumPy array in the form of the original features
    # ["API_perc","Mannitol_perc",	"MCC_perc",	"Lactose_perc", (...)]
    input_X = my_data.copy()
    input_X = input_X.iloc[[0]]
    input_X.loc[:, 'API_perc'] = value[0]
    input_X.loc[:, 'Mannitol_perc'] = value[1]
    input_X.loc[:, 'MCC_perc'] = value[2]
    input_X.loc[:, 'Lactose_perc'] = value[3]
    input_X.loc[:, 'Calcium_silicate_perc'] = value[4]
    input_X.loc[:, 'HPMC_perc'] = value[5]
    input_X.loc[:, 'Sodium_bicarbonate_perc'] = value[6]
    input_X.loc[:, 'SSG_perc'] = value[7]
    input_X.loc[:, 'CC_Na_perc'] = value[8]
    input_X.loc[:, 'Crospovidone_perc'] = value[9]
    input_X.loc[:, 'L_HPC_perc'] = value[10]
    input_X.loc[:, 'Pregelatinized_starch_perc'] = value[11]
    input_X.loc[:, 'Sodium_carboxymethyl_starch_perc'] = value[12]
    input_X.loc[:, 'Mg_stearate_perc'] = value[13]
    input_X.loc[:, 'Aerosil_perc'] = value[14]
    input_X.loc[:, 'Sodium_stearyl_fumarate_perc'] = value[15]
    input_X.loc[:, 'Colloidal_silicon_dioxide_perc'] = value[16]
    input_X.loc[:, 'Talc_perc'] = value[17]
    input_X.loc[:, 'X2HP_bCD_perc'] = value[18]
    input_X.loc[:, 'bCD_perc'] = value[19]
    input_X.loc[:, 'CD_methacrylate_perc'] = value[20]
    input_X.loc[:, 'Amberlite_IRP_64_69_perc'] = value[21]
    input_X.loc[:, 'Eudragit_EPO_perc'] = value[22]
    input_X.loc[:, 'Poloxamer_188_perc'] = value[23]
    input_X.loc[:, 'PVP_perc'] = value[24]
    input_X.loc[:, 'SLS_perc'] = value[25]
    input_X.loc[:, 'PVA_perc'] = value[26]
    input_X.loc[:, 'Camphor_perc'] = value[27]
    input_X.loc[:, 'Hardness_N'] = value[28]
    input_X.loc[:, 'GATS7i'] = value[29]
    input_X.loc[:, 'Thickness_mm'] = value[30]
    input_X.loc[:, 'GGI7'] = value[31]
    input_X.loc[:, 'MATS4p'] = value[32]
    input_X.loc[:, 'MIC2'] = value[33]
    input_X.loc[:, 'Punch_mm'] = value[34]
    input_X.loc[:, 'nT12Ring'] = value[35]
    input_X.loc[:, 'XLogP'] = value[36]
    input_X.loc[:, 'GATS7p'] = value[37]
    input_X.loc[:, 'nF8HeteroRing'] = value[38]


    input_X = pd.DataFrame(input_X)
    input_X = input_X.astype(float)

    my_data_h2o = h2o.H2OFrame(input_X)

    # Prediction is calculated based on the input_X array
    prediction = my_model.predict(my_data_h2o)
    prediction = prediction.as_data_frame()
    prediction = float(prediction.iloc[0])

    # And retuned to the Output of the callback function
    return value, value, "Prediction: {}".format(round(prediction, 1))


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
    webbrowser.open_new(f"http://127.0.0.1:8050")
    app.run_server(host='127.0.0.1', port=8050, use_reloader=False, debug=True)

