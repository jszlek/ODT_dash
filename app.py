import string
import dash
from dash import dcc
from dash import html
import statistics
import psutil
import random
import csv
import sys
from h2o.automl import H2OAutoML
import h2o
from pathlib import Path
from configparser import ConfigParser
config = ConfigParser(allow_no_value=True)
config.read('config.ini')

my_threads = int(config['DEFAULT']['my_threads'])
my_max_ram_allowed = int(config['DEFAULT']['my_max_ram_allowed'])


# create app containter
app = dash.Dash()


# -------------------------------
# Random key generator - function
# -------------------------------
def random_key_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# get current directory (PosixPath)
# -----------------------
my_current_dir = Path.cwd()

# check runtime mode - either many servers on the machine (server_multicore = F) or one server per one machine (server_multicore = T)
# -------------------------------------------
my_cores = psutil.cpu_count()-2

# check system free mem and apply it to the server
# ------------------------------------------------
memfree = psutil.virtual_memory().total
memfree_g = int(round(memfree / 1024 / 1024 / 1024 / my_cores))

if memfree_g < 2:
    memfree_g = 2

# generate random port number
# -------------------------------
my_port_number = random.randint(54322, 65000)

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
         port=my_port_number,
#         ice_root=str(my_export_dir),
         name=str(cluster_name),
         start_h2o=True)
# -------------------------------------

#--------------------------------------
# Load h2o model
#--------------------------------------

my_model = h2o.load_model('model/DeepLearning_grid__2_AutoML_20210831_185216_model_20')

# create app layout
app.layout = html.Div([
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF'],
        multi=True
    ),

    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF']
    ),

    html.Label('Text Input'),
    dcc.Input(value='MTL', type='text'),

    html.Label('Slider'),
    dcc.Slider(
        min=0,
        max=9,
        marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
        value=5,
    ),
], style={'columnCount': 2})

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)