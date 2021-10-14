#!/usr/bin/env bash

MY_H2O_JAR=`find ~/anaconda3/envs/ODT_dash -name h2o.jar`

if [ -f "$MY_H2O_JAR" ]; then
    echo "$MY_H2O_JAR exists."
    echo "Running h2o server ..."
    conda activate ODT_dash
    ~/anaconda3/envs/ODT_dash/bin/python app.py
else
    echo "$MY_H2O_JAR does not exist."
fi
