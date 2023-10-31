# taxi-trips-predictions
[![image](https://raw.githubusercontent.com/jupyter/design/0ed7f5798358c203d8bc6c1ce0f46d9c8294fd4e/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/tmvfb/taxi-trips-predictions/blob/main/notebook.ipynb)

## Description
The task is to create a machine learning model that predicts the count of taxi trips for the next hour in [Chicago's community areas](https://en.wikipedia.org/wiki/Community_areas_in_Chicago). This is a time series task.

Datasets with their respective descriptions can be found by the following links:
https://data.cityofchicago.org/Transportation/Taxi-Trips-2022/npd7-ywjz  
https://data.cityofchicago.org/Transportation/Taxi-Trips-2023/e55j-2ewb

This repository contains:
1. A notebook with machine learning model predicting the count of taxi trips.
2. Corresponding code for time series feature generation from pandas dataframe.
3. A shell script to create the cluster of Docker containers to run the PySpark code.

## Setup (from command line)
To run the notebook, you need Docker installed. When done, run:
```
$ sh start_local_cluster.sh
```
Access the local cluster by the address you get in the terminal window.
Add the SPARK_MASTER_IP variable that you get in the terminal to the .ipynb file, cell 8.

## Tools used
* Docker
* PySpark
* Pandas
* LightGBM
* Catboost
