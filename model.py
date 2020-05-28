"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor,Pool, cv
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from math import sqrt
import catboost
import math
import pandas as pd
import numpy as np
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    combined_data = feature_vector_df.copy()

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    combined_data['Placement_Datetime'] = pd.to_datetime(combined_data['Placement - Time'])
    combined_data.loc[:, 'Placement_Date'] = combined_data['Placement_Datetime'].dt.date
    combined_data['Confirmation_datetime'] = pd.to_datetime(combined_data['Confirmation - Time'])
    combined_data['Trip_Duration'] = (combined_data['Confirmation_datetime'] - combined_data['Placement_Datetime']).map(
        lambda x: x.total_seconds())


    combined_data.drop(["Confirmation_datetime", "Placement_Date", "Placement_Datetime"], axis=1, inplace=True)
    combined_data.drop(["Arrival at Destination - Day of Month", "Arrival at Destination - Weekday (Mo = 1)",
                        "Arrival at Destination - Time"],
                       axis=1, inplace=True)
    combined_data.drop('Trip_Duration', axis=1, inplace=True)
    combined_data['Temperature'] = combined_data['Temperature'].fillna((combined_data['Temperature'].mean()))
    combined_data['Precipitation in millimeters'] = combined_data['Precipitation in millimeters'].fillna(0)
    combined_data['Arrival at Pickup - Time'] = pd.to_datetime(combined_data['Arrival at Pickup - Time'])
    combined_data['A_hour'] = combined_data['Arrival at Pickup - Time'].dt.hour
    combined_data['A_seconds'] = combined_data['Arrival at Pickup - Time'].dt.second
    combined_data['A_minutes'] = combined_data['Arrival at Pickup - Time'].dt.minute
    combined_data['am_or_pm_confirm'] = combined_data['Confirmation - Time'].astype('str').apply(
        lambda x: x.split(' ')[-1])
    combined_data['Confirmation - Time'] = pd.to_datetime(combined_data['Confirmation - Time'])
    combined_data['C_hour'] = combined_data['Confirmation - Time'].dt.hour
    combined_data['C_min'] = combined_data['Confirmation - Time'].dt.minute
    combined_data['C_sec'] = combined_data['Confirmation - Time'].dt.second
    combined_data.drop(['Arrival at Pickup - Time', 'Confirmation - Time'], axis=1, inplace=True)
    combined_data.drop('Order No', axis=1, inplace=True)
    combined_data.drop(['Pickup - Time', 'Placement - Time', 'Rider Id'], axis=1, inplace=True)
    transport = {"Vehicle Type": {"Bike": 1, "Other": 2},
                 "Personal or Business": {"Personal": 1, "Business": 2, }}
    combined_data.replace(transport, inplace=True)
    combined_data = pd.get_dummies(combined_data)
    print(combined_data)

    """
    predict_vector['Pickup - Time'] = pd.to_datetime(predict_vector['Pickup - Time'])
    predict_vector['Pickup - Hour'] = [i.hour for i in predict_vector['Pickup - Time']]
    predict_vector['Pickup - Min'] = [i.minute for i in predict_vector['Pickup - Time']]
    predict_vector['Pickup - Sec'] = [i.second for i in predict_vector['Pickup - Time']]

    predict_vector['Arrival at Pickup - Time'] = pd.to_datetime(predict_vector['Arrival at Pickup - Time'])
    predict_vector['Arrival at Pickup - Hour'] = [i.hour for i in predict_vector['Arrival at Pickup - Time']]
    predict_vector['Arrival at Pickup - Min'] = [i.minute for i in predict_vector['Arrival at Pickup - Time']]
    predict_vector['Arrival at Pickup - Sec'] = [i.second for i in predict_vector['Arrival at Pickup - Time']]

    predict_vector['Confirmation - Time'] = pd.to_datetime(predict_vector['Confirmation - Time'])
    predict_vector['Confirmation - Hour'] = [i.hour for i in predict_vector['Confirmation - Time']]
    predict_vector['Confirmation - Min'] = [i.minute for i in predict_vector['Confirmation - Time']]
    predict_vector['Confirmation - Sec'] = [i.second for i in predict_vector['Confirmation - Time']]

    predict_vector['Placement - Time'] = pd.to_datetime(predict_vector['Placement - Time'])
    predict_vector['Placement - Hour'] = [i.hour for i in predict_vector['Placement - Time']]
    predict_vector['Placement - Min'] = [i.minute for i in predict_vector['Placement - Time']]
    predict_vector['Placement - Sec'] = [i.second for i in predict_vector['Placement - Time']]

    predict_vector = predict_vector.drop('Order No', axis=1)
    predict_vector = predict_vector.drop('User Id', axis=1)
    predict_vector = predict_vector.drop('Personal or Business', axis=1)
    predict_vector = predict_vector.drop('Rider Id', axis=1)
    predict_vector = predict_vector.drop('Vehicle Type', axis=1)
    predict_vector = predict_vector.drop('Precipitation in millimeters', axis=1)
    predict_vector = predict_vector.drop('Platform Type', axis=1)
    predict_vector = predict_vector.drop('Pickup - Time', axis=1)
    predict_vector = predict_vector.drop('Arrival at Pickup - Time', axis=1)
    predict_vector = predict_vector.drop('Confirmation - Time', axis=1)
    predict_vector = predict_vector.drop('Placement - Time', axis=1)

    predict_vector['Temperature'] = predict_vector['Temperature'].fillna(predict_vector['Temperature'].mean())

    predict_vector = pd.DataFrame(predict_vector)
    """
    # ------------------------------------------------------------------------

    return combined_data

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return [round(i) for i in prediction.tolist()]
