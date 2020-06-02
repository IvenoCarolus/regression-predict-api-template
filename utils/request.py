"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np
import json

def preprocess_data(data):
    """Funtion just does preprocessing on the test data but will be used on the test dataset before converting to json string


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    combined_data = []
    if (type(data) != pd.DataFrame):
        # Convert the json string to a python dictionary object
        feature_vector_dict = json.loads(data)

        # Load the dictionary as a Pandas DataFrame.
        feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

        combined_data = feature_vector_df.copy()
        return combined_data
    else:
        combined_data = data

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
    """combined_data.drop(["Arrival at Destination - Day of Month", "Arrival at Destination - Weekday (Mo = 1)",
                        "Arrival at Destination - Time"],
                       axis=1, inplace=True)"""
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
    #combined_data=combined_data.drop("Time from Pickup to Arrival", axis=1, inplace=True)
    combined_data = pd.get_dummies(combined_data)
    return combined_data[pd.read_csv('data/train_data.csv').shape[0]:]

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Zindi challenge.
train = pd.read_csv('/home/explore-student/team_5_sendy_api/utils/data/train_data.csv')
test = pd.read_csv('/home/explore-student/team_5_sendy_api/utils/data/test_data.csv')
prev_test = test
riders = pd.read_csv('/home/explore-student/team_5_sendy_api/utils/data/riders.csv')
test = test.merge(riders, how='left', on='Rider Id')
test = pd.concat((train, test)).reset_index(drop=True)
test=preprocess_data(test)

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[0].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

url = 'http://ec2-52-31-213-28.eu-west-1.compute.amazonaws.com:5000/api_v0.1'
#url = 'http://127.0.0.1:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {prev_test.iloc[0].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()[0]}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
