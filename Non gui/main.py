#!/usr/bin/env python
"""Contains a model to predict premier league soccer games after the half time

This model is built using LSTM layers and one dense layer before the 
softmax output layer.

*tensorflow == "2.2.0"
*pandas == "1.5.0"
*numpy="1.23.3"
"""

# Importing packages
import time
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense, Dropout
import pandas as pd
import numpy as np
from tensorflow import keras
import os

# Reading the dataset
data = pd.read_csv(
    "Non gui/data/final_dataset.csv", encoding="ISO-8859–1"
)
m = 7000

# Parsing the y value from string to integer
results = data["FTR"][:7000].to_numpy()
results = np.where(results == "H", 0, results)
results = np.where(results == "D", 1, results)
results = np.where(results == "A", 2, results)
results = np.asarray(results).astype(np.int64)

# Assigning integers for each team in the dataset
team_name_mapping = {k: v for (k, v) in enumerate(data["HomeTeam"].unique())}
teams = data["HomeTeam"].unique()

home_team_in_num = []
for team in data["HomeTeam"]:
    for i in range(len(team_name_mapping.keys())):
        if team == team_name_mapping[i]:
            home_team_in_num.append(np.where(teams == team)[0][0])

away_team_in_num = []
for team in data["AwayTeam"]:
    for i in range(len(team_name_mapping.keys())):
        if team == team_name_mapping[i]:
            away_team_in_num.append(np.where(teams == team)[0][0])

# Converting to numpy array, and concatenating teams with the other x values
home_team = np.array(home_team_in_num, dtype=np.int8).reshape(m, 1)
away_team = np.array(away_team_in_num, dtype=np.int8).reshape(m, 1)
partial_data = data[
    [
        "HTHG",
        "HTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
    ]
].to_numpy()

data = np.concatenate((home_team, away_team, partial_data[:7000]), axis=1)
data = np.where(data == None, 0, data)
data = np.asarray(data).astype(np.float32)

# Splitting the data and converting from np array to tensor
x_train = tf.convert_to_tensor(data[:6950].reshape(6950, 5, 2))
x_test = tf.convert_to_tensor(data[6950:].reshape(50, 5, 2))
y_train = tf.convert_to_tensor(results[:6950])
y_test = tf.convert_to_tensor(results[6950:])

# Configuring the model with 4 LSTM layers 4 Dropout layers and 2 Dense Layers
# Dropout used to help with overfitting
model = keras.models.Sequential()
model.add(keras.layers.LSTM(
    units=512, return_sequences=True, input_shape=(5, 2)))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation="softmax"))

# Setting the optimizer, loss function, and metric
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Fitting the model with the training data, and printing the accuracy when used with the test set
model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test, y_test))
print(model.summary())
# model.save("Self Work\preimer_pridict\premier_league.h5")


# Giving the user the teams to choose from
print("\nAvilable Teams To Pick From: ", end="")
for team in teams:
    print("\n", team)

# Creating a reversed team_mapping to get the chosen clubs
reverse_team = {v: k for k, v in team_name_mapping.items()}

print("")
home_team = input("Home Team: ")
away_team = input("Away Team: ")
home_team = reverse_team[home_team]
away_team = reverse_team[away_team]

# Getting and validating the half time data from the user
half_time_home_goals = input("Half Time Home Goals: ")
half_time_away_goals = input("Half Time Away Goals: ")
if half_time_home_goals.isdigit() or half_time_away_goals.isdigit():
    half_time_home_goals = int(half_time_home_goals)
    half_time_away_goals = int(half_time_away_goals)
else:
    exit(1)

home_shots = input("Home Shots: ")
away_shots = input("Away Shots: ")
if home_shots.isdigit() or away_shots.isdigit():
    home_shots = int(home_shots)
    away_shots = int(away_shots)
else:
    exit(1)

home_shots_target = input("Home Shots On Target: ")
away_shots_target = input("Away Shots On Target: ")
if home_shots_target.isdigit() or away_shots_target.isdigit():
    home_shots_target = int(home_shots_target)
    away_shots_target = int(away_shots_target)
else:
    exit(1)

home_fouls = input("Fouls done by home side: ")
away_fouls = input("Fouls done by away side: ")
if home_fouls.isdigit() or away_fouls.isdigit():
    home_fouls = int(home_fouls)
    away_fouls = int(away_fouls)
else:
    exit(1)

# Using the data provided by the user to predict a result
predictions = model.predict(
    [
        [
            [home_team, away_team],
            [half_time_home_goals, half_time_away_goals],
            [home_shots, away_shots],
            [home_shots_target, away_shots_target],
            [home_fouls, away_fouls],
        ]
    ]
)

results = ["Home Win", "Draw", "Away Win"]
predictions = predictions[0]

index = np.argmax(predictions)

# Outputting the result of the prediction
if results[index] == 0:
    print("\n I Predict", team_name_mapping[home_team])
elif results[index] == 1:
    print("\n I Predict Draw")
else:
    print("\n I Predict", team_name_mapping[away_team])

time.sleep(3)
