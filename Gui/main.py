#!/usr/bin/env python
"""Contains a model to predict premier league soccer games after the half time

This model is built using LSTM layers and one dense layer before the 
softmax output layer.

*tensorflow == "2.2.0"
*pandas == "1.5.0"
*numpy="1.23.3"
"""

from PIL import Image, ImageTk
import tkinter as tk
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import os
from tkinter import messagebox
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
# model = keras.models.load_model("premier_league.h5")

root = tk.Tk()

root.geometry("500x600")
root.resizable(0, 0)
root.title("Premier League Predictor")

pilImage = Image.open(
    "Gui/skysports-premier-league-graphic_4983467.jpg"
)
pilImage = pilImage.resize((500, 200), Image.ANTIALIAS)
image = ImageTk.PhotoImage(pilImage)

prem_img = tk.Label(root, image=image)
prem_img.grid(row=0, column=0, columnspan=4)

title = tk.Label(
    root, text="Welcome To Your Live Premier League Predictor", font=("cursive", 15)
)
title.grid(row=1, column=0, pady=40, columnspan=4)

TEAMS = teams[:-1]

TEAMS = np.sort(TEAMS)

# Home Team in GUI
hometeam_label = tk.Label(root, text="Home Team", font="cursive")
hometeam_label.grid(row=3, column=0, pady=10)

variable_hometeam = tk.StringVar(root)
variable_hometeam.set("Man United")  # default value

hometeam_selector = tk.OptionMenu(root, variable_hometeam, *TEAMS)
hometeam_selector.grid(row=4, column=0)

# Away Team in GUI
awayteam_label = tk.Label(root, text="Away Team", font="cursive")
awayteam_label.grid(row=3, column=3, pady=10)

variable_awayteam = tk.StringVar(root)
variable_awayteam.set("Man City")  # default value

awayteam_selector = tk.OptionMenu(root, variable_awayteam, *TEAMS)
awayteam_selector.grid(row=4, column=3)

# Home Half Time Goals in GUI
homegoals_label = tk.Label(root, text="HG", font="cursive")
homegoals_label.grid(row=3, column=1, pady=10)

homegoals_entry = tk.Entry(root, width=5, font="cursive")
homegoals_entry.grid(row=4, column=1)

# Away Half Time Goals in GUI
awaygoals_label = tk.Label(root, text="AG", font="cursive")
awaygoals_label.grid(row=3, column=2, pady=10)

awaygoals_entry = tk.Entry(root, width=5, font="cursive")
awaygoals_entry.grid(row=4, column=2)

# Home Half Time Shots in GUI
homeshots_label = tk.Label(root, text="HS", font="cursive")
homeshots_label.grid(row=5, column=0, pady=10)

homeshots_entry = tk.Entry(root, width=5, font="cursive")
homeshots_entry.grid(row=6, column=0)

# Home Half Time Shots on Target in GUI
awayshots_label = tk.Label(root, text="HST", font="cursive")
awayshots_label.grid(row=5, column=1, pady=10)

awayshots_entry = tk.Entry(root, width=5, font="cursive")
awayshots_entry.grid(row=6, column=1)

# Away Half Time Shots on Target in GUI
homeshotstarget_label = tk.Label(root, text="AST", font="cursive")
homeshotstarget_label.grid(row=5, column=2, pady=10)

homeshotstarget_entry = tk.Entry(root, width=5, font="cursive")
homeshotstarget_entry.grid(row=6, column=2)

# Away Half Time Shots in GUI
awayshotstarget_label = tk.Label(root, text="AS", font="cursive")
awayshotstarget_label.grid(row=5, column=3, pady=10)

awayshotstarget_entry = tk.Entry(root, width=5, font="cursive")
awayshotstarget_entry.grid(row=6, column=3)

# Home Half Time Fouls on Target in GUI
homefouls_label = tk.Label(root, text="HF", font="cursive")
homefouls_label.grid(row=7, column=1, pady=10)

homefouls_entry = tk.Entry(root, width=5, font="cursive")
homefouls_entry.grid(row=8, column=1)

# Away Half Time Fouls on Target in GUI
awayfouls_label = tk.Label(root, text="AF", font="cursive")
awayfouls_label.grid(row=7, column=2, pady=10)

awayfouls_entry = tk.Entry(root, width=5, font="cursive")
awayfouls_entry.grid(row=8, column=2)

# Submit Prediction Function


def get_info():
    home_team = variable_hometeam.get()
    away_team = variable_awayteam.get()
    home_goals = homegoals_entry.get()
    away_goals = awaygoals_entry.get()
    home_shots = homeshots_entry.get()
    away_shots = awayshots_entry.get()
    home_shots_target = homeshotstarget_entry.get()
    away_shots_target = awayshotstarget_entry.get()
    homefouls = homefouls_entry.get()
    awayfouls = awayfouls_entry.get()

    if (
        home_goals.isalpha()
        or away_goals.isalpha()
        or home_shots.isalpha()
        or away_shots.isalpha()
        or home_shots_target.isalpha()
        or away_shots_target.isalpha()
        or homefouls.isalpha()
        or awayfouls.isalpha()
    ):
        messagebox.showerror("Error", "Only Numerical Values Allowed")
        homegoals_entry.delete(0, "end")
        awaygoals_entry.delete(0, "end")
        awayshots_entry.delete(0, "end")
        homeshots_entry.delete(0, "end")
        homeshotstarget_entry.delete(0, "end")
        awayshotstarget_entry.delete(0, "end")
        homefouls_entry.delete(0, "end")
        awayfouls_entry.delete(0, "end")

    if (
        home_goals == ""
        or away_goals == ""
        or home_shots == ""
        or away_shots == ""
        or home_shots_target == ""
        or away_shots_target == ""
        or homefouls == ""
        or awayfouls == ""
    ):
        messagebox.showerror("Error", "Blanks can't be empty")
        homegoals_entry.delete(0, "end")
        awaygoals_entry.delete(0, "end")
        awayshots_entry.delete(0, "end")
        homeshots_entry.delete(0, "end")
        homeshotstarget_entry.delete(0, "end")
        awayshotstarget_entry.delete(0, "end")
        homefouls_entry.delete(0, "end")
        awayfouls_entry.delete(0, "end")

    away_goals = int(away_goals)
    home_goals = int(home_goals)
    home_shots = int(home_shots)
    away_shots = int(away_shots)
    home_shots_target = int(home_shots_target)
    away_shots_target = int(away_shots_target)
    homefouls = int(homefouls)
    awayfouls = int(awayfouls)

    reverse_team = {}
    for k, v in team_name_mapping.items():
        reverse_team[v] = k

    home_team = reverse_team[home_team]
    away_team = reverse_team[away_team]

    predictions = model.predict(
        [
            [
                [home_team, away_team],
                [home_goals, away_goals],
                [home_shots, away_shots],
                [home_shots_target, away_shots_target],
                [homefouls, awayfouls],
            ]
        ]
    )

    print(predictions)
    predictions = predictions[0]

    index = np.argmax(predictions)

    result = ""
    if results[index] == 0:
        result = team_name_mapping[home_team]
    elif results[index] == 1:
        result = "Draw"
    else:
        result = team_name_mapping[away_team]

    result_window = tk.Toplevel(root)
    predictor_says = tk.Label(
        result_window, text="The Winner is : ", font=("cursive", 30)
    )
    predictor_says.pack()
    result_ = tk.Label(result_window, text=result, font=("cursive", 40))
    result_.pack()


# Submit the prediction
predict_btn = tk.Button(root, text="Predict", font=(
    "cursive", 16), command=get_info)
predict_btn.grid(row=9, column=1, columnspan=2, pady=15)

root.mainloop()
