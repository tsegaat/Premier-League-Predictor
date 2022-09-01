import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers.core import Dense, Dropout
import time

data = pd.read_csv(
    "Self Work\preimer_pridict\data\\final_dataset.csv", encoding="ISO-8859–1"
)
data = data[
    [
        "HomeTeam",
        "AwayTeam",
        "HTHG",
        "HTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
    ]
]

team_name_mapping = {k: v for (k, v) in enumerate(data["HomeTeam"].unique())}
teams = data["HomeTeam"].unique()

home_team_in_num = []
for team in data["HomeTeam"]:
    for i in range(len(team_name_mapping.keys())):
        if team == team_name_mapping[i]:
            values = list(team_name_mapping.values())
            home_team_in_num.append(values.index(team))

away_team_in_num = []
for team in data["AwayTeam"]:
    for i in range(len(team_name_mapping.keys())):
        if team == team_name_mapping[i]:
            values = list(team_name_mapping.values())
            away_team_in_num.append(values.index(team))

matches = []
for (
    home_team,
    away_team,
    half_time_home_goals,
    half_time_away_goals,
    home_shot,
    away_shot,
    home_shot_target,
    away_shot_target,
    home_fouls,
    away_fouls,
) in zip(
    home_team_in_num,
    away_team_in_num,
    data["HTHG"],
    data["HTAG"],
    data["HS"],
    data["AS"],
    data["HST"],
    data["AST"],
    data["HF"],
    data["AF"],
):
    matches.append(
        [
            [home_team, away_team],
            [half_time_home_goals, half_time_away_goals],
            [home_shot, away_shot],
            [home_shot_target, away_shot_target],
            [home_fouls, away_fouls],
        ]
    )

results = []
for result in data["FTR"]:
    if result == "H":
        results.append(0)
    elif result == "D":
        results.append(1)
    elif result == "A":
        results.append(2)

results = np.array(results, np.int8)
matches = np.array(matches, np.int32)

x_train = np.array(matches[:6000], dtype=np.int32)
x_test = np.array(matches[6000:], dtype=np.int32)
y_train = np.array(results[:6000], dtype=np.int32)
y_test = np.array(results[6000:], dtype=np.int32)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=836, return_sequences=True, input_shape=(5, 2)))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(keras.layers.LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))

nadam = keras.optimizers.Nadam(lr=0.0001)
model.compile(
    optimizer=nadam, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=200)

model.save("Self Work\preimer_pridict\premier_league.h5")

print("\nAvilable Teams To Pick From: ", end="")
for team in teams:
    print("\n", team)

reverse_team = {}
for k, v in team_name_mapping.items():
    reverse_team[v] = k

print("")
home_team = input("Home Team: ")
away_team = input("Away Team: ")
home_team = reverse_team[home_team]
away_team = reverse_team[away_team]


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

if results[index] == 0:
    print("\n I Predict", team_name_mapping[home_team])
elif results[index] == 1:
    print("\n I Predict Draw")
else:
    print("\n I Predict", team_name_mapping[away_team])

time.sleep(3)
