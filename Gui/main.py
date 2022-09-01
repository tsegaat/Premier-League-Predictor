import code
from PIL import Image, ImageTk
import tkinter as tk
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
import os
from tkinter import messagebox

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# codecs = ['ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855',
#           'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949',
#           'cp950', 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258',
#           'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr', 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
#           'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6',
#           'iso8859_7', 'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t',
#           'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004',
#           'shift_jisx0213', 'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8', 'utf_8_sig']

# for x in range(len(codecs)):
#     try:
#         data = pd.read_csv(
#             "Premier League Predict/Gui/final_dataset.csv", encoding=codecs[x]
#         )
#         print("\nThis is the right codec ", codecs[x])
#     except:
#         print("the codec is not it ", codecs[x])

data = pd.read_csv(
    "Premier League Predict/Gui/final_dataset.csv", encoding="ISO-8859–1"
)
print(data)
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

# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(units=836, return_sequences=True, input_shape=(5, 2)))
# model.add(Dropout(0.2))

# model.add(keras.layers.LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))

# model.add(keras.layers.LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))

# model.add(keras.layers.LSTM(units=100))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation="softmax"))

# nadam = keras.optimizers.Nadam(lr=0.0001)
# model.compile(
#     optimizer=nadam, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )

# model.fit(x_train, y_train, epochs=10)

model = keras.models.load_model("premier_league.h5")
# print("\nAvilable Teams To Pick From: ", end="")
# for team in teams:
#     print("\n", team)

# reverse_team = {}
# for k, v in team_name_mapping.items():
#     reverse_team[v] = k

# print("")
# home_team = input("Home Team: ")
# away_team = input("Away Team: ")
# home_team = reverse_team[home_team]
# away_team = reverse_team[away_team]


# half_time_home_goals = input("Half Time Home Goals: ")
# half_time_away_goals = input("Half Time Away Goals: ")
# if half_time_home_goals.isalpha() or half_time_away_goals.isalpha():
#     half_time_home_goals = int(half_time_home_goals)
#     half_time_away_goals = int(half_time_away_goals)
# else:
#     exit(1)

# home_shots = input("Home Shots: ")
# away_shots = input("Away Shots: ")
# if home_shots.isalpha() or away_shots.isalpha():
#     home_shots = int(home_shots)
#     away_shots = int(away_shots)
# else:
#     exit(1)

# home_shots_target = input("Home Shots On Target: ")
# away_shots_target = input("Away Shots On Target: ")
# if home_shots_target.isalpha() or away_shots_target.isalpha():
#     home_shots_target = int(home_shots_target)
#     away_shots_target = int(away_shots_target)
# else:
#     exit(1)

# home_fouls = input("Fouls done by home side: ")
# away_fouls = input("Fouls done by away side: ")
# if home_fouls.isalpha() or away_fouls.isalpha():
#     home_fouls = int(home_fouls)
#     away_fouls = int(away_fouls)
# else:
#     exit(1)

# predictions = model.predict(
#     [
#         [
#             [home_team, away_team],
#             [half_time_home_goals, half_time_away_goals],
#             [home_shots, away_shots],
#             [home_shots_target, away_shots_target],
#             [home_fouls, away_fouls],
#         ]
#     ]
# )

# results = ["Home Win", "Draw", "Away Win"]
# predictions = predictions[0]

# index = np.argmax(predictions)

# result = ""
# if results[index] == 0:
#     result = team_name_mapping[home_team]
# elif results[index] == 1:
#     result = "Draw"
# else:
#     result = team_name_mapping[away_team]

root = tk.Tk()

root.geometry("500x600")
root.resizable(0, 0)
root.title("Premier League Predictor")

pilImage = Image.open(
    "skysports-premier-league-graphic_4983467.jpg"
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
