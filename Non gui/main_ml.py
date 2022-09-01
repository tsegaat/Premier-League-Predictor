# data preprocessing
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb

# the outcome (dependent variable) has only a limited number of possible values.
# Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression

# A random forest is a meta estimator that fits a number of decision tree classifiers
# on various sub-samples of the dataset and use averaging to improve the predictive
# accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier

# a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


from time import time

# F1 score (also F-score or F-measure) is a measure of a test's accuracy.
# It considers both the precision p and the recall r of the test to compute
# the score: p is the number of correct positive results divided by the number of
# all positive results, and r is the number of correct positive results divided by
# the number of positive results that should have been returned. The F1 score can be
# interpreted as a weighted average of the precision and recall, where an F1 score
# reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score


def train_classifier(clf, X_train, y_train):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    """ Makes predictions using a fit classifier based on F1 score. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return (
        f1_score(target, y_pred, pos_label="H"),
        sum(target == y_pred) / float(len(y_pred)),
    )


def train_predict(clf, X_train, y_train, X_test, y_test):
    """ Train and predict using a classifer based on F1 score. """

    # Indicate the classifier and the training set size
    print(
        "Training a {} using a training set size of {}. . .".format(
            clf.__class__.__name__, len(X_train)
        )
    )

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print(
        "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc)
    )

    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


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

X_train, X_test, y_train, y_test = train_test_split(matches, results, test_size=0.2)
print(X_train[0])
X_train = X_train.reshape(-1, 10)
print(X_train[0])
X_test = X_test.reshape(-1, 10)

clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=912, kernel="rbf")
clf_C = xgb.XGBClassifier(seed=82)

# clf_A.fit(X_train, y_train)
# clf_B.fit(X_train, y_train)

clf_B.fit(X_train, y_train)

reverse_team = {}
for k, v in team_name_mapping.items():
    reverse_team[v] = k

home_team = reverse_team["Tottenham"]
away_team = reverse_team["Leicester"]

print(clf_B.predict([[home_team, away_team, 2, 2, 2, 4, 1, 2, 6, 5]]))
print(clf_B.score(X_test, y_test))


root = tk.Tk()

root.geometry("500x600")
root.resizable(0, 0)
root.title("Premier League Predictor")

pilImage = Image.open(
    "Self Work\premier_league_gui\skysports-premier-league-graphic_4983467.jpg"
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
hometeam_label.grid(row=2, column=0, pady=10)

variable_hometeam = tk.StringVar(root)
variable_hometeam.set("Man United")  # default value

hometeam_selector = tk.OptionMenu(root, variable_hometeam, *TEAMS)
hometeam_selector.grid(row=3, column=0)

# Away Team in GUI
awayteam_label = tk.Label(root, text="Away Team", font="cursive")
awayteam_label.grid(row=2, column=3, pady=10)

variable_awayteam = tk.StringVar(root)
variable_awayteam.set("Man City")  # default value

awayteam_selector = tk.OptionMenu(root, variable_awayteam, *TEAMS)
awayteam_selector.grid(row=3, column=3)

# Home Half Time Goals in GUI
homegoals_label = tk.Label(root, text="HG", font="cursive")
homegoals_label.grid(row=2, column=1, pady=10)

homegoals_entry = tk.Entry(root, width=5, font="cursive")
homegoals_entry.grid(row=3, column=1)

# Away Half Time Goals in GUI
awaygoals_label = tk.Label(root, text="AG", font="cursive")
awaygoals_label.grid(row=2, column=2, pady=10)

awaygoals_entry = tk.Entry(root, width=5, font="cursive")
awaygoals_entry.grid(row=3, column=2)

# Home Half Time Shots in GUI
homeshots_label = tk.Label(root, text="HS", font="cursive")
homeshots_label.grid(row=4, column=0, pady=10)

homeshots_entry = tk.Entry(root, width=5, font="cursive")
homeshots_entry.grid(row=5, column=0)

# Home Half Time Shots on Target in GUI
awayshots_label = tk.Label(root, text="HST", font="cursive")
awayshots_label.grid(row=4, column=1, pady=10)

awayshots_entry = tk.Entry(root, width=5, font="cursive")
awayshots_entry.grid(row=5, column=1)

# Away Half Time Shots on Target in GUI
homeshotstarget_label = tk.Label(root, text="AST", font="cursive")
homeshotstarget_label.grid(row=4, column=2, pady=10)

homeshotstarget_entry = tk.Entry(root, width=5, font="cursive")
homeshotstarget_entry.grid(row=5, column=2)

# Away Half Time Shots in GUI
awayshotstarget_label = tk.Label(root, text="AS", font="cursive")
awayshotstarget_label.grid(row=4, column=3, pady=10)

awayshotstarget_entry = tk.Entry(root, width=5, font="cursive")
awayshotstarget_entry.grid(row=5, column=3)

# Home Half Time Fouls on Target in GUI
homefouls_label = tk.Label(root, text="HF", font="cursive")
homefouls_label.grid(row=6, column=1, pady=10)

homefouls_entry = tk.Entry(root, width=5, font="cursive")
homefouls_entry.grid(row=7, column=1)

# Away Half Time Fouls on Target in GUI
awayfouls_label = tk.Label(root, text="AF", font="cursive")
awayfouls_label.grid(row=6, column=2, pady=10)

awayfouls_entry = tk.Entry(root, width=5, font="cursive")
awayfouls_entry.grid(row=7, column=2)

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

    predictions = clf_B.predict(
        [
            [
                home_team,
                away_team,
                home_goals,
                away_goals,
                home_shots,
                away_shots,
                home_shots_target,
                away_shots_target,
                homefouls,
                awayfouls,
            ]
        ]
    )

    print(predictions)
    predictions = predictions[0]

    result = ""
    if predictions == 0:
        result = team_name_mapping[home_team]
    elif predictions == 1:
        result = "Draw"
    else:
        result = team_name_mapping[away_team]

    result_window = tk.Toplevel(root)
    predictor_says = tk.Label(
        result_window, text="Predictor Says: ", font=("cursive", 30)
    )
    predictor_says.pack()
    result_ = tk.Label(result_window, text=result, font=("cursive", 40))
    result_.pack()


# Submit the prediction
predict_btn = tk.Button(root, text="Predict", font=("cursive", 16), command=get_info)
predict_btn.grid(row=8, column=1, columnspan=2, pady=15)

root.mainloop()

# # train_predict(clf_A, X_train, y_train, X_test, y_test)
# # print("")
# train_predict(clf_B, X_train, y_train, X_test, y_test)
# print("")
# # train_predict(clf_C, X_train, y_train, X_test, y_test)
# # print("")

