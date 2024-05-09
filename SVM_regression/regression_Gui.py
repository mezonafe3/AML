from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, mean_squared_error,
                             mean_absolute_error)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA
# making the root
from tkinter import *
from tkinter import messagebox

root = Tk()
root.title("Data Entry Form")
root.geometry("580x300")

title = Label(root, text="Data Entry Form", font="calibre 20 bold")
title.grid(row=0, column=0, pady=10, sticky="nsew")

input_label = Label(root, text="Hours Studied: ", font="calibre 12")
input_label.grid(row=1, column=0, sticky="e")

hours_studied = Entry(root, width=20)
hours_studied.grid(row=1, column=1, padx=15, pady=10, sticky="w")

input_label = Label(root, text="Previous Scores: ", font="calibre 12")
input_label.grid(row=2, column=0, sticky="e")

previous_scores = Entry(root, width=20)
previous_scores.grid(row=2, column=1, padx=15, pady=10, sticky="w")

input_label = Label(root, text="Sleep Hours: ", font="calibre 12")
input_label.grid(row=3, column=0, sticky="e")

sleep_hours = Entry(root, width=20)
sleep_hours.grid(row=3, column=1, padx=15, pady=10, sticky="w")

input_label = Label(root, text="Sample Question Papers Practiced: ", font="calibre 12")
input_label.grid(row=4, column=0, sticky="e")

sample_question = Entry(root, width=20)
sample_question.grid(row=4, column=1, padx=15, pady=10, sticky="w")

input_label = Label(root, text="Extracurricular Activities: ", font="calibre 12")
input_label.grid(row=5, column=0, sticky="e")

extracurricular_activities = Entry(root, width=20)
extracurricular_activities.grid(row=5, column=1, padx=15, pady=10, sticky="w")


def button_clicked():
    parameter1 = hours_studied.get()
    parameter2 = previous_scores.get()
    parameter3 = sleep_hours.get()
    parameter4 = sample_question.get()
    parameter5 = extracurricular_activities.get()
    if not all([parameter1, parameter2, parameter3, parameter4, parameter5]):
        messagebox.showerror("Error", "All parameters are required!")
        return
    if parameter5.lower() == 'yes':
        ea_yes = 1
        ea_no = 0
    else:
        ea_no = 1
        ea_yes = 0
    stats = np.array([parameter1, parameter2, parameter3, parameter4, ea_no, ea_yes]).reshape(1, -1)
    reg = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    stats_scaled = scaler.transform(stats)
    prediction = reg.predict(stats_scaled)
    messagebox.showinfo("the performance index is : ", prediction)
    return


button = Button(root, text="Predict !!", command=button_clicked, font="calibre 12")
button.grid(row=6, column=1, padx=38, sticky="e")

root.mainloop()
