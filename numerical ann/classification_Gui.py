from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import tensorflow as tf

# Making the root
root = Tk()
root.title("Data Entry Form")
root.geometry("700x400")

# A title for the gui
title = Label(root, text="Data Entry Form",font="calibre 20 bold")
title.grid(row=0, column=0,pady=10, columnspan=4)

# Making every label and its entry as a grid
input_label = Label(root, text="Age: ",font="calibre 12")
input_label.grid(row=1, column=0)

Age = Entry(root, width=20)
Age.grid(row=1, column=1,padx=15, pady=10)

# Create a label for the dropdown menu
label = ttk.Label(root, text="Race: ", font="calibre 12")
label.grid(row=2, column=0)

races = ['White', 'Black','Other']
Race = ttk.Combobox(root, state="readonly", values=races)
Race.grid(row=2, column=1, padx=15, pady=10)

input_label = Label(root, text="Marital Status: ",font="calibre 12")
input_label.grid(row=3, column=0)

Marital_Status=['Married' ,'Divorced' ,'Single ' ,'Widowed' ,'Separated']
Marital_Status= ttk.Combobox(root, state="readonly", values=Marital_Status)
Marital_Status.grid(row=3, column=1,padx=15, pady=10)

input_label = Label(root, text="N Stage: ",font="calibre 12")
input_label.grid(row=4, column=0)

N_Stage=['N1', 'N2', 'N3']
N_Stage=ttk.Combobox(root, state="readonly", values=N_Stage)
N_Stage.grid(row=4, column=1,padx=15, pady=10)

input_label = Label(root, text="6th Stage: ",font="calibre 12")
input_label.grid(row=5, column=0)

th_Stage=['IIA', 'IIIA','IIIC','IIB','IIIB']
th_Stage= ttk.Combobox(root, state="readonly", values=th_Stage)
th_Stage.grid(row=5, column=1,padx=15, pady=10)

input_label = Label(root, text="differentiate: ",font="calibre 12")
input_label.grid(row=6, column=0)

differentiate=['Poorly differentiated','Moderately differentiated','Well differentiated','Undifferentiated']
differentiate= ttk.Combobox(root, state="readonly", values=differentiate)
differentiate.grid(row=6, column=1,padx=15, pady=10)

input_label = Label(root, text="Grade: ",font="calibre 12")
input_label.grid(row=7, column=0)

Grade = Entry(root, width=20)
Grade.grid(row=7, column=1,padx=15, pady=10)

input_label = Label(root, text="A Stage: ",font="calibre 12")
input_label.grid(row=1, column=4)

Stage=['Regional', 'Distant']
Stage =ttk.Combobox(root, state="readonly", values=Stage)
Stage.grid(row=1, column=5,padx=15, pady=10)

input_label = Label(root, text="Tumor Size: ",font="calibre 12")
input_label.grid(row=2, column=4)

Tumor_Size = Entry(root, width=20)
Tumor_Size.grid(row=2, column=5,padx=15, pady=10)

input_label = Label(root, text="Estrogen Status: ",font="calibre 12")
input_label.grid(row=3, column=4)

Estrogen_Status=['Positive', 'Negative']
Estrogen_Status = ttk.Combobox(root, state="readonly", values=Estrogen_Status)
Estrogen_Status.grid(row=3, column=5,padx=15, pady=10)

input_label = Label(root, text="Progesterone Status: ",font="calibre 12")
input_label.grid(row=4, column=4)

Progesterone_Status=['Positive', 'Negative']
Progesterone_Status = ttk.Combobox(root, state="readonly", values=Progesterone_Status)
Progesterone_Status.grid(row=4, column=5,padx=15, pady=10)

input_label = Label(root, text="Regional Node Examined: ",font="calibre 12")
input_label.grid(row=5, column=4)

Regional_Node_Examined = Entry(root, width=20)
Regional_Node_Examined.grid(row=5, column=5,padx=15, pady=10)

input_label = Label(root, text="Reginol Node Positive: ",font="calibre 12")
input_label.grid(row=6, column=4)

Reginol_Node_Positive = Entry(root, width=20)
Reginol_Node_Positive.grid(row=6, column=5,padx=15, pady=10)

input_label = Label(root, text="Survival Months: ",font="calibre 12")
input_label.grid(row=7, column=4)

Survival_Months = Entry(root, width=20)
Survival_Months.grid(row=7, column=5,padx=15, pady=10)

# Load the model
filename = 'ANNS_model_best.h5'
loaded_model = tf.keras.models.load_model(filename)

# My function
def predict(parameter1, parameter2,parameter3,parameter4,parameter5,parameter6,parameter7,parameter8,parameter9, parameter10,parameter11,parameter12,parameter13,parameter14):
    test= [[parameter1, parameter2,parameter3,parameter4,parameter5,parameter6,parameter7,parameter8,parameter9, parameter10,parameter11,parameter12,parameter13,parameter14]]
    # Define the column names
    column_names = ['Age','Race','Marital Status','N Stage','6th Stage','differentiate','Grade','A Stage','Tumor Size','Estrogen Status','Progesterone Status','Regional Node Examined','Reginol Node Positive','Survival Months']

    df = pd.DataFrame(test, columns=column_names)
    df['Race'] = df['Race'].map({'White': 0, 'Black': 1, 'Other': 2})
    df['Marital Status'] = df['Marital Status'].map(
        {'Married': 0, 'Divorced': 1, 'Single ': 2, 'Widowed': 3, 'Separated': 4})
    df['N Stage'] = df['N Stage'].map({'N1': 0, 'N2': 1, 'N3': 2})
    df['6th Stage'] = df['6th Stage'].map({'IIA': 0, 'IIIA': 1, 'IIIC': 2, 'IIB': 3, 'IIIB': 4})
    df['differentiate'] = df['differentiate'].map(
        {'Poorly differentiated': 0, 'Moderately differentiated': 1, 'Well differentiated': 2,
         'Undifferentiated': 3})
    df['A Stage'] = df['A Stage'].map({'Regional': 0, 'Distant': 1})
    df['Estrogen Status'] = df['Estrogen Status'].map({'Positive': 0, 'Negative': 1})
    df['Progesterone Status'] = df['Progesterone Status'].map({'Positive': 0, 'Negative': 1})
    print(df.values)
    predict = loaded_model.predict(df.values)
    print(predict)
    return predict

def button_clicked():
    parameter1=float(Age.get())
    parameter2=Race.get()
    parameter3=Marital_Status.get()
    parameter4=N_Stage.get()
    parameter5=th_Stage.get()
    parameter6=differentiate.get()
    parameter7=float(Grade.get())
    parameter8 = Stage.get()
    parameter9 = float(Tumor_Size.get())
    parameter10 = Estrogen_Status.get()
    parameter11 = Progesterone_Status.get()
    parameter12 = float(Regional_Node_Examined.get())
    parameter13 = float(Reginol_Node_Positive.get())
    parameter14 = float(Survival_Months.get())
    if not all([parameter1, parameter2,parameter3,parameter4,parameter5,parameter6,parameter7,parameter8,parameter9, parameter10,parameter11,parameter12,parameter13,parameter14]):
        messagebox.showerror("Error", "All parameters are required!")
        return
    else:
        percentage=(predict(parameter1, parameter2,parameter3,parameter4,parameter5,parameter6,parameter7,parameter8,parameter9,parameter10,parameter11,parameter12,parameter13,parameter14))
        if np.round(percentage)==0:
            messagebox.showinfo("Result", f"This patient is Alive. with priority{(1-percentage)*100}")
        else:
            messagebox.showinfo("Result", f"This patient is Dead. with priority{(percentage)*100}")
        print()

# Making button
button = Button(root, text="Predict !!", command=button_clicked,font="calibre 10")
button.grid(row=10, column=3)

root.mainloop()
