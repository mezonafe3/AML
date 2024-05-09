import pandas as pd
from joblib import dump
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('Breast_Cancer.csv')
categorical_cols = ["Race", "Marital Status", "N Stage", "6th Stage", "differentiate",
                    "A Stage", "Estrogen Status", "Progesterone Status"]
for i in categorical_cols:
    unique_race_values = df[i].unique()
    print(f"Unique {i} Values:")
    print(unique_race_values)

df['Race'] = df['Race'].map({'White': 0, 'Black': 1,'Other': 2})
df['Marital Status'] = df['Marital Status'].map({'Married':0 ,'Divorced':1 ,'Single ':2 ,'Widowed':3 ,'Separated':4})
df['N Stage'] = df['N Stage'].map({'N1':0, 'N2':1, 'N3':2})
df['6th Stage'] = df['6th Stage'].map({'IIA':0, 'IIIA':1 ,'IIIC':2 ,'IIB':3 ,'IIIB':4})
df['differentiate'] = df['differentiate'].map({'Poorly differentiated':0, 'Moderately differentiated':1, 'Well differentiated':2,
 'Undifferentiated':3})
df['A Stage'] = df['A Stage'].map({'Regional':0, 'Distant':1})
df['Estrogen Status'] = df['Estrogen Status'].map({'Positive': 0, 'Negative': 1})
df['Progesterone Status'] = df['Progesterone Status'].map({'Positive': 0, 'Negative': 1})

#encoded_data = pd.get_dummies(df, columns=categorical_cols)
#x = encoded_data.drop('Status',axis=1)
#y = encoded_data['Status'].values
#print(encoded_data.values)
x = df.drop('Status', axis=1).values

df['Status'] = df['Status'].map({'Alive': 0, 'Dead': 1})
y = df['Status'].values
#he=LabelEncoder()
#y=he.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42,shuffle=True)
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.fit_transform(x_test)
model=tf.keras.Sequential(
    [
        layers.Dense(units=16,kernel_initializer='he_uniform',input_dim=14,activation='relu'),
        layers.Dense(units=16,kernel_initializer='he_uniform',activation='relu'),
        layers.Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid')

    ]
)
print("cancer ann",model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history= model.fit(x_train,y_train,batch_size=20,epochs=120,validation_split=0.2,shuffle=True)

y_pred = model.predict(x_test)

y_pred = (y_pred>0.5)
conf=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("cancer df",conf)
print("cancer acc",acc)

# Save the model using joblib
model.save('ANNS_model_best.h5')
# from joblib import dump
# filename = 'ANNS_model_best.joblib'
# dump(model, filename)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])

cm_display.plot()
plt.show()

plt.plot(history2.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()