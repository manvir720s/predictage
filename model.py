import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Food_Preference.csv')

data.drop(['Timestamp', 'Participant_ID'], axis =1, inplace = True)

#remove missing values
data.dropna(axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data['Age'])
age_bins=pd.qcut(data['Age'], q=2, labels=[0,1])
data['Age']=age_bins
categorical_features = ['Gender', 'Nationality', 'Food', 'Juice', 'Dessert']
'''
def get_uniques(df, columns):
    return {column: list(df[column].unique()) for column in columns}

get_uniques(data, categorical_features)

binary_features=['Gender', 'Food', 'Juice']
ordinal_features = ['Dessert']
nominal_features=['Nationality']
'''

def binary_encode(df, column, positive_label): 
    df = df.copy()
    df[column]=df[column].apply(lambda x: 1 if x == positive_label else 0)
    return df

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))  
    return df

def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column])
    df = pd.concat([df, dummies], axis = 1)
    df.drop(column, axis=1, inplace=True)
    return df

print(data)
data = binary_encode(data, 'Gender', 'Male')
data = binary_encode(data, 'Food', 'Traditional Food')
data = binary_encode(data, 'Juice', 'Fresh Juice')


data["Dessert"] = data["Dessert"].replace(['No', 'Maybe' , 'Yes'],[0, 1, 2])

data = onehot_encode(data, 'Nationality')
print(data)
y=data['Age']
X=data.drop('Age', axis=1)

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

model=LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))