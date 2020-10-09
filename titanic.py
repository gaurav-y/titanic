import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
import re
import os.path
from os import path

def create_titanic_pkl():
# if not path.exists("titanic_rfc.pkl"):
    print("creating pickle file")
    train_df = pd.read_csv('/Users/gauravyadav/Downloads/Titanic_dataset/train.csv')

    # Relatives - Count and Not Alone - bool
    train_df['relatives'] = train_df['SibSp'] + train_df['Parch']
    train_df.loc[train_df['relatives'] > 0, 'not_alone'] = 0
    train_df.loc[train_df['relatives'] == 0, 'not_alone'] = 1
    train_df['not_alone'] = train_df['not_alone'].astype(int)

    # Drop PassengerId
    train_df = train_df.drop(['PassengerId'], axis=1)

    # cabin missing values handling
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    train_df['Cabin'] = train_df['Cabin'].fillna("U0")
    train_df['Deck'] = train_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    train_df['Deck'] = train_df['Deck'].map(deck)
    train_df['Deck'] = train_df['Deck'].fillna(0)
    train_df['Deck'] = train_df['Deck'].astype(int)

    # drop the cabin feature
    train_df = train_df.drop(['Cabin'], axis=1)

    # Age missing values filled with random number
    mean = train_df["Age"].mean()
    std = train_df["Age"].std()
    is_null = train_df["Age"].isnull().sum()

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    age_slice = train_df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age     
    train_df["Age"] = age_slice
    train_df["Age"] = train_df["Age"].astype(int)

    # Embarked handling missing values
    common_value = train_df['Embarked'].mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(common_value)

    # Fare datatype change to int
    train_df['Fare'] = train_df['Fare'].fillna(0)
    train_df['Fare'] = train_df['Fare'].astype(int)

    # Titles extract from name
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    train_df['Title'] = train_df['Title'].map(titles)
    # filling NaN with 0, to get safe
    train_df['Title'] = train_df['Title'].fillna(0)
    train_df = train_df.drop(['Name'], axis=1)

    # encoding gender
    genders = {"male": 0, "female": 1}
    train_df['Sex'] = train_df['Sex'].map(genders)

    # ticket not considered for now because of ambiguous data
    train_df = train_df.drop(['Ticket'], axis=1)

    # Embarked encoding
    ports = {"S": 0, "C": 1, "Q": 2}
    train_df['Embarked'] = train_df['Embarked'].map(ports)

    # Binning of Age column
    train_df['Age'] = train_df['Age'].astype(int)
    train_df.loc[ train_df['Age'] <= 11, 'Age'] = 0
    train_df.loc[(train_df['Age'] > 11) & (train_df['Age'] <= 18), 'Age'] = 1
    train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 22), 'Age'] = 2
    train_df.loc[(train_df['Age'] > 22) & (train_df['Age'] <= 27), 'Age'] = 3
    train_df.loc[(train_df['Age'] > 27) & (train_df['Age'] <= 33), 'Age'] = 4
    train_df.loc[(train_df['Age'] > 33) & (train_df['Age'] <= 40), 'Age'] = 5
    train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 66), 'Age'] = 6
    train_df.loc[ train_df['Age'] > 66, 'Age'] = 6

    # Binning of Fare - Quantile cut - qcut for BINNING of 'Fare'
    # via train_df['Fare'].describe()
    # or by the following code:
    # df['quantile_ex_1'] = pd.qcut(df['ext price'], q=4)
    # df['quantile_ex_2'] = pd.qcut(df['ext price'], q=10, precision=0)
    train_df.loc[ train_df['Fare'] <= 7.91, 'Fare'] = 0
    train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
    train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
    train_df.loc[(train_df['Fare'] > 31) & (train_df['Fare'] <= 99), 'Fare']   = 3
    train_df.loc[(train_df['Fare'] > 99) & (train_df['Fare'] <= 250), 'Fare']   = 4
    train_df.loc[ train_df['Fare'] > 250, 'Fare'] = 5
    train_df['Fare'] = train_df['Fare'].astype(int)

    # Additional columns

    # Age * Pclass
    train_df['Age_Class']= train_df['Age'] * train_df['Pclass']
    # Fare per Person
    train_df['Fare_Per_Person'] = train_df['Fare']/(train_df['relatives']+1)
    train_df['Fare_Per_Person'] = train_df['Fare_Per_Person'].astype(int)

    # Features and target
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    # Model 
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    # Pickle out the file
    pout = open('titanic_rfc.pkl', 'wb')
    pickle.dump(random_forest, pout)