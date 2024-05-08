import pandas as pd
import joblib
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def ingest_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    df = df[['survived', 'pclass', 'sex', 'age']]
    # suppression des lignes avec des valeurs manquantes
    df.dropna(axis = 0, inplace = True)
    # remplacement des valeurs non numeriques par des valeurs numeriques
    df = df.replace('male', 1).replace('female', 0)
    return df

def train_model(df: pd.DataFrame) -> ClassifierMixin:
    # instantiate model
    model = KNeighborsClassifier(4)
    # train model
    y = df['survived']
    X = df.drop(columns = ['survived'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 
    model.fit(X_train,y_train)
    # evaluate model
    score = model.score(X_test, y_test)
    print(f'model score : {score}')
    return model

if __name__ == '__main__':
    df = ingest_data('titanic.xls')
    df = clean_data(df)
    model = train_model(df)
    joblib.dump(model, filename = 'model_titanic.joblil')