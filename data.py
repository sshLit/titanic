import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

NUM = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "TicketFreq", "FarePerPerson"]
CAT = ["Pclass", "Sex", "Embarked", "Title", "Deck", "FamilyGroup", "IsAlone", "HasCabin", "AgeMissing", "SexPclass"]
BASE_NUM = ["Age", "SibSp", "Parch", "Fare"]
BASE_CAT = ["Pclass", "Sex", "Embarked"]

TITLES = {
    "Mr": "Mr", "Mrs": "Mrs", "Mme": "Mrs", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss", "Master": "Master",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer", "Capt": "Officer",
}


def load_data():
    """Читает train и test, добавляет признаки и возвращает X, y и X_test."""
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    full = add_features(pd.concat([train, test], ignore_index=True))
    return full[: len(train)], train["Survived"], full[len(train):]


def add_features(df):
    """Добавляет новые признаки: обращение, семья, палуба, билет, пол вместе с классом."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r", ([^.]+)\.")[0].map(TITLES).fillna("Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FamilyGroup"] = pd.cut(df["FamilySize"], [0, 1, 4, 11], labels=["alone", "small", "large"]).astype(str)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["TicketFreq"] = df.groupby("Ticket")["Ticket"].transform("count")
    # Fare - цена билета на всю группу, на человека она информативнее
    df["FarePerPerson"] = df["Fare"] / df["TicketFreq"]
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["SexPclass"] = df["Sex"] + "_" + df["Pclass"].astype(str)
    return df


class AgeByTitleImputer(BaseEstimator, TransformerMixin):
    """Заполняет пропуски в Age медианой по Title, медианы считаются на train-части фолда."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Title")["Age"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["Age"] = X["Age"].fillna(X["Title"].map(self.medians_))
        return X


def log1p_fare(X):
    """Логарифмирует Fare и FarePerPerson."""
    X = X.copy()
    X[["Fare", "FarePerPerson"]] = np.log1p(X[["Fare", "FarePerPerson"]])
    return X


def make_encoder(config, encoding):
    """Возвращает кодировщик категорий: onehot, ordinal или target."""
    if encoding == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if encoding == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if encoding == "target":
        return TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=config.general.seed))
    raise ValueError(f"Неизвестный encoding: {encoding}")


def make_preprocessor(config, encoding="onehot", scale=True, log_fare=True, age_by_title=True, base_features=False,
                      drop=()):
    """Собирает препроцессинг: пропуски, логарифм Fare, скейлинг, кодирование категорий."""
    num, cat =(BASE_NUM, BASE_CAT) if base_features else (NUM, CAT)
    num = [col for col in num if col not in drop]

    num_steps = [SimpleImputer(strategy="median")]
    if scale:
        num_steps.append(StandardScaler())
    cat_steps = [SimpleImputer(strategy="most_frequent"), make_encoder(config, encoding)]
    columns = ColumnTransformer([("num", make_pipeline(*num_steps), num), ("cat", make_pipeline(*cat_steps), cat)])

    steps = []
    if age_by_title:
        steps.append(AgeByTitleImputer())
    if log_fare:
        steps.append(FunctionTransformer(log1p_fare))
    return make_pipeline(*steps, columns)
