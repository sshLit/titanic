import itertools

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data import make_preprocessor
from mlp import MLP

MODELS = {
    "linreg": LinearRegression,
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "catboost": CatBoostClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
    "mlp": MLP,
}


def make_model(config, name, params, prep):
    """Собирает Pipeline из препроцессинга и модели с заданными параметрами."""
    if name not in ("linreg", "knn"):
        params = {"random_state": config.general.seed, **params}
    return make_pipeline(make_preprocessor(config, **prep), MODELS[name](**params))


def get_folds(config, n_splits):
    """StratifiedKFold на n_splits фолдов, при n_splits=1 один сплит 80/20."""
    if n_splits == 1:
        return StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.general.seed)
    return StratifiedKFold(n_splits, shuffle=True, random_state=config.general.seed)


def predict_proba(model, X):
    """Вероятность выжить. У линейной регрессии её нет, поэтому предсказание обрезается до [0, 1]."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0, 1)


def cross_validate(model, X, y, X_test, folds):
    """Обучает модель на каждом фолде, возвращает OOF и среднее предсказание на test."""
    oof =np.full(len(X), np.nan)
    test_pred = np.zeros(len(X_test))
    for train_idx, val_idx in folds.split(X, y):
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = predict_proba(fold_model, X.iloc[val_idx])
        test_pred += predict_proba(fold_model, X_test) / folds.get_n_splits()
    return oof, test_pred


def score(oof, y, folds):
    """Accuracy по фолдам (среднее и std) и ROC-AUC по всем OOF."""
    accs =[accuracy_score(y.iloc[val_idx], oof[val_idx] >= 0.5) for _, val_idx in folds.split(oof, y)]
    mask = ~np.isnan(oof)
    return {"accuracy": np.mean(accs), "std": np.std(accs), "roc_auc": roc_auc_score(y[mask], oof[mask])}


def param_grid(params, grid):
    """Перебирает все комбинации сетки, для каждой отдаёт подпись и полный набор параметров."""
    for values in itertools.product(*grid.values()):
        combo = dict(zip(grid, values))
        yield ",".join(f"{k}={v}" for k, v in combo.items()), {**params, **combo}


def ensemble(method, oof, test, y, folds):
    """Объединяет предсказания моделей: среднее, голосование или стекинг."""
    if method == "mean":
        return oof.mean(axis=1).to_numpy(), test.mean(axis=1).to_numpy()
    if method == "voting":
        return (oof >= 0.5).mean(axis=1).to_numpy(), (test >= 0.5).mean(axis=1).to_numpy()

    meta = LinearRegression() if method == "stacking_linear" else Ridge()
    meta_oof = cross_val_predict(meta, oof, y, cv=folds)
    meta_test = meta.fit(oof, y).predict(test)
    return np.clip(meta_oof, 0, 1), np.clip(meta_test, 0, 1)
