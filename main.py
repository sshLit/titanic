import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from config import config
from data import load_data
from utils import cross_validate, ensemble, get_folds, make_model, param_grid, score


def evaluate(config, name, params, prep, data, folds):
    """Кросс-валидация одной конфигурации модели: метрики и предсказания."""
    X, y, X_test = data
    oof, test_pred = cross_validate(make_model(config, name, params, prep), X, y, X_test, folds)
    return {"params": params, **score(oof, y, folds), "oof": oof, "test": test_pred}


def report(row):
    """Печатает результат одного запуска."""
    print(f"{row['name']}: accuracy {row['accuracy']:.4f}, std {row['std']:.4f}, roc_auc {row['roc_auc']:.4f}")


def save_submission(ids, proba, path):
    """Сохраняет предсказания на test в формате сабмита."""
    pd.DataFrame({"PassengerId": ids, "Survived": (proba >= 0.5).astype(int)}).to_csv(path, index=False)


def run(config):
    """Весь прогон: сетки, абляция, ансамбли, holdout и сохранение результатов."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    data = load_data()
    X, y, X_test = data
    print(X.head().T.to_string())
    folds = get_folds(config, 5)

    rows = []
    for name in config.get("only") or config.models:
        model_cfg = config.models[name]
        for label, params in param_grid(model_cfg.get("params", {}), model_cfg.get("grid", {})):
            rows.append({"stage": "grid", "model": name, "name": f"{name}[{label}]" if label else name,
                         **evaluate(config, name, params, model_cfg.get("prep", {}), data, folds)})
            report(rows[-1])

    best = pd.DataFrame(rows).sort_values("accuracy", ascending=False, kind="stable").drop_duplicates("model")
    best = best.set_index("model")

    for name in config.ablation.models:
        if name not in best.index:
            continue
        for variant, prep in config.ablation.variants.items():
            rows.append({"stage": "ablation", "model": name, "name": f"{name}:{variant}",
                         **evaluate(config, name, best.loc[name, "params"], prep, data, folds)})
            report(rows[-1])

    members = [name for name in config.ensemble.models if name in best.index]
    if len(members) > 1:
        oof = pd.DataFrame({name: best.loc[name, "oof"] for name in members})
        test_proba = pd.DataFrame({name: best.loc[name, "test"] for name in members})
        for method in config.ensemble.methods:
            ens_oof, ens_test = ensemble(method, oof, test_proba, y, folds)
            rows.append({"stage": "ensemble", "model": f"ensemble_{method}", "name": f"ensemble_{method}",
                         "params": {"members": members},
                         **score(ens_oof, y, folds), "oof": ens_oof, "test": ens_test})
            report(rows[-1])
            save_submission(X_test["PassengerId"], ens_test, out_dir / f"submission_ensemble_{method}.csv")

    top_name, top = best.index[0], best.iloc[0]
    top_prep = config.models[top_name].get("prep", {})
    rows.append({"stage": "holdout", "model": top_name, "name": f"{top_name} (1 fold)",
                 **evaluate(config, top_name, top["params"], top_prep, data, get_folds(config, 1))})
    report(rows[-1])
    save_submission(X_test["PassengerId"], top["test"], out_dir / f"submission_{top_name}_5fold.csv")
    save_submission(X_test["PassengerId"], rows[-1]["test"], out_dir / f"submission_{top_name}_1fold.csv")

    results = pd.DataFrame(rows)
    pd.DataFrame({name: best.loc[name, "oof"] for name in best.index}).to_csv(out_dir / "oof.csv", index=False)
    results["params"] = results["params"].map(json.dumps)
    results.drop(columns=["oof", "test"]).to_csv(out_dir / "results.csv", index=False)

    summary = results[results["stage"].isin(["grid", "ensemble"])]
    summary = summary.sort_values("accuracy", ascending=False, kind="stable").drop_duplicates("model")
    print("\nЛучшая конфигурация каждого эксперимента:")
    print(summary[["name", "accuracy", "std", "roc_auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    run(OmegaConf.merge(config, OmegaConf.from_cli()))
