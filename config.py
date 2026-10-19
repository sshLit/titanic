from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 42,
    },
    'models': {
        'linreg': {},
        'logreg': {
            'params': {'solver': 'saga', 'max_iter': 5000},
            'grid': {
                'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],  # обратная сила регуляризации
                'l1_ratio': [0, 0.5, 1],  # типы штрафов, 0 - Ridge, 0.5 - ElasticNet, 1 - Lasso
            },
        },
        'knn': {
            'grid': {
                'n_neighbors': [3, 5, 9, 15, 25],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
            },
        },
        'tree': {
            'prep': {'encoding': 'ordinal', 'scale': False},
            'grid': {
                'max_depth': [3, 4, 5, 7],
                'min_samples_leaf': [1, 5, 10],
                'criterion': ['gini', 'entropy'],
            },
        },
        'random_forest': {
            'params': {'n_estimators': 500, 'n_jobs': -1},
            'prep': {'encoding': 'ordinal', 'scale': False},
            'grid': {
                'max_depth': [5, 7],
                'min_samples_leaf': [1, 3, 5],
                'max_features': ['sqrt', 0.5],
            },
        },
        'catboost': {
            'params': {'iterations': 500, 'verbose': 0, 'allow_writing_files': False},
            'prep': {'scale': False},
            'grid': {
                'depth': [4, 6],
                'learning_rate': [0.03, 0.1],
                'l2_leaf_reg': [3, 10],
            },
        },
        'lightgbm': {
            'params': {'n_estimators': 300, 'subsample': 0.8, 'subsample_freq': 1, 'verbose': -1},
            'prep': {'encoding': 'ordinal', 'scale': False},
            'grid': {
                'learning_rate': [0.02, 0.05],
                'num_leaves': [7, 15],
                'min_child_samples': [10, 30],
                'colsample_bytree': [0.6, 1.0],
                'reg_lambda': [0, 5],
            },
        },
        'xgboost': {
            'params': {'n_estimators': 300},
            'prep': {'encoding': 'ordinal', 'scale': False},
            'grid': {
                'max_depth': [3, 5],
                'learning_rate': [0.03, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.6, 1.0],
                'min_child_weight': [1, 5],
                'gamma': [0, 1],
            },
        },
        'mlp': {
            'grid': {
                'n_layers': [1, 2],
                'batchnorm': [False, True],
                'dropout': [0, 0.3],
                'lr': [0.001, 0.01],
            },
        },
    },
    'ensemble': {
        'models': ['catboost', 'lightgbm', 'random_forest', 'logreg', 'knn'],
        'methods': ['mean', 'voting', 'stacking_linear', 'stacking_ridge'],
    },
    'ablation': {
        'models': ['logreg', 'lightgbm'],
        'variants': {
            'default': {},
            'base_features': {'base_features': True},
            'no_age_by_title': {'age_by_title': False},
            'no_scaling': {'scale': False},
            'no_log_fare': {'log_fare': False},
            'ordinal_encoding': {'encoding': 'ordinal'},
            'target_encoding': {'encoding': 'target'},
            'drop_correlated': {'drop': ['FamilySize', 'Fare']},
        },
    },
}

config = OmegaConf.create(config)
