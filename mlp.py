import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class MLP(ClassifierMixin, BaseEstimator):
    """Нейросеть pyTotch."""

    def __init__(self, hidden_size=64, n_layers=1, dropout=0.0, batchnorm=False, lr=0.001, epochs=100,
                 batch_size=64, random_state=0):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        """Собирает сеть и учит её: AdamW, косинусное расписание learning rate, бинарная кросс-энтропия."""
        torch.manual_seed(self.random_state)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)

        layers, size = [], X.shape[1]
        for _ in range(self.n_layers):
            layers.append(nn.Linear(size, self.hidden_size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers += [nn.ReLU(), nn.Dropout(self.dropout)]
            size = self.hidden_size
        layers.append(nn.Linear(size, 1))
        self.net_ = nn.Sequential(*layers)

        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.BCEWithLogitsLoss()
        self.net_.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X)).split(self.batch_size):
                # BatchNorm не может посчитать статистику по одному объекту
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                loss = loss_fn(self.net_(X[idx]).squeeze(1), y[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Вероятности классов 0 и 1."""
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)
        proba = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - proba, proba])
