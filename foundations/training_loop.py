import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            # Forward pass
            y_hat = X @ w + b
            err=y_hat-y
            dl_dw=(2.0/n)*(X.T @ err)
            dl_db=(2.0/n)*np.sum(err)
            w=w-lr*dl_dw
            b=b-lr*dl_db
        return (np.round(w, 5), round(float(b), 5))
