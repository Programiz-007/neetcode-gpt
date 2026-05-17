import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        dot_prod=np.dot(X,weights)
        return np.round(dot_prod,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        def sum_sqe(mpre,gt):
            n=len(mpre)
            s=0
            for i in range(n):
                s=s+((mpre[i]-gt[i])**2)
            return s
        n=len(ground_truth)
        mse=(1/n)*sum_sqe(model_prediction,ground_truth)
        val=mse[0]
        val=np.round(val,5)
        return val

