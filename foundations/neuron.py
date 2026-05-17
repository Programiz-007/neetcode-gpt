import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        def Sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
            # z is a 1D NumPy array
            # Formula: 1 / (1 + e^(-z))
            # return np.round(your_answer, 5)
            z=1/(1+np.exp(-z))
            return np.round(z,5)
        def Relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
            # z is a 1D NumPy array
            # Formula: max(0, z) element-wise
            z=np.maximum(0,z)
            return z
        dot_prod=np.dot(x,w)
        z=dot_prod+b
        if activation=="sigmoid":
            z=Sigmoid(z)
        if activation=="relu":
            z=np.round(Relu(z),5)
        return z
        
