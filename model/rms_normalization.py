import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        sum=0
        for i in range(len(x)):
            sum=sum+(x[i]**2)
        s=(1/len(x))*(sum)
        s=s+eps
        rms=s**0.5
        out=gamma
        for i in range(len(x)):
            x[i]=x[i]/rms
            out[i]=out[i]*x[i]
        
        return np.round(out,4)
        
        
        
