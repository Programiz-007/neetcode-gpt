class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        def der_fn(x):
            return 2*x
        ans=init
        while iterations>0:
            der=der_fn(ans)
            ans=ans-(learning_rate*der)
            iterations=iterations-1
        return round(ans,5)


        
