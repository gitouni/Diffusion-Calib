import geatpy as ea
import numpy as np
from typing import Callable, List, Union
from joblib import Parallel, delayed
from typing import Optional
class MyProblem(ea.Problem):
    def __init__(self, lb:List, ub:List, name:str, loss_fn:Callable, cv_loss_fn:Union[Callable,None], requires_multporc=False):
        M = 1 # objective dim
        maxormins = [1]  # minimize
        Dim = len(lb)
        varTypes = [0] * Dim # continuous
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        if not requires_multporc:
            self.loss_fn = loss_fn
            self.cv = cv_loss_fn
        else:
            self.loss_fn = lambda x_list: Parallel(n_jobs=-1)(delayed(loss_fn)(x) for x in x_list)
            if cv_loss_fn is not None:
                self.cv = lambda x_list: Parallel(n_jobs=-1)(delayed(cv_loss_fn)(x) for x in x_list)
            else:
                self.cv = None
        self.multiproc = requires_multporc

    def evalVars(self, Vars):
        f = self.loss_fn(Vars)
        if self.cv is not None:
            CV = self.cv(Vars)
            return np.array(f).reshape(-1, 1), np.array(CV).reshape(-1, 1)
        else:
            return np.array(f).reshape(-1, 1)
        
    def calReferObjV(self):
        return np.array([[0.0]])
    

def ea_optimize(lb:List, ub:List, x0:np.ndarray, name:str, loss_fn:Callable, cv_loss_fn:Union[Callable,None], NIND:int, MAXGEN:int, requires_multiproc:bool, logTras:Optional[int]=None):
    problem = MyProblem(lb, ub, name, loss_fn, cv_loss_fn, requires_multiproc)
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=NIND),
        MAXGEN=MAXGEN,
        logTras=logTras, 
        trappedValue=1e-6, 
        maxTrappedCount=10
        )
    if np.ndim(x0) == 1:
        prophet = x0[None,:]
    else:
        prophet = x0
    res = ea.optimize(algorithm,
                        prophet=prophet,
                        verbose=False,
                        drawing=0,
                        outputMsg=False,
                        drawLog=False,
                        saveFlag=False)
    return res
