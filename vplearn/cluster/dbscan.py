from vplearn._model import InstanceBaseModel

import pandas as pd
import numpy as np

class DBSCAN(InstanceBaseModel):
        
    def __init__(self, epsilon: float, minpts: int):
        super().__init__()
        
        if not isinstance(epsilon, float):
            raise TypeError('epsilon must be an float')
        if not isinstance(minpts, int):
            raise TypeError('minpts must be an integer')
        
        self.epsilon = epsilon
        self.minpts = minpts
        
    
    def fit(self, X: pd.DataFrame) -> None:
        self._check_fit_input_format(X)
        
        X = self._convert_to_numpy(X)
        
        self.n = X.shape[0]
        
        p, q = np.meshgrid(np.arange(self.n), np.arange(self.n))
        self.dist = np.sqrt(np.sum(((X[p] - X[q])**2),2))
        
        self.visited = np.full((self.n), False)
        self.noise = np.full((self.n),False)
        
        self.idx = np.full((self.n),0)
        self.C = 0
        self.input = X
        
        self._run()
        self._sort()
        
        return self.cluster, self.noise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError('DBSCAN does not support predict method')
    
    def _run(self):
        for i in range(len(self.input)):
            if self.visited[i] == False:
                self.visited[i] = True
                self.neighbors = self._regionQuery(i)
                if len(self.neighbors) >= self.minpts:
                    self.C += 1
                    self._expandCluster(i)
                else : self.noise[i] = True
        return self.idx, self.noise

    def _regionQuery(self, i):
        g = self.dist[i,:] < self.epsilon
        Neighbors = np.where(g)[0].tolist()
        return Neighbors

    def _expandCluster(self, i):
        self.idx[i] = self.C
        k = 0
       
        while True:
            if len(self.neighbors) <= k:return
            j = self.neighbors[k]
            if self.visited[j] != True:
                self.visited[j] = True

                self.neighbors2 = self._regionQuery(j)
                v = [self.neighbors2[i] for i in np.where(self.idx[self.neighbors2]==0)[0]]

                if len(self.neighbors2) >=  self.minpts:
                    self.neighbors = self.neighbors+v

            if self.idx[j] == 0 : self.idx[j] = self.C
            k += 1
            
    def _sort(self):
        
        cnum = np.max(self.idx)
        self.cluster = []
        self.noise = []
        for i in range(cnum):
           
            k = np.where(self.idx == (i+1))[0].tolist()
            self.cluster.append([self.input[k,:]])
       
        self.noise = self.input[np.where(self.idx == 0)[0].tolist(),:]
        return self.cluster, self.noise