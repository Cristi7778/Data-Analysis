import numpy as np

class ACP:
    def __init__(self, X):
        # asumam ca X este numpy.ndarray cu valori standardizate
        self.X = X
        # calcul matricea de varianta-covarianta penttru X
        self.Cov = np.cov(X, rowvar=False)  # variabilele sunt pe coloane
        print(self.Cov.shape)
        # calcul valori proprii si vectori proprii pentru matricea de varianta-covarianta
        self.valoriProprii, self.vectoriProprii = np.linalg.eigh(self.Cov)
        # sortare descrescatoare valori si vectori proprii
        k_desc = [k for k in reversed(np.argsort(self.valoriProprii))]
        print(k_desc)
        self.alpha = self.valoriProprii[k_desc]
        self.A = self.vectoriProprii[:, k_desc]
        # regularizare vectori proprii
        for j in range(self.A.shape[1]):
            minCol = np.min(self.A[:, j])
            maxCol = np.max(self.A[:, j])
            if np.abs(minCol) > np.abs(maxCol):
                self.A[:, j] = -self.A[:, j]

        # calcul componente principale
        # self.C = self.X @ self.A
        self.C = np.matmul(self.X, self.A)  # altertiva de inmultire matriceala in numpy
        # calcul corelatie dintre variabilelel initiale si componentele principale
        self.Rxc = self.A * np.sqrt(self.alpha)
        # calitatea reprezantarii observatiilor
        self.C2 = self.C * self.C
        # self.C2 = np.square(self.C)



    def getAlpha(self):
        # return self.valoriProprii
        return self.alpha

    def getA(self):
        # return self.vectoriProprii
        return self.A
    def getComponente(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc
    def getScoruri(self):
        return self.C / np.sqrt(self.alpha)
    def getCalObs(self):
        C2SL = np.sum(self.C2, axis=1)  # sume pe linii
        return np.transpose(self.C2.T / C2SL)
    def getContribObs(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getComunalitati(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)  # suma cumulativa pe linii