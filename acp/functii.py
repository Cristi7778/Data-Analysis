import numpy as np


def standardizare(X):  # asumam ca primim ca parametru un numpy.ndarray
    medii = np.mean(X, axis=0)  # facem media pe coloane
    print(medii.shape)
    abateriStd = np.std(X, axis=0)  # abatere standard pentru fiecare variabila
    return (X - medii) / abateriStd
