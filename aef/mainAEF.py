import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
import seaborn as sb

warnings.filterwarnings('ignore')

# Citirea datelor
data = pd.read_csv('../dataIN/dateDSAD.csv', index_col=0)
data = data.fillna(data.mean())

# Standardizarea datelor
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Calcularea matricei de corelație
corr_matrix = scaled_data.corr().abs()

# Selectarea triunghiului superior al matricei de corelație
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identificarea variabilelor cu o corelație mai mare de 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Eliminarea variabilelor puternic corelate
scaled_data = scaled_data.drop(columns=to_drop)

# Testul Bartlett
chi_square_value, p_value = calculate_bartlett_sphericity(scaled_data)
print('Bartlett\'s test statistic:', chi_square_value)
print('Bartlett\'s p-value:', p_value)

# Testul KMO
kmo_all, kmo_model = calculate_kmo(scaled_data)
print('KMO:', kmo_model)

# Salvare matrice indici KMO
matrice_indici_kmo_df = pd.DataFrame(data=kmo_all[:, np.newaxis], index=scaled_data.columns, columns=['Indici KMO'])
matrice_indici_kmo_df.to_csv('../out_aef/Indici_kmo.csv', index_label='Factor de risc')

# Realizare corelograma matrice de corelatie
plt.figure(figsize=(29, 29))
plt.title('Corelograma indicilor KMO', fontsize=40, color='k', verticalalignment='bottom')
sb.heatmap(data=matrice_indici_kmo_df, cmap='bwr', vmin=-1, vmax=1, annot=True)

# Analiza factorilor exploratorii
fa = FactorAnalyzer(25, rotation=None)
fa.fit(scaled_data)

# Folosirea criteriului Kaiser pentru determinarea numarului de factori
# cu eigenvalues mai mari decat 1
ev, v = fa.get_eigenvalues()
print(ev)
numar_factori_semnificativi = ev[ev > 1].shape[0]

fa = FactorAnalyzer(numar_factori_semnificativi, rotation='varimax', method='minres', use_smc=True)
fa.fit(scaled_data)

# Salvarea factor loadings
factor_loadings_df = pd.DataFrame(data=fa.loadings_, index=scaled_data.columns, columns=[f'Factor {i+1}' for i in range(numar_factori_semnificativi)])
factor_loadings_df.to_csv('../out_aef/Factor_loadings.csv', index_label='Factor de risc')

# Creare corelograma a factorilor de corelatie din FA
plt.figure(figsize=(29, 29))
plt.title('Corelograma factorilor de corelație din FA', fontsize=40, color='k', verticalalignment='bottom')
sb.heatmap(data=factor_loadings_df, cmap='bwr', vmin=-1, vmax=1, annot=True)

# Extragere valori proprii din FA
valori_proprii = fa.get_eigenvalues()

# Realizare grafic al variantei explicate de factori din FA
plt.figure(figsize=(29, 29))
plt.title('Varianta explicată de factorii comuni FA', fontsize=20, color='k', verticalalignment='bottom')
plt.xlabel('Componente principale', fontsize=16, color='b', verticalalignment='top')
plt.ylabel('Varianta explicata - valori proprii', fontsize=16, color='b', verticalalignment='bottom')
plt.axhline(y=1, color='r', linestyle='-')
plt.plot([f'C{i+1}' for i in range(len(valori_proprii[1]))], valori_proprii[1], 'bo-')

# Scoruri factoriale
scores = fa.transform(scaled_data)
scores_df = pd.DataFrame(data=scores, index=data.index, columns=[f'Factor {i+1}' for i in range(numar_factori_semnificativi)])
scores_df.to_csv('../out_aef/Scoruri_factoriale.csv', index_label='Țară/Regiune')

# Creare corelograma scorurilor
plt.figure(figsize=(50, 50))
plt.title('Corelograma scorurilor', fontsize=40, color='k', verticalalignment='bottom')
sb.heatmap(data=scores_df, cmap='bwr', vmin=-1, vmax=1, annot=True)

# Realizare cerc al corelațiilor pentru spațiul factorilor
plt.figure(figsize=(22, 22))
plt.title('Cercul corelațiilor', fontsize=20, color='k', verticalalignment='bottom')
T = [t for t in np.arange(0, np.pi * 2, 0.01)]
X = [np.cos(t) for t in T]
Y = [np.sin(t) for t in T]
plt.plot(X, Y)
plt.axhline(y=0, color='g')
plt.axvline(x=0, color='g')
plt.xlabel(factor_loadings_df.columns[0], fontsize=16, color='b', verticalalignment='top')
plt.ylabel(factor_loadings_df.columns[1], fontsize=16, color='b', verticalalignment='bottom')
plt.scatter(factor_loadings_df.iloc[:, 0], factor_loadings_df.iloc[:, 1], color='y')
texts = []
for i in range(factor_loadings_df.shape[0]):
    t = plt.text(factor_loadings_df.iloc[i, 0], factor_loadings_df.iloc[i, 1], factor_loadings_df.index[i], fontsize=10)
    texts.append(t)
plt.show()