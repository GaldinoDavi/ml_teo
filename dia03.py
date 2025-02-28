#%%
# Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Lendo o dataframe
df = pd.read_excel("dados/dados_cerveja_nota.xlsx")
df

# %%
# Explorando os dados
plt.plot(df['cerveja'], df['nota'], 'o')
plt.grid(True)
plt.title('Nota x Cerveja')
plt.xlabel('Cerveja')
plt.ylabel('Nota')
plt.show()

# %%
# Importando sklearn para trabalhar com regressão
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[['cerveja']], df['nota'])

# %%
print(reg.coef_ , reg.intercept_)

# %%
a, b = reg.intercept_, reg.coef_[0]

# %%
print (f"a = {a} e b = {b}")
# %%
# Realizando a predição
y_estimado = reg.predict(df[['cerveja']])
y_estimado

# %%
# Plotando os pontos + a predição
plt.plot(df['cerveja'], df['nota'], 'o')
plt.plot(df['cerveja'], y_estimado, 'r')
plt.grid(True)
plt.title('Nota x Cerveja')
plt.xlabel('Cerveja')
plt.ylabel('Nota')
plt.show()

# %%
# Utilizando árvore de decisão
from sklearn import tree
arvore = tree.DecisionTreeRegressor(max_depth=2)

# %%
arvore.fit(df[['cerveja']], df['nota'])
# %%
y_estimado_arvore = arvore.predict(df[['cerveja']])
y_estimado_arvore

# %%
# Plotando os pontos + a predição + arvore
plt.plot(df['cerveja'], df['nota'], 'o')
plt.plot(df['cerveja'], y_estimado, 'r')
plt.plot(df['cerveja'], y_estimado_arvore, 'g')
plt.grid(True)
plt.title('Nota x Cerveja')
plt.xlabel('Cerveja')
plt.ylabel('Nota')
plt.legend(['Pontos', 'Regressão', 'Árvore'])
plt.show()

# %%
