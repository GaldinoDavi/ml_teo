# %%
import pandas as pd

# %%
df = pd.read_excel("dados/dados_cerveja_nota.xlsx")
df

# %%
# Cria uma coluna que informa se foi aprovado ou não
df['Aprovado'] = df['nota'] >= 5
df

# %%
# Aplicando a regressão Logística
from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=None, fit_intercept = True)

features = ['cerveja']
target = 'Aprovado'

# Aqui o modelo aprende (Ajuste de modelo)
reg.fit(df[features], df[target])

reg_predict = reg.predict(df[features])
reg_predict

# %%
from sklearn import metrics

reg_acc = metrics.accuracy_score(df[target], reg_predict)
reg_acc

# %%
