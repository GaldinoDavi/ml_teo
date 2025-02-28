#%%$ agora sim tem celulas
import pandas as pd
# %%
df = pd.read_parquet('C:/Users/Davi/OneDrive/Documentos/ml_basic/dados/dados_clones.parquet')
# %%
df

# %%
df.columns
### 
##'Massa(em kilos)', 
# 'General Jedi encarregado',
# 'Estatura(cm)', 
# 'Distância Ombro a ombro', 
# 'Tamanho do crânio',
# 'Tamanho dos pés', 
# 'Tempo de existência(em meses)', 
# 'Status ',
# 'Status_bool'

# %% Análise descritiva
df['Status_bool'] = df['Status '] == 'Apto'
# %%
df

# %%
# Verificando tamanho do ombro
df.groupby(['Distância Ombro a ombro'])['Status_bool'].mean()

# %%
# Verificando tamanho do crânio
df.groupby(['Tamanho do crânio'])['Status_bool'].mean()

# %%
# Verificando tamanho dos pés
df.groupby(['Tamanho dos pés'])['Status_bool'].mean()

# %%
# Verificando tempo de existência
df.groupby(['Tempo de existência(em meses)'])['Status_bool'].mean()

# %%
# Verificando estatura
df.groupby(['Estatura(cm)'])['Status_bool'].mean()

# %%
# Verificando a massa
df.groupby(['Massa(em kilos)'])['Status_bool'].mean()


# %%
# Verificando general
df.groupby(['General Jedi encarregado'])['Status_bool'].mean()


# %%
## Criando uma lista com as features

features = ['Estatura(cm)','Massa(em kilos)',
             'Distância Ombro a ombro', 'Tamanho do crânio',
               'Tamanho dos pés']
# %%
## Utilizar one-hot encoding para features
from feature_engine import encoding

#%%
df_f = df[features]

# %%
onehot = encoding.OneHotEncoder(variables=['Distância Ombro a ombro', 'Tamanho do crânio',
               'Tamanho dos pés'])
onehot.fit(df_f)
df_f = onehot.transform(df_f)
df_f

# %%
## Aplicando o modelo de machine learning
# Utilizando decision tree

from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier(max_depth=8)
arvore.fit(df_f, df['Status '])

# %%
import matplotlib.pyplot as plt
plt.figure(dpi = 900)

tree.plot_tree(arvore,
               class_names= arvore.classes_,
               feature_names= df_f.columns,
               filled=True)
# %%
