import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
data = np.genfromtxt('./data/breast-cancer-wisconsin.data', delimiter=',', dtype=np.int32)

# Separar os atributos das classes
x = data[:, 1:-1]
y = data[:, -1]


# Pré-processamento dos dados
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Criar o classificador MLP
mlp = MLPClassifier(hidden_layer_sizes=(6,6),
                    max_iter=1000, random_state=42,verbose=True)

# Treinar o classificador
mlp.fit(X_train, y_train)

# Avaliar o desempenho do classificador
accuracy = mlp.score(X_test, y_test)
print(f"Acurácia: {accuracy}")



# Fazer previsões
y_pred = mlp.predict(X_test)

# Calcular a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(confusion)

# Criar um dataframe com a matriz de confusão
confusion_df = pd.DataFrame(confusion, index=['Classe Negativa', 'Classe Positiva'], columns=['Classe Negativa (Prev.)', 'Classe Positiva (Prev.)'])


# Plotar a matriz de confusão como um gráfico de calor
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Rótulo Previsto')
plt.ylabel('Rótulo Real')
plt.show(block=False)

# Calcular a taxa de verdadeiros positivos e falsos positivos para a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()