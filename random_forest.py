import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Wczytanie danych (możesz zmodyfikować ścieżki)
df = pd.read_csv("./robots/mybots/DataCollectingBot.data/battle_data.csv")

# Przygotowanie danych
X = df.drop("hit", axis=1).values
y = df["hit"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trening lasu losowego
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc * 100:.2f}%")

# Zapis modelu i skalera
joblib.dump(clf, "forest_model.pkl")
joblib.dump(scaler, "forest_scaler.pkl")
