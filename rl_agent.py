import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Wczytaj dane
df = pd.read_csv("./robots/mybots/DataCollectingBot.data/battle_data.csv")

# 2. Przygotuj dane wejściowe i etykiety
X = df.drop("hit", axis=1).values  # cechy
y = df["hit"].values               # klasy: 0 lub 1

# 3. Podziel dane na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Konwersja na tensory PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 6. Definicja prostej sieci neuronowej
class ShootingNet(nn.Module):
    def __init__(self):
        super(ShootingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # wynik: prawdopodobieństwo trafienia
        )

    def forward(self, x):
        return self.model(x)

# 7. Inicjalizacja modelu, funkcji kosztu i optymalizatora
model = ShootingNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Trening
epochs = 30
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 9. Ewaluacja
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    predicted_classes = (preds > 0.5).float()
    accuracy = (predicted_classes.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
    print(f"\nAccuracy on test set: {accuracy*100:.2f}%")

# 10. Wykres lossów
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# 11. Zapis modelu i skalera
torch.save(model.state_dict(), "shooting_model.pt")
import joblib
joblib.dump(scaler, "scaler.pkl")