from copy import deepcopy
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv("./robots/mybots/DataCollectingBot.data/battle_data.csv")

X = df.drop("hit", axis=1).drop("enemy_y", axis=1).drop("enemy_x", axis=1).values  # cechy
y = df["hit"].values               # klasy: 0 lub 1

def check_attributes_importance(x, y):
    selector = SelectKBest(score_func=f_classif, k='all')  # lub k=5 dla top 5
    X_new = selector.fit_transform(x, y)

    # Zobacz ranking
    scores = selector.scores_
    columns = df.drop("hit", axis=1).columns
    feature_scores = pd.Series(scores, index=columns).sort_values(ascending=False)
    print(feature_scores)

    # Wizualizacja
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores.values, y=feature_scores.index)
    plt.title("ANOVA F-score cech")
    plt.xlabel("F-score")
    plt.grid(True)
    plt.show()


# 2. Przygotuj dane wejściowe i etykiety


# 3. Podziel dane na treningowe i testowe
X_train_to_split, X_test, y_train_to_split, y_test = train_test_split(X, y, test_size=0.2, random_state=39, shuffle=True, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_to_split, y_train_to_split, test_size=0.2, random_state=39, shuffle=True, stratify=y_train_to_split)

# 4. Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# 5. Konwersja na tensory PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 6. Definicja prostej sieci neuronowej
class ShootingNet(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super(ShootingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
    
l2_reg = 1e-4

# 7. Inicjalizacja modelu, funkcji kosztu i optymalizatora
model = ShootingNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)

# 8. Trening
epochs = 1000
losses = []

max_patience = 50
cur_patience = 0
best_loss = 123781
best_model = None

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    
    if loss.item() > best_loss:
        cur_patience += 1
    else:
        best_model = deepcopy(model)
        best_loss = loss.item()
        cur_patience = 0
        
    if cur_patience > max_patience:
        break
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    
model = best_model

def get_optimal_threshold(
    precisions: np.array, 
    recalls: np.array, 
    thresholds: np.array
) -> Tuple[int, float]:
    
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_idx, optimal_threshold
    
def eval_model(
    model: nn.Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    threshold: Optional[float] = None
):
    model.eval()
    with torch.no_grad():
        init_pred = torch.sigmoid(model(x))
    
    auroc = roc_auc_score(y, init_pred)
        
    prec, rec, thresholds = precision_recall_curve(y, init_pred)
    optimal_idx, best_thr = get_optimal_threshold(prec, rec, thresholds)
    
    disp = PrecisionRecallDisplay(prec, rec)
    disp.plot()
    plt.title(f"Precision-recall curve (opt. thresh.: {best_thr:.4f})")
    plt.axvline(rec[optimal_idx], color="green", linestyle="-.")
    plt.axhline(prec[optimal_idx], color="green", linestyle="-.")
    plt.show()

    
    thr = best_thr
    if (threshold is not None):
        thr = threshold
        
    pred = (init_pred>thr).float()
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    accuracy = (pred.eq(y).sum().item()) / y.size(0)
    
    print(f"\nAccuracy on test set: {accuracy*100:.2f}%")
    print(f"AUROC: {100 * auroc:.2f}%")
    print(f"precision: {100*precision:.2f}%")
    print(f"recall: {100*recall:.2f}%")
    print(f"f1: {100*f1:.2f}%")
    print(f"Threshold: {thr}")

eval_model(model, X_valid_tensor, y_valid_tensor)
eval_model(model, X_test_tensor, y_test_tensor, 0.5)

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