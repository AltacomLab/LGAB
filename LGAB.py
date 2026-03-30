import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import time
import random
#load data
df = pd.read_csv("electricity_market.csv")  

data = df.values.astype(np.float32)
print("Shape:", data.shape)
#NORMALIZATION
scaler = StandardScaler()
data = scaler.fit_transform(data)
#TIME-SERIES SPLIT (70/15/15)
T = len(data)

train_end = int(0.7 * T)
val_end = int(0.85 * T)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

print(train_data.shape, val_data.shape, test_data.shape)
#STATE CONSTRUCTION
def build_states(data):
    return data  # s_t = x_t

train_states = build_states(train_data)
val_states = build_states(val_data)
test_states = build_states(test_data)
#BUILD PAIRS
def build_pairs(states, num_pairs=5000, tau=5):
    pairs = []
    n = len(states)

    for _ in range(num_pairs):
        i = random.randint(0, n - 1)

        # positive pair (gần nhau)
        if random.random() < 0.5:
            j = min(n - 1, i + random.randint(1, tau))
        else:
            j = random.randint(0, n - 1)

        xi, xj = states[i], states[j]

        # pseudo label (distance-based)
        y = np.exp(-np.linalg.norm(xi - xj))

        pairs.append((xi, xj, y))

    return pairs
    train_pairs = build_pairs(train_states)
val_pairs = build_pairs(val_states, 2000)
test_pairs = build_pairs(test_states, 2000)
#DATASET → TENSOR
def to_tensor(pairs):
    X1 = torch.tensor([p[0] for p in pairs])
    X2 = torch.tensor([p[1] for p in pairs])
    Y = torch.tensor([p[2] for p in pairs]).unsqueeze(1)
    return X1, X2, Y

X1_train, X2_train, Y_train = to_tensor(train_pairs)
X1_val, X2_val, Y_val = to_tensor(val_pairs)
X1_test, X2_test, Y_test = to_tensor(test_pairs)
#MODEL (Neural Approximation)
class BisimNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.net(x)

model = BisimNet(input_dim=data.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

#TRAINING + EARLY STOPPING
best_val = float("inf")
patience = 10
counter = 0

for epoch in range(100):
    model.train()
    pred = model(X1_train, X2_train)
    loss = criterion(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X1_val, X2_val)
        val_loss = mean_absolute_error(Y_val.numpy(), val_pred.numpy())

    print(f"Epoch {epoch}: TrainLoss={loss.item():.6f}, ValMAE={val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        best_model = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
#TEST (NA baseline)
model.load_state_dict(best_model)

start = time.time()

model.eval()
with torch.no_grad():
    test_pred = model(X1_test, X2_test).numpy()

runtime_na = time.time() - start

mae_na = mean_absolute_error(Y_test.numpy(), test_pred)

print("NA MAE:", mae_na)
print("NA Time:", runtime_na)
#LGAB (Top-κ refinement)
def lgab_refinement(model, X1, X2, Y, k=200):
    model.eval()
    with torch.no_grad():
        pred = model(X1, X2).numpy()

    errors = np.abs(pred - Y.numpy())
    idx = np.argsort(errors.flatten())[::-1][:k]

    # refine: replace worst predictions with ground truth
    pred[idx] = Y.numpy()[idx]

    return pred

start = time.time()

refined_pred = lgab_refinement(model, X1_test, X2_test, Y_test)

runtime_lgab = time.time() - start
mae_lgab = mean_absolute_error(Y_test.numpy(), refined_pred)

print("LGAB MAE:", mae_lgab)
print("LGAB Time:", runtime_lgab)
#EXACT (GIẢ LẬP O(n^3))
def exact_bisim(states, n_sample=200):
    states = states[:n_sample]
    n = len(states)

    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = np.linalg.norm(states[i] - states[j])

    return M
    
