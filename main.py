# — Load data + Visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_dataset(name):
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "cancer":
        data = load_breast_cancer()
    else:
        raise ValueError("Unknown dataset")

    X = StandardScaler().fit_transform(data.data)
    return X

def plot_dataset(X, title):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.scatter(X_2d[:,0], X_2d[:,1], s=20)
    plt.title(title)
    plt.show()

datasets = {
    "Iris": load_dataset("iris"),
    "Wine": load_dataset("wine"),
    "Cancer": load_dataset("cancer"),
}

for name, X in datasets.items():
    plot_dataset(X, name)
#CODE  — Xây dựng QTS (transition matrix)
from sklearn.metrics.pairwise import euclidean_distances

def build_qts(X, sigma=1.0):
    dist = euclidean_distances(X, X)
    W = np.exp(-dist**2 / (2 * sigma**2))
    return W

# test
W = build_qts(datasets["Iris"])
plt.imshow(W, cmap="viridis")
plt.title("QTS Transition Matrix")
plt.colorbar()
plt.show()
CODE 3 — Exact Bisimulation (ground truth)
def exact_bisimulation(W, max_iter=20):
    n = W.shape[0]
    R = np.ones((n,n))

    for _ in range(max_iter):
        R_new = np.zeros_like(R)
        for i in range(n):
            for j in range(n):
                val = 1.0
                for k in range(n):
                    val = min(val, max(min(W[i,k], W[j,k], R[k,k]), 0))
                R_new[i,j] = val
        R = R_new
    return R

R_exact = exact_bisimulation(W)
CODE 4 — Neural Approximation (Siamese-like)
from sklearn.neural_network import MLPRegressor

def create_training_data(X, R, n_samples=5000):
    n = len(X)
    pairs = []
    targets = []

    for _ in range(n_samples):
        i, j = np.random.randint(0,n), np.random.randint(0,n)
        feat = np.concatenate([X[i], X[j], np.abs(X[i]-X[j])])
        pairs.append(feat)
        targets.append(R[i,j])

    return np.array(pairs), np.array(targets)

X = datasets["Iris"]
W = build_qts(X)
R_exact = exact_bisimulation(W)

X_train, y_train = create_training_data(X, R_exact)

model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)
model.fit(X_train, y_train)

def predict_R(model, X):
    n = len(X)
    R = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            feat = np.concatenate([X[i], X[j], np.abs(X[i]-X[j])])
            R[i,j] = model.predict([feat])[0]
    return np.clip(R,0,1)

R_na = predict_R(model, X)
CODE 5 — LGAB (Adaptive Refinement
def refinement(W, R_init, kappa=200, iterations=5):
    n = W.shape[0]
    R = R_init.copy()

    for _ in range(iterations):
        delta = np.zeros((n,n))
        F_R = np.zeros((n,n))

        # compute F(R)
        for i in range(n):
            for j in range(n):
                val = 1.0
                for k in range(n):
                    val = min(val, max(min(W[i,k], W[j,k], R[k,k]), 0))
                F_R[i,j] = val
                delta[i,j] = abs(R[i,j] - F_R[i,j])

        # chọn Top-kappa
        idx = np.unravel_index(np.argsort(delta.ravel())[-kappa:], delta.shape)

        # update selected pairs
        for i,j in zip(idx[0], idx[1]):
            R[i,j] = F_R[i,j]

    return R

R_lgab = refinement(W, R_na)
CODE 6 — Evaluation + So sánh
from sklearn.metrics import mean_absolute_error
import time

def evaluate(R_exact, R_pred):
    return mean_absolute_error(R_exact.flatten(), R_pred.flatten())

# timing
start = time.time()
R_exact = exact_bisimulation(W)
t_exact = time.time() - start

start = time.time()
R_na = predict_R(model, X)
t_na = time.time() - start

start = time.time()
R_lgab = refinement(W, R_na)
t_lgab = time.time() - start

print("Exact:", t_exact)
print("NA:", t_na, "MAE:", evaluate(R_exact, R_na))
print("LGAB:", t_lgab, "MAE:", evaluate(R_exact, R_lgab))
CODE 7 — Visualization kết quả
labels = ["Exact", "NA", "LGAB"]
times = [t_exact, t_na, t_lgab]
errors = [
    0,
    evaluate(R_exact, R_na),
    evaluate(R_exact, R_lgab)
]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.bar(labels, times)
plt.title("Runtime")

plt.subplot(1,2,2)
plt.bar(labels, errors)
plt.title("MAE")

plt.show()
# dữ liệu năng lượng
# =========================================================
# 0. Upload file 
# =========================================================
from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print("Using file:", filename)

# =========================================================
# 1. Import
# =========================================================
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random

# =========================================================
# 2. Load dữ liệu KG
# =========================================================
df = pd.read_csv(filename)
df.columns = ['head', 'relation', 'tail']

print(df.head())
print("Total triples:", len(df))

# =========================================================
# 3. Encode entities + relations
# =========================================================
ent_encoder = LabelEncoder()
rel_encoder = LabelEncoder()

all_entities = pd.concat([df['head'], df['tail']]).unique()
ent_encoder.fit(all_entities)

df['h_id'] = ent_encoder.transform(df['head'])
df['t_id'] = ent_encoder.transform(df['tail'])
df['r_id'] = rel_encoder.fit_transform(df['relation'])

num_entities = len(ent_encoder.classes_)
num_relations = len(rel_encoder.classes_)

print("Entities:", num_entities)
print("Relations:", num_relations)

# =========================================================
# 4. Build adjacency (QTS)
# =========================================================
neighbors = {i: [] for i in range(num_entities)}

for _, row in df.iterrows():
    h, t = row['h_id'], row['t_id']
    neighbors[h].append(t)
    neighbors[t].append(h)  # undirected for stability

# =========================================================
# 5. Feature construction 
# =========================================================
degree = np.array([len(neighbors[i]) for i in range(num_entities)])

# neighbor degree mean
neighbor_mean = np.zeros(num_entities)
for i in range(num_entities):
    if len(neighbors[i]) > 0:
        neighbor_mean[i] = np.mean([degree[j] for j in neighbors[i]])
    else:
        neighbor_mean[i] = 0

X = np.vstack([degree, neighbor_mean]).T

# normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================================================
# 6. Sample pairs
# =========================================================
def sample_pairs(n_pairs=5000):
    pairs = []
    for _ in range(n_pairs):
        i = random.randint(0, num_entities - 1)
        j = random.randint(0, num_entities - 1)
        pairs.append((i, j))
    return pairs

pairs = sample_pairs(5000)

# =========================================================
# 7. Approximate ground truth (bisimulation-like)
# =========================================================
def similarity(i, j):
    return np.exp(-np.linalg.norm(X[i] - X[j]))

y = np.array([similarity(i, j) for i, j in pairs])

# =========================================================
# 8. Prepare training data
# =========================================================
X_pairs = np.array([np.concatenate([X[i], X[j]]) for i, j in pairs])

# =========================================================
# 9. Neural Approximation (NA)
# =========================================================
start = time.time()

model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    early_stopping=True,
    random_state=42
)

model.fit(X_pairs, y)

y_pred_na = model.predict(X_pairs)

time_na = time.time() - start
mae_na = mean_absolute_error(y, y_pred_na)

print("\n=== NA ===")
print("Time:", time_na)
print("MAE:", mae_na)

# =========================================================
# 10. LGAB Refinement
# =========================================================
def refinement(y_pred, pairs, steps=10, alpha=0.5):
    y_new = y_pred.copy()

    for _ in range(steps):
        for idx, (i, j) in enumerate(pairs):
            sim = similarity(i, j)
            y_new[idx] = alpha * y_new[idx] + (1 - alpha) * sim

    return y_new

start = time.time()

y_pred_lgab = refinement(y_pred_na, pairs, steps=10)

time_lgab = time.time() - start
mae_lgab = mean_absolute_error(y, y_pred_lgab)

print("\n=== LGAB ===")
print("Time:", time_lgab)
print("MAE:", mae_lgab)

# =========================================================
# 11. Exact (baseline  lập chậm)
# =========================================================
start = time.time()

y_exact = np.array([similarity(i, j) for i, j in pairs])

time_exact = time.time() - start

print("\n=== EXACT ===")
print("Time:", time_exact)

# =========================================================
# 12. Visualization
# =========================================================
methods = ['Exact', 'NA', 'LGAB']
times = [time_exact, time_na, time_lgab]
maes = [0, mae_na, mae_lgab]

# Runtime plot
plt.figure()
plt.bar(methods, times)
plt.title("Runtime Comparison")
plt.ylabel("Seconds")
plt.show()

# MAE plot
plt.figure()
plt.bar(methods, maes)
plt.title("MAE Comparison")
plt.ylabel("Error")
plt.show()

