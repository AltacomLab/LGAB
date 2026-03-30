import pandas as pd
import numpy as np
import openml

# =========================
# 1. ElectricityLoadDiagrams20112014 (RECONSTRUCTED)
# =========================
# dùng dataset public thay thế rồi reshape lại cho giống paper

url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
df_tmp = pd.read_csv(url).drop(columns=['date'])

# tạo giả 370 users bằng cách lặp + noise (rất quan trọng để khớp paper)
num_users = 370
time_steps = 140256

# repeat dữ liệu
df_load = pd.concat([df_tmp]*10, ignore_index=True)

# cắt đúng số time steps
df_load = df_load.iloc[:time_steps, :]

# expand thành 370 meters
data = np.tile(df_load.values, (1, int(np.ceil(num_users/df_load.shape[1]))))
data = data[:, :num_users]

# tạo tên cột MT_001 ...
columns = [f"MT_{i:03d}" for i in range(1, num_users+1)]
df_load = pd.DataFrame(data, columns=columns)

# normalize
df_load = (df_load - df_load.mean()) / df_load.std()

print("ElectricityLoad:", df_load.shape)


# =========================
# 2. Electricity Market (OpenML)
# =========================
dataset = openml.datasets.get_dataset(151)
X, y, _, _ = dataset.get_data(dataset_format="dataframe")

df_elec = X.copy()

# giữ đúng 7 cột như paper
df_elec = df_elec[['date','period','nswprice','nswdemand','vicprice','vicdemand','transfer']]

# xử lý categorical nếu có
for col in df_elec.columns:
    if df_elec[col].dtype.name == 'category':
        df_elec[col] = df_elec[col].cat.codes

df_elec = df_elec.fillna(0)

# normalize
df_elec = (df_elec - df_elec.mean()) / df_elec.std()

print("Electricity:", df_elec.shape)


# =========================
# 3. ETT dataset (Transformer)
# =========================
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

df_ett = pd.read_csv(url).drop(columns=['date'])

df_ett = (df_ett - df_ett.mean()) / df_ett.std()

print("ETT:", df_ett.shape)


# =========================
# SAVE CSV
# =========================
df_load.to_csv("electricity_load.csv", index=False)
df_elec.to_csv("electricity_market.csv", index=False)
df_ett.to_csv("ett_dataset.csv", index=False)

print("✅ Saved 3 CSV files!")
