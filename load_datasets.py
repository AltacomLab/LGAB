import pandas as pd
from google.colab import files
import os

def upload_and_clean_csv():
    """Upload file từ máy, convert TSF/CSV thành CSV chuẩn"""
    uploaded = files.upload()
    csv_file = list(uploaded.keys())[0]
    print("File uploaded:", csv_file)
    
    # Load dữ liệu, tách bằng whitespace
    df = pd.read_csv(csv_file, sep=r'\s+', header=None, engine='python')
    print("Data shape:", df.shape)
    
    # Lưu file sạch
    clean_csv = f"{os.path.splitext(csv_file)[0]}_clean.csv"
    df.to_csv(clean_csv, index=False)
    print("✅ Saved cleaned CSV:", clean_csv)
    
    # Tự động download về máy
    files.download(clean_csv)
    
    return df
