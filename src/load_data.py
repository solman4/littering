import numpy as np
import os

# Đường dẫn đến thư mục dữ liệu đã chuẩn hóa
normalized_path = 'D:/Detective_littering/data/combined_data/'

# Tải dữ liệu
try:
    X_train_image = np.load(os.path.join(normalized_path, 'X_train_image.npy'))
    X_val_image = np.load(os.path.join(normalized_path, 'X_val_image.npy'))
    X_test_image = np.load(os.path.join(normalized_path, 'X_test_image.npy'))
    
    X_train_non_image = np.load(os.path.join(normalized_path, 'X_train_non_image_scaled.npy'))
    X_val_non_image = np.load(os.path.join(normalized_path, 'X_val_non_image_scaled.npy'))
    X_test_non_image = np.load(os.path.join(normalized_path, 'X_test_non_image_scaled.npy'))
    
    y_train = np.load(os.path.join(normalized_path, 'y_train.npy'))
    y_val = np.load(os.path.join(normalized_path, 'y_val.npy'))
    y_test = np.load(os.path.join(normalized_path, 'y_test.npy'))
    
    print("Shapes:")
    print("X_train_image:", X_train_image.shape)
    print("X_train_non_image:", X_train_non_image.shape)
    print("y_train:", y_train.shape)
    print("X_val_image:", X_val_image.shape)
    print("X_val_non_image:", X_val_non_image.shape)
    print("y_val:", y_val.shape)
    print("X_test_image:", X_test_image.shape)
    print("X_test_non_image:", X_test_non_image.shape)
    print("y_test:", y_test.shape)
except FileNotFoundError as e:
    print(f"Error: {e}. Vui lòng đảm bảo rằng các file .npy đã được lưu vào thư mục '{normalized_path}'.")
except Exception as e:
    print(f"Unexpected error: {e}")
