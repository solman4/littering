# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121

def build_model(image_shape=(224, 224, 3), non_image_features=108):
    """
    Xây dựng mô hình Deep Learning với hai đầu vào: hình ảnh và các đặc trưng không phải hình ảnh.
    
    Args:
        image_shape (tuple): Kích thước hình ảnh đầu vào (height, width, channels).
        non_image_features (int): Số lượng đặc trưng không phải hình ảnh.
    
    Returns:
        model (tensorflow.keras.Model): Mô hình đã được xây dựng.
    """
    # Nhánh Hình Ảnh (DenseNet121)
    image_input = Input(shape=image_shape, name='image_input')
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = False  # Đóng băng các lớp DenseNet để giữ trọng số đã huấn luyện trước
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Nhánh Các Đặc Trưng Khác (MLP)
    non_image_input = Input(shape=(non_image_features,), name='non_image_input')
    y = Dense(128, activation='relu')(non_image_input)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    
    # Kết Hợp Các Nhánh
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.5)(z)
    
    # Lớp Output
    output = Dense(1, activation='sigmoid')(z)
    
    # Xây dựng mô hình
    model = Model(inputs=[image_input, non_image_input], outputs=output)
    
    # Tóm tắt mô hình
    model.summary()
    
    return model
