# src/train_model.py

import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

def get_callbacks():
    """
    Thiết lập các callback cho quá trình huấn luyện mô hình.
    
    Returns:
        callbacks (list): Danh sách các callback.
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        filepath='D:/Detective_littering/models/best_model.keras',  # Sử dụng định dạng .keras
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir='D:/Detective_littering/logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    
    return [early_stop, checkpoint, reduce_lr, tensorboard]

def load_data(normalized_path):
    """
    Tải dữ liệu đã chuẩn hóa từ các file .npy.
    
    Args:
        normalized_path (str): Đường dẫn đến thư mục chứa các file .npy.
    
    Returns:
        Tuple: Các mảng dữ liệu đã được tải.
    """
    # Tải dữ liệu hình ảnh
    X_train_image = np.load(os.path.join(normalized_path, 'X_train_image.npy'))
    X_val_image = np.load(os.path.join(normalized_path, 'X_val_image.npy'))
    X_test_image = np.load(os.path.join(normalized_path, 'X_test_image.npy'))
    
    # Tải dữ liệu đặc trưng không phải hình ảnh
    X_train_non_image = np.load(os.path.join(normalized_path, 'X_train_non_image_scaled.npy'))
    X_val_non_image = np.load(os.path.join(normalized_path, 'X_val_non_image_scaled.npy'))
    X_test_non_image = np.load(os.path.join(normalized_path, 'X_test_non_image_scaled.npy'))
    
    # Tải nhãn
    y_train = np.load(os.path.join(normalized_path, 'y_train.npy'))
    y_val = np.load(os.path.join(normalized_path, 'y_val.npy'))
    y_test = np.load(os.path.join(normalized_path, 'y_test.npy'))
    
    return X_train_image, X_train_non_image, y_train, X_val_image, X_val_non_image, y_val, X_test_image, X_test_non_image, y_test

def shuffle_data(X_image, X_non_image, y):
    """
    Xáo trộn dữ liệu theo cùng một thứ tự.
    
    Args:
        X_image (np.ndarray): Dữ liệu hình ảnh.
        X_non_image (np.ndarray): Dữ liệu đặc trưng không phải hình ảnh.
        y (np.ndarray): Nhãn.
    
    Returns:
        Tuple: Dữ liệu đã được xáo trộn.
    """
    assert len(X_image) == len(X_non_image) == len(y), "All inputs must have the same length."
    indices = np.arange(len(X_image))
    np.random.seed(42)  # Đảm bảo tính tái lập
    np.random.shuffle(indices)
    return X_image[indices], X_non_image[indices], y[indices]

def combined_generator(image_data_gen, X_image, X_non_image, y, batch_size=32):
    """
    Generator kết hợp image generator và non-image data.
    
    Args:
        image_data_gen (ImageDataGenerator): ImageDataGenerator instance.
        X_image (np.ndarray): Dữ liệu hình ảnh.
        X_non_image (np.ndarray): Dữ liệu đặc trưng không phải hình ảnh.
        y (np.ndarray): Nhãn.
        batch_size (int): Kích thước batch.
    
    Yields:
        Tuple: (inputs_dict, label_batch)
    """
    image_gen = image_data_gen.flow(X_image, y, batch_size=batch_size, shuffle=False)
    num_samples = len(X_image)
    steps = np.ceil(num_samples / batch_size)
    
    while True:  # Đảm bảo generator loop vô hạn
        for step in range(int(steps)):
            img_batch, label_batch = next(image_gen)
            batch_start = step * batch_size
            batch_end = batch_start + batch_size
            non_img_batch = X_non_image[batch_start:batch_end]
            # Trả về dưới dạng dictionary để khớp với tên đầu vào của mô hình
            yield {'image_input': img_batch, 'non_image_input': non_img_batch}, label_batch

def main():
    # Đường dẫn tới dữ liệu đã chuẩn hóa
    normalized_path = 'D:/Detective_littering/data/combined_data/'
    
    # Tải dữ liệu
    X_train_image, X_train_non_image, y_train, X_val_image, X_val_non_image, y_val, X_test_image, X_test_non_image, y_test = load_data(normalized_path)
    
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
    
    # Xáo trộn dữ liệu
    X_train_image, X_train_non_image, y_train = shuffle_data(X_train_image, X_train_non_image, y_train)
    X_val_image, X_val_non_image, y_val = shuffle_data(X_val_image, X_val_non_image, y_val)
    
    # Xây dựng mô hình
    model = build_model(image_shape=(224, 224, 3), non_image_features=X_train_non_image.shape[1])
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    # Thiết lập callbacks
    callbacks = get_callbacks()
    
    # Data Augmentation cho hình ảnh huấn luyện
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()  # Không tăng cường dữ liệu cho validation
    
    # Tạo custom generators kết hợp cả hình ảnh và các đặc trưng không phải hình ảnh
    train_generator = combined_generator(train_datagen, X_train_image, X_train_non_image, y_train, batch_size=32)
    val_generator = combined_generator(val_datagen, X_val_image, X_val_non_image, y_val, batch_size=32)
    
    # Tính số bước mỗi epoch và validation
    steps_per_epoch = int(np.ceil(len(X_train_image) / 32))  # Chuyển đổi float thành int
    validation_steps = int(np.ceil(len(X_val_image) / 32))  # Chuyển đổi float thành int
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=100,
        callbacks=callbacks
    )
    
    # Lưu lịch sử huấn luyện
    np.save(os.path.join(normalized_path, 'history.npy'), history.history)
    
    # Đánh giá mô hình trên tập test
    print("\nEvaluating model on test data:")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        {'image_input': X_test_image, 'non_image_input': X_test_non_image},
        y_test
    )
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")

if __name__ == "__main__":
    main()
