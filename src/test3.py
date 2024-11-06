import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load mô hình đã huấn luyện
model = load_model('D:/Detective_littering/models/best_model.keras')

# Hàm tải và tiền xử lý ảnh từ đường dẫn
def load_and_preprocess_image(filepath, target_size=(224, 224)):  # Điều chỉnh target_size theo mô hình của bạn
    img = Image.open(filepath).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Tùy vào cách xử lý ảnh đã sử dụng khi huấn luyện
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Đường dẫn ảnh để dự đoán
filepath = 'D:/Detective_littering/data/raw/incorrect_behavior/images/12.jpg'

# Tải và tiền xử lý ảnh
img_array = load_and_preprocess_image(filepath)

# Tạo dữ liệu giả cho đầu vào không phải hình ảnh
non_image_input = np.zeros((1, 108))  # Điều chỉnh kích thước (10) theo yêu cầu của mô hình của bạn

# Dự đoán
prediction = model.predict([img_array, non_image_input])
predicted_label = (prediction > 0.5).astype(int)

# Hiển thị ảnh và dự đoán
img = Image.open(filepath)
plt.imshow(img)
plt.title(f'Dự đoán: {predicted_label[0]}, Xác suất: {prediction[0][0]:.2f}')
plt.axis('off')
plt.show()
