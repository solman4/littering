import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
model = load_model('D:/Detective_littering/models/best_model.keras')

# Đường dẫn video
video_path = 'D:/Detective_littering/live.mp4'

# Mở video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý khung hình (resize, chuyển đổi định dạng, vv.)
    frame_resized = cv2.resize(frame, (224, 224))  # Kích thước phù hợp với đầu vào mô hình
    img_array = frame_resized / 255.0  # Chuẩn hóa pixel
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

    # Giả sử bạn không cần input phi hình ảnh, nếu có thì thêm vào đây
    non_image_input = np.array([[0] * 108])  # Thay đổi tùy theo input của bạn

    # Dự đoán
    prediction = model.predict([img_array, non_image_input])
    predicted_label = int(prediction[0] > 0.5)

    # Hiển thị kết quả
    label = 'Dung Quy Dinh' if predicted_label == 1 else 'Xa Rac'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
