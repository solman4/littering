# src/inference/video_processing.py

import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from src.feature_extraction import extract_features_from_keypoints  # Đảm bảo rằng feature_extraction.py có thể được import
import os

def load_resources(model_path, scaler_path, object_detection_model_path=None):
    """
    Tải mô hình đã huấn luyện, scaler và mô hình phát hiện đối tượng.
    
    Args:
        model_path (str): Đường dẫn tới mô hình Keras đã lưu.
        scaler_path (str): Đường dẫn tới scaler đã lưu.
        object_detection_model_path (str): Đường dẫn tới mô hình phát hiện đối tượng (YOLO).
    
    Returns:
        model: Mô hình Keras đã tải.
        scaler: Scaler đã tải.
        yolo_model: Mô hình YOLO đã tải.
    """
    # Tải mô hình Keras
    model = load_model(model_path)
    print("Đã tải mô hình Keras.")
    
    # Tải scaler
    scaler = joblib.load(scaler_path)
    print("Đã tải scaler.")
    
    # Tải mô hình YOLOv5 (Ultralytics)
    if object_detection_model_path:
        yolo_model = YOLO(object_detection_model_path)
    else:
        # Sử dụng mô hình mặc định (có thể thay đổi tùy theo nhu cầu)
        yolo_model = YOLO('yolov5s.pt')  # Thay bằng đường dẫn mô hình của bạn nếu cần
    print("Đã tải mô hình YOLO.")
    
    return model, scaler, yolo_model

def initialize_pose_detector():
    """
    Khởi tạo MediaPipe Pose.
    
    Returns:
        pose: Đối tượng Pose của MediaPipe.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    return pose

def extract_pose_keypoints(pose, frame):
    """
    Trích xuất keypoints từ một khung hình sử dụng MediaPipe Pose.
    
    Args:
        pose: Đối tượng Pose của MediaPipe.
        frame: Khung hình (image) từ video.
    
    Returns:
        keypoints: Mảng numpy chứa keypoints (33 landmarks x 3 giá trị).
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    else:
        # Trả về mảng mặc định nếu không tìm thấy pose
        return np.zeros(33 * 3)

def detect_objects(yolo_model, frame, classes=['person', 'bin', 'xarac_item']):
    """
    Phát hiện đối tượng trong khung hình sử dụng mô hình YOLO.
    
    Args:
        yolo_model: Mô hình YOLO đã tải.
        frame: Khung hình (image) từ video.
        classes (list): Danh sách các lớp đối tượng cần phát hiện.
    
    Returns:
        detections: Danh sách các phát hiện với thông tin bounding box và lớp.
    """
    results = yolo_model.predict(source=frame, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])  # Sửa đổi để lấy phần tử đầu tiên
            conf = box.conf.cpu().numpy()[0]
            cls_name = result.names[cls]
            if cls_name in classes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'label': cls_name,
                    'confidence': float(conf)
                })
    print(f"Detected objects: {detections}")  # Thêm dòng này để kiểm tra
    return detections

def preprocess_image(person_crop):
    """
    Xử lý hình ảnh người để phù hợp với đầu vào của mô hình.
    
    Args:
        person_crop (numpy.ndarray): Hình ảnh người được cắt từ khung hình.
    
    Returns:
        preprocessed_image (numpy.ndarray): Hình ảnh đã được xử lý.
    """
    # Resize hình ảnh thành 224x224
    image_resized = cv2.resize(person_crop, (224, 224))
    
    # Chuyển đổi màu BGR sang RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa giá trị pixel
    image_normalized = image_rgb / 255.0
    
    # Mở rộng chiều để phù hợp với batch size (1, 224, 224, 3)
    preprocessed_image = np.expand_dims(image_normalized, axis=0)
    
    return preprocessed_image

def annotate_frame(frame, detections, predictions, threshold=0.5):
    """
    Vẽ bounding boxes và nhãn lên khung hình.
    
    Args:
        frame: Khung hình (image) từ video.
        detections (list): Danh sách các đối tượng được phát hiện.
        predictions (dict): Dự đoán hành vi cho từng người.
        threshold (float): Ngưỡng xác suất để xác định hành vi.
    
    Returns:
        frame: Khung hình đã được đánh dấu.
    """
    for person_id, pred in predictions.items():
        if person_id >= len(detections):
            continue  # Đảm bảo person_id không vượt quá danh sách detections
        person = detections[person_id]
        x, y, w, h = person['bbox']
        label = 'Littering' if pred >= threshold else 'Correct'
        color = (0, 0, 255) if pred >= threshold else (0, 255, 0)
        confidence = pred
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Vẽ nhãn
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_video(input_video_path, output_video_path, model, scaler, yolo_model, pose):
    """
    Xử lý video để phát hiện và đánh dấu hành vi xả rác.
    
    Args:
        input_video_path (str): Đường dẫn tới video đầu vào.
        output_video_path (str): Đường dẫn tới video đầu ra.
        model: Mô hình Keras đã tải.
        scaler: Scaler đã tải.
        yolo_model: Mô hình YOLO đã tải.
        pose: Đối tượng Pose của MediaPipe.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Mở video đầu vào
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {input_video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video thông số - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # Định nghĩa codec và tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Hoặc 'XVID', 'MJPG'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    total_littering_events = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        print(f"Đang xử lý khung hình: {frame_number}/{total_frames}")
        
        # Phát hiện đối tượng
        detections = detect_objects(yolo_model, frame)
        print(f"Detections in frame {frame_number}: {detections}")  # Kiểm tra detections
        
        # Lấy danh sách người
        persons = [det for det in detections if det['label'] == 'person']
        print(f"Number of persons detected: {len(persons)}")  # Kiểm tra số lượng người
        
        predictions = {}
        
        for idx, person in enumerate(persons):
            x, y, w, h = person['bbox']
            person_crop = frame[y:y+h, x:x+w]
            
            # Trích xuất keypoints
            keypoints = extract_pose_keypoints(pose, person_crop)
            print(f"Keypoints for person {idx}: {keypoints}")  # Kiểm tra keypoints
            
            # Kiểm tra nếu keypoints là mảng mặc định (chứa toàn 0)
            if np.all(keypoints == 0):
                print(f"Không phát hiện pose cho người {idx} trong khung hình {frame_number}. Bỏ qua.")
                continue
            
            # Trích xuất đặc trưng
            try:
                features = extract_features_from_keypoints(keypoints)
                features = np.array(features)  # Chuyển list thành numpy array nếu cần
            except ValueError as ve:
                print(f"Error extracting features for person {idx} in frame {frame_number}: {ve}")
                continue
            
            # Kiểm tra số lượng đặc trưng
            if len(features) != 108:
                print(f"Đặc trưng không đúng kích thước cho người {idx} trong khung hình {frame_number}. Có {len(features)} đặc trưng.")
                continue
            
            # Chuẩn hóa đặc trưng
            features_scaled = scaler.transform([features])
            
            # Xử lý đầu vào hình ảnh
            preprocessed_image = preprocess_image(person_crop)
            
            # Dự đoán hành vi với cả hai đầu vào
            prediction = model.predict([preprocessed_image, features_scaled])
            
            # Xử lý kết quả dự đoán
            if isinstance(prediction, list):
                prediction = prediction[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0][0]
            predictions[idx] = prediction
            
            if prediction >= 0.5:
                total_littering_events += 1
        
        # Annotate frame
        annotated_frame = annotate_frame(frame, persons, predictions)
        
        # Thêm timestamp
        current_time = frame_number / fps
        cv2.putText(annotated_frame, f"Time: {current_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ghi khung hình đã được đánh dấu vào video đầu ra
        out.write(annotated_frame)
        
        # Hiển thị khung hình (tuỳ chọn)
        # Nếu bạn không cần hiển thị video trong quá trình chạy script, bạn có thể comment các dòng sau
        # cv2.imshow('Littering Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    print(f"Total Littering Events Detected: {total_littering_events}")
    
    # Giải phóng các đối tượng
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # Gán trực tiếp các đường dẫn
    input_video_path = "D:/Detective_littering/input_videos/live.mp4"
    output_video_path = "D:/Detective_littering/output_videos/output_video_annotated.mp4"
    model_path = "models/best_model.keras"
    scaler_path = "models/scaler_combined.joblib"
    yolo_model_path = "yolov5su.pt"  # Đảm bảo rằng tệp này tồn tại ở thư mục dự án hoặc chỉ định đường dẫn đầy đủ
    
    # Kiểm tra các đường dẫn
    if not os.path.exists(input_video_path):
        print(f"Video đầu vào không tồn tại: {input_video_path}")
        return
    if not os.path.exists(model_path):
        print(f"Mô hình Keras không tồn tại: {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"Scaler không tồn tại: {scaler_path}")
        return
    if not os.path.exists(yolo_model_path):
        print(f"Mô hình YOLO không tồn tại: {yolo_model_path}")
        return
    
    # Tải mô hình, scaler và YOLO
    model, scaler, yolo_model = load_resources(model_path, scaler_path, yolo_model_path)
    
    # Kiểm tra cấu trúc mô hình
    print("Model Inputs:", model.inputs)
    if len(model.inputs) != 2:
        print(f"Mô hình yêu cầu {len(model.inputs)} đầu vào, nhưng bạn đang cung cấp 2. Vui lòng điều chỉnh mã nguồn để cung cấp đầy đủ các đầu vào.")
        return
    
    # Khởi tạo pose detector
    pose = initialize_pose_detector()
    
    # Xử lý video
    process_video(input_video_path, output_video_path, model, scaler, yolo_model, pose)
    
    print("Đã hoàn tất việc xử lý video.")

if __name__ == "__main__":
    main()
