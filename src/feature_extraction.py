# src/feature_extraction.py

import os
import numpy as np

def extract_features_from_keypoints(keypoints):
    """
    Trích xuất 108 đặc trưng từ keypoints.

    Args:
        keypoints (numpy.ndarray): Mảng keypoints (33 landmarks x 3 giá trị).

    Returns:
        normalized_features (list): Danh sách các đặc trưng đã được chuẩn hóa.
    """
    keypoints = np.array(keypoints).reshape(-1, 3)  # 33 landmarks x 3
    features = []

    # 1. Bao gồm keypoints gốc (99 đặc trưng)
    raw_keypoints = keypoints.flatten()
    features.extend(raw_keypoints)

    # 2. Các đặc trưng bổ sung (9 đặc trưng)
    # Lấy các điểm khớp quan trọng
    nose = keypoints[0]
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    left_elbow = keypoints[13]
    right_elbow = keypoints[14]
    left_wrist = keypoints[15]
    right_wrist = keypoints[16]
    left_hip = keypoints[23]
    right_hip = keypoints[24]
    left_knee = keypoints[25]
    right_knee = keypoints[26]
    left_ankle = keypoints[27]
    right_ankle = keypoints[28]

    # Tính toán khoảng cách
    def calculate_distance(a, b):
        return np.linalg.norm(a - b)

    # Khoảng cách giữa hai vai
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    features.append(shoulder_width)

    # Khoảng cách giữa hông và đầu gối trái
    left_leg_length = calculate_distance(left_hip, left_knee)
    features.append(left_leg_length)

    # Khoảng cách giữa hông và đầu gối phải
    right_leg_length = calculate_distance(right_hip, right_knee)
    features.append(right_leg_length)

    # Khoảng cách từ hông đến cổ tay trái
    left_arm_reach = calculate_distance(left_hip, left_wrist)
    features.append(left_arm_reach)

    # Khoảng cách từ hông đến cổ tay phải
    right_arm_reach = calculate_distance(right_hip, right_wrist)
    features.append(right_arm_reach)

    # Tính toán góc
    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    # Góc khuỷu tay trái
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    features.append(left_elbow_angle)

    # Góc khuỷu tay phải
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    features.append(right_elbow_angle)

    # Góc đầu gối trái
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    features.append(left_knee_angle)

    # Góc đầu gối phải
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    features.append(right_knee_angle)

    # 3. Kiểm tra số lượng đặc trưng
    expected_feature_length = 99 + 9  # 99 keypoints + 9 bổ sung = 108
    if len(features) != expected_feature_length:
        raise ValueError(f"Đặc trưng có kích thước không đúng: {len(features)}. Cần phải là {expected_feature_length}.")

    # 4. Chuẩn hóa các đặc trưng bằng chiều cao người (khoảng cách từ mũi đến mắt cá chân trái)
    person_height = calculate_distance(nose, left_ankle)
    if person_height > 0:
        normalized_features = [f / person_height for f in features]
    else:
        normalized_features = features

    return normalized_features

def process_features(dataset_type):
    """
    Xử lý các tệp keypoints để trích xuất đặc trưng và lưu chúng vào thư mục tương ứng.

    Args:
        dataset_type (str): Loại tập dữ liệu ('train', 'validation', 'test').
    """
    classes = ['correct_behavior', 'incorrect_behavior']
    for class_name in classes:
        keypoint_dir = f'D:/Detective_littering/data/pose_data/keypoints/{dataset_type}/{class_name}'
        feature_dir = f'D:/Detective_littering/data/pose_data/features/{dataset_type}/{class_name}'
        os.makedirs(feature_dir, exist_ok=True)
        keypoint_files = [f for f in os.listdir(keypoint_dir) if f.endswith('.npy')]
        print(f"\nProcessing features for {dataset_type}/{class_name}: {len(keypoint_files)} files")
        for keypoint_file in keypoint_files:
            keypoints = np.load(os.path.join(keypoint_dir, keypoint_file))
            # Kiểm tra nếu keypoints là mảng mặc định (chứa toàn 0)
            if np.all(keypoints == 0):
                print(f"File {keypoint_file} chứa Pose mặc định. Bạn có thể cần xử lý đặc biệt.")
                # Bạn có thể quyết định bỏ qua hoặc gán giá trị đặc trưng mặc định
                # Ví dụ: bỏ qua
                continue
            try:
                features = extract_features_from_keypoints(keypoints)
            except ValueError as ve:
                print(f"Error processing file {keypoint_file}: {ve}")
                continue
            feature_path = os.path.join(feature_dir, os.path.splitext(keypoint_file)[0] + '.npy')
            np.save(feature_path, np.array(features))
        print(f"Đã trích xuất đặc trưng cho {len(keypoint_files)} tệp trong {dataset_type}/{class_name}")

# Xử lý đặc trưng cho tập train, validation và test
if __name__ == "__main__":
    process_features('train')
    process_features('validation')
    process_features('test')
