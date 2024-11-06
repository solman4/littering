import cv2
import mediapipe as mp
import os
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(image_path, pose):
    """
    Trích xuất keypoints từ hình ảnh sử dụng MediaPipe Pose.

    Args:
        image_path (str): Đường dẫn tới hình ảnh.
        pose (mp.solutions.pose.Pose): Đối tượng Pose của MediaPipe.

    Returns:
        np.ndarray: Mảng keypoints nếu thành công.
        np.ndarray: Mảng mặc định nếu không tìm thấy Pose.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc hình ảnh: {image_path}")
        # Trả về mảng mặc định nếu không thể đọc hình ảnh
        return np.zeros(33 * 3)  # 33 landmarks với 3 giá trị mỗi landmark (x, y, z)
    
    # Chuyển đổi ảnh từ BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Xử lý ảnh để trích xuất Pose
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    else:
        print(f"Không phát hiện tư thế trong ảnh {os.path.basename(image_path)}. Gán giá trị mặc định.")
        # Trả về mảng mặc định nếu không tìm thấy Pose
        return np.zeros(33 * 3)  # 33 landmarks với 3 giá trị mỗi landmark (x, y, z)

def process_dataset(base_dir, splits, behaviors, extensions=('.jpeg', '.jpg', '.png')):
    """
    Xử lý toàn bộ dataset để trích xuất và lưu Pose files.

    Args:
        base_dir (str): Đường dẫn tới thư mục gốc chứa dataset.
        splits (list): Danh sách các tập dữ liệu (ví dụ: ['train', 'validation', 'test']).
        behaviors (list): Danh sách các lớp hành vi (ví dụ: ['correct_behavior', 'incorrect_behavior']).
        extensions (tuple): Các định dạng hình ảnh hợp lệ.
    """
    try:
        for split in splits:
            for behavior in behaviors:
                # Đường dẫn tới thư mục hình ảnh và annotations
                images_dir = os.path.join(base_dir, split, behavior, 'images')
                annotations_dir = os.path.join(base_dir, split, behavior, 'annotations')

                # Đường dẫn tới thư mục lưu Pose
                keypoints_dir = os.path.join(base_dir, 'pose_data', 'keypoints', split, behavior)
                os.makedirs(keypoints_dir, exist_ok=True)

                print(f"\nProcessing split: {split}, behavior: {behavior}")
                print(f"Images directory: {images_dir}")
                print(f"Keypoints directory: {keypoints_dir}")

                # Lấy danh sách file hình ảnh
                images = [img for img in os.listdir(images_dir) if img.lower().endswith(extensions)]
                print(f"Số lượng hình ảnh: {len(images)}")

                for img_name in images:
                    base_name = os.path.splitext(img_name)[0]
                    image_path = os.path.join(images_dir, img_name)
                    keypoints = extract_keypoints(image_path, pose)

                    # Lưu keypoints dưới dạng .npy với tên khớp
                    output_path = os.path.join(keypoints_dir, base_name + '.npy')
                    np.save(output_path, keypoints)
                    print(f"Lưu Pose cho {img_name}: {base_name}.npy")
    finally:
        # Đóng MediaPipe Pose
        pose.close()

if __name__ == "__main__":
    # Đường dẫn tới thư mục dataset đã được xử lý
    dataset_base_dir = r'D:\Detective_littering\data\processed'

    # Các tập dữ liệu và lớp hành vi
    data_splits = ['train', 'validation', 'test']
    behaviors = ['correct_behavior', 'incorrect_behavior']

    # Gọi hàm xử lý dataset
    process_dataset(dataset_base_dir, data_splits, behaviors)

    print("\nĐã hoàn tất việc trích xuất Pose cho toàn bộ dataset.")
