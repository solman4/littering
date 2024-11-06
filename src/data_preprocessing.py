# src/data_preprocessing.py

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
from PIL import Image
import xml.etree.ElementTree as ET

# Bỏ qua các cảnh báo không quan trọng
warnings.filterwarnings('ignore')


def extract_object_features(xml_file):
    """
    Trích xuất trung tâm của các đối tượng 'bin' và 'xarac_item' từ file XML.
    Nếu không tìm thấy đối tượng nào, trả về các giá trị mặc định [0, 0] cho mỗi đối tượng.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return [0, 0, 0, 0]  # Trả về giá trị mặc định nếu XML không hợp lệ
    
    bin_center = [0, 0]
    xarac_center = [0, 0]
    
    # Lấy kích thước hình ảnh để chuẩn hóa tọa độ
    size = root.find('size')
    if size is None:
        print(f"Warning: 'size' tag not found in {xml_file}. Using default image size.")
        width = 1.0  # Tránh chia cho 0
        height = 1.0
    else:
        try:
            width = float(size.find('width').text)
            height = float(size.find('height').text)
        except AttributeError as e:
            print(f"Warning: Missing 'width' or 'height' in 'size' tag of {xml_file}: {e}. Using default image size.")
            width = 1.0
            height = 1.0
    
    # Đếm số lượng đối tượng liên quan được phát hiện
    bin_count = 0
    xarac_count = 0
    
    for obj in root.findall('object'):
        name = obj.find('name').text if obj.find('name') is not None else None
        if name not in ['bin', 'xarac_item']:
            continue  # Bỏ qua các đối tượng không liên quan
        
        bndbox = obj.find('bndbox')
        if bndbox is None:
            print(f"Warning: 'bndbox' tag not found for object '{name}' in {xml_file}. Skipping this object.")
            continue
        
        try:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
        except AttributeError as e:
            print(f"Warning: Missing coordinate in 'bndbox' for object '{name}' in {xml_file}: {e}. Skipping this object.")
            continue
        
        center_x = (xmin + xmax) / 2 / width
        center_y = (ymin + ymax) / 2 / height
        
        if name == 'bin':
            bin_center[0] += center_x
            bin_center[1] += center_y
            bin_count += 1
        elif name == 'xarac_item':
            xarac_center[0] += center_x
            xarac_center[1] += center_y
            xarac_count += 1
    
    # Trung bình hóa nếu có nhiều hơn một đối tượng
    if bin_count > 0:
        bin_center = [coord / bin_count for coord in bin_center]
    if xarac_count > 0:
        xarac_center = [coord / xarac_count for coord in xarac_center]
    
    return bin_center + xarac_center


def calculate_distances(pose_keypoints, bin_center, xarac_center):
    """
    Tính khoảng cách Euclidean từ các điểm tay (left_wrist và right_wrist) đến bin và xarac_item.
    """
    # Giả sử index 15 là tay trái và index 16 là tay phải
    try:
        left_wrist = pose_keypoints[15:17]  # [x, y]
        right_wrist = pose_keypoints[16:18]  # [x, y]
    except IndexError as e:
        print(f"Warning: Pose keypoints array is too short: {e}. Returning default distances.")
        return [0.0, 0.0, 0.0, 0.0]
    
    # Tính khoảng cách
    distance_left_to_bin = np.linalg.norm(np.array(left_wrist) - np.array(bin_center))
    distance_right_to_bin = np.linalg.norm(np.array(right_wrist) - np.array(bin_center))
    distance_left_to_xarac = np.linalg.norm(np.array(left_wrist) - np.array(xarac_center))
    distance_right_to_xarac = np.linalg.norm(np.array(right_wrist) - np.array(xarac_center))
    
    return [distance_left_to_bin, distance_right_to_bin, distance_left_to_xarac, distance_right_to_xarac]


def calculate_angles(pose_keypoints, bin_center, xarac_center):
    """
    Tính góc giữa vector vai-cổ và vector vai-bin để thêm đặc trưng về hướng ném rác.
    """
    import math
    
    # Giả sử index 11 là left_shoulder và index 0 là nose (đại diện cho cổ)
    try:
        left_shoulder = pose_keypoints[11:13]  # [x, y]
        nose = pose_keypoints[0:2]             # [x, y]
    except IndexError as e:
        print(f"Warning: Pose keypoints array is too short for angle calculation: {e}. Returning 0.0 degrees.")
        return 0.0
    
    # Vector từ vai đến cổ
    vec_shoulder_neck = np.array(nose) - np.array(left_shoulder)
    
    # Vector từ vai đến bin
    vec_shoulder_bin = np.array(bin_center) - np.array(left_shoulder)
    
    # Tính góc giữa hai vector
    norm_shoulder_neck = np.linalg.norm(vec_shoulder_neck)
    norm_shoulder_bin = np.linalg.norm(vec_shoulder_bin)
    
    if norm_shoulder_neck == 0 or norm_shoulder_bin == 0:
        return 0.0  # Trả về 0 độ nếu vector có độ dài bằng 0
    
    dot_product = np.dot(vec_shoulder_neck, vec_shoulder_bin)
    angle_rad = np.arccos(np.clip(dot_product / (norm_shoulder_neck * norm_shoulder_bin), -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Tải và chuẩn hóa hình ảnh bằng cách chia cho 255 để đưa về khoảng [0, 1].
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image) / 255.0  # Chuẩn hóa pixel về [0, 1]
        # image = image.flatten()  # Xóa dòng này để giữ hình ảnh ở định dạng 4 chiều
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_combined_data(pose_folder, object_folder, split):
    """
    Tải và kết hợp dữ liệu pose, object detection và hình ảnh cho từng split (train, validation, test).
    Bao gồm việc trích xuất các đặc trưng bổ sung và xử lý dữ liệu thiếu.
    """
    X_combined_non_image = []
    X_combined_image = []
    y_combined = []
    total_files = 0
    skipped_files = 0
    
    for class_label, label in zip(['correct_behavior', 'incorrect_behavior'], [1, 0]):
        pose_class_folder = os.path.join(pose_folder, split, class_label)
        object_class_folder = os.path.join(object_folder, split, class_label)
        image_class_folder = os.path.join(object_folder, split, class_label, 'images')
        annotation_class_folder = os.path.join(object_folder, split, class_label, 'annotations')
        
        if not os.path.exists(pose_class_folder):
            print(f"Error: Pose folder does not exist: {pose_class_folder}")
            continue
        if not os.path.exists(object_class_folder):
            print(f"Error: Object folder does not exist: {object_class_folder}")
            continue
        if not os.path.exists(image_class_folder):
            print(f"Error: Image folder does not exist: {image_class_folder}")
            continue
        if not os.path.exists(annotation_class_folder):
            print(f"Error: Annotation folder does not exist: {annotation_class_folder}")
            continue
        
        pose_files = sorted([f for f in os.listdir(pose_class_folder) if f.endswith('.npy')])
        xml_files = sorted([f for f in os.listdir(annotation_class_folder) if f.endswith('.xml')])
        image_files = sorted([f for f in os.listdir(image_class_folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')])
        
        # Kiểm tra sự đồng bộ giữa số lượng pose files, xml files và image files
        min_len = min(len(pose_files), len(xml_files), len(image_files))
        if len(pose_files) != len(xml_files) or len(pose_files) != len(image_files):
            print(f"Warning: Mismatch in number of pose, XML, and image files for {split}/{class_label}. Using first {min_len} files.")
        
        for i in range(min_len):
            pose_file = pose_files[i]
            xml_file = xml_files[i]
            image_file = image_files[i]
            total_files += 1
            
            pose_path = os.path.join(pose_class_folder, pose_file)
            xml_path = os.path.join(annotation_class_folder, xml_file)
            image_path = os.path.join(image_class_folder, image_file)
            
            # Kiểm tra sự tồn tại của các file
            if not os.path.exists(pose_path):
                print(f"Warning: Pose file does not exist: {pose_path}")
                skipped_files += 1
                continue
            if not os.path.exists(xml_path):
                print(f"Warning: XML file does not exist: {xml_path}")
                skipped_files += 1
                continue
            if not os.path.exists(image_path):
                print(f"Warning: Image file does not exist: {image_path}")
                skipped_files += 1
                continue
            
            # Kiểm tra tên file trùng khớp
            pose_base = os.path.splitext(pose_file)[0]
            xml_base = os.path.splitext(xml_file)[0]
            image_base = os.path.splitext(image_file)[0]
            if pose_base != xml_base or pose_base != image_base:
                print(f"Warning: Pose file '{pose_file}', XML file '{xml_file}', and Image file '{image_file}' do not match.")
                skipped_files += 1
                continue
            
            try:
                pose_keypoints = np.load(pose_path)
            except Exception as e:
                print(f"Error loading pose file {pose_path}: {e}")
                skipped_files += 1
                continue
            
            try:
                object_features = extract_object_features(xml_path)
                print(f"Processed file: {pose_file} | Object Features: {object_features}")
            except Exception as e:
                print(f"Error parsing XML file {xml_path}: {e}")
                skipped_files += 1
                continue
            
            # Kiểm tra xem có đối tượng nào được phát hiện không (cả bin và xarac_item đều không)
            if all(coord == 0 for coord in object_features):
                # Bỏ qua mẫu dữ liệu này
                print(f"Skipped file: {pose_file} | No relevant objects detected.")
                skipped_files += 1
                continue
            
            # Xử lý hình ảnh
            image_features = load_and_preprocess_image(image_path)
            if image_features is None:
                print(f"Skipped file: {image_file} | Image loading failed.")
                skipped_files += 1
                continue
            
            try:
                distances = calculate_distances(pose_keypoints, object_features[:2], object_features[2:])
                angle = calculate_angles(pose_keypoints, object_features[:2], object_features[2:])
                # Kết hợp các đặc trưng không phải hình ảnh
                combined_features = np.hstack((pose_keypoints, object_features, distances, angle))
                X_combined_non_image.append(combined_features)
                X_combined_image.append(image_features)
                y_combined.append(label)
                print(f"Added sample: {pose_file} | Label: {label}")
            except Exception as e:
                print(f"Error processing data for files {pose_file}, {xml_file}, and {image_file}: {e}")
                skipped_files += 1
                continue
    
    print(f"Total files processed for split '{split}': {total_files}")
    print(f"Total files skipped for split '{split}': {skipped_files}")
    print(f"Total samples collected for split '{split}': {len(X_combined_non_image)}")
    
    return X_combined_non_image, X_combined_image, np.array(y_combined)


def check_missing_values(X, y):
    """
    Kiểm tra và thông báo về các giá trị thiếu trong dữ liệu.
    """
    if np.isnan(X).any():
        print("Dữ liệu có giá trị thiếu.")
    else:
        print("Không có giá trị thiếu trong dữ liệu.")
    
    unique_labels = np.unique(y)
    print(f"Các nhãn lớp: {unique_labels}")


def impute_missing_values(X_train, X_val, X_test):
    """
    Thay thế các giá trị thiếu bằng giá trị trung bình của từng đặc trưng.
    """
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    return X_train, X_val, X_test


def main():
    # Đường dẫn tới các thư mục
    pose_folder = r'D:\Detective_littering\data\pose_data\keypoints'
    object_folder = r'D:\Detective_littering\data\processed'
    
    splits = ['train', 'validation', 'test']
    combined_data_splits_non_image = {}
    combined_data_splits_image = {}
    y_splits = {}
    
    for split in splits:
        print(f"\nLoading data for split: {split}")
        combined_data_non_image, combined_data_image, y = load_combined_data(pose_folder, object_folder, split)
        combined_data_splits_non_image[split] = combined_data_non_image
        combined_data_splits_image[split] = combined_data_image
        y_splits[split] = y
        print(f"Total samples collected for split '{split}': {len(combined_data_non_image)}")
    
    # Kiểm tra nếu không có dữ liệu nào được tải
    for split in splits:
        if len(combined_data_splits_non_image[split]) == 0:
            print(f"Error: X_{split} is empty. Please check your data and preprocessing steps.")
            return
        if len(y_splits[split]) == 0:
            print(f"Error: y_{split} is empty. Please check your data and preprocessing steps.")
            return
    
    # Kiểm tra giá trị thiếu
    for split in splits:
        print(f"\nChecking for missing values in split '{split}':")
        # Tách non-image và image features để kiểm tra
        non_image = np.array(combined_data_splits_non_image[split])
        image = np.array(combined_data_splits_image[split])
        check_missing_values(non_image, y_splits[split])
        check_missing_values(image, y_splits[split])
        # Không cần gán lại trong dictionary vì đã tách riêng biệt
    
    # Thay thế các giá trị thiếu nếu có (nếu cần)
    # Bạn có thể bật chức năng này nếu dữ liệu của bạn có giá trị thiếu
    # print("\nImputing missing values for non-image features...")
    # for split in splits:
    #     non_image, image = combined_data_splits_non_image[split], combined_data_splits_image[split]
    #     combined_data_splits_non_image[split], _, _ = impute_missing_values(non_image, None, None)
    
    # Chuẩn hóa các đặc trưng không phải hình ảnh
    # Chỉ áp dụng StandardScaler lên các đặc trưng không phải hình ảnh
    print("\nNormalizing non-image features...")
    scaler = StandardScaler()
    
    # Tách các đặc trưng không phải hình ảnh và hình ảnh
    X_train_non_image = combined_data_splits_non_image['train']
    X_val_non_image = combined_data_splits_non_image['validation']
    X_test_non_image = combined_data_splits_non_image['test']
    
    # Chuẩn hóa các đặc trưng không phải hình ảnh
    X_train_non_image_scaled = scaler.fit_transform(X_train_non_image)
    X_val_non_image_scaled = scaler.transform(X_val_non_image)
    X_test_non_image_scaled = scaler.transform(X_test_non_image)
    
    # Lưu scaler để sử dụng sau này
    scaler_path = r'D:/Detective_littering/models/scaler_combined.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")
    
    # Lưu trữ dữ liệu đã chuẩn hóa
    print("\nSaving the processed and scaled data...")
    combined_path = 'D:/Detective_littering/data/combined_data'
    os.makedirs(combined_path, exist_ok=True)
    
    # Lưu dữ liệu hình ảnh đã chuẩn hóa
    np.save(os.path.join(combined_path, 'X_train_image.npy'), combined_data_splits_image['train'])
    np.save(os.path.join(combined_path, 'X_val_image.npy'), combined_data_splits_image['validation'])
    np.save(os.path.join(combined_path, 'X_test_image.npy'), combined_data_splits_image['test'])
    
    # Lưu dữ liệu đặc trưng không phải hình ảnh đã chuẩn hóa
    np.save(os.path.join(combined_path, 'X_train_non_image_scaled.npy'), X_train_non_image_scaled)
    np.save(os.path.join(combined_path, 'X_val_non_image_scaled.npy'), X_val_non_image_scaled)
    np.save(os.path.join(combined_path, 'X_test_non_image_scaled.npy'), X_test_non_image_scaled)
    
    # Lưu các nhãn
    np.save(os.path.join(combined_path, 'y_train.npy'), y_splits['train'])
    np.save(os.path.join(combined_path, 'y_val.npy'), y_splits['validation'])
    np.save(os.path.join(combined_path, 'y_test.npy'), y_splits['test'])
    
    print("Data has been successfully processed and saved.")


if __name__ == "__main__":
    main()
