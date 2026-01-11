import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
import gc

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ML Libraries
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Memory profiling
import psutil
import tracemalloc

# MediaPipe (only for RandomForest)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. RandomForest evaluation will be skipped.")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    BASE_DIR = Path('.')
    DATASETS_DIR = BASE_DIR / 'datasets'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'evaluation_results'
    
    # Dataset paths
    DATASET_KAPIL = DATASETS_DIR / 'asl'
    DATASET_AYURAJ = DATASETS_DIR / 'asl_dataset'
    
    # Model paths
    RF_MODEL_PATH = MODELS_DIR / 'gesture_model_optimized.pkl'
    RESNET_MODEL_PATH = MODELS_DIR
    
    # Settings
    MAX_SAMPLES_PER_CLASS = 100
    BATCH_SIZE = 32
    IMG_SIZE_RESNET = 224
    COMMON_CLASSES = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directories
Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(Config.RESULTS_DIR / 'plots').mkdir(exist_ok=True)
(Config.RESULTS_DIR / 'reports').mkdir(exist_ok=True)

# ==================== UTILITY FUNCTIONS ====================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    """Get GPU memory usage if available"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            return gpu_info['current'] / 1024 / 1024
        except:
            return 0
    return 0

def save_results_to_file(results, filename):
    """Save results dictionary to JSON file"""
    filepath = Config.RESULTS_DIR / 'reports' / f'{filename}_{Config.TIMESTAMP}.json'
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to: {filepath}")

def save_text_report(content, filename):
    """Save text report"""
    filepath = Config.RESULTS_DIR / 'reports' / f'{filename}_{Config.TIMESTAMP}.txt'
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Text report saved to: {filepath}")

def save_confusion_matrix(cm, class_names, title, filename):
    """Save confusion matrix plot"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filepath = Config.RESULTS_DIR / 'plots' / f'{filename}_{Config.TIMESTAMP}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {filepath}")

def create_comparison_plots(all_results):
    """Create comprehensive comparison visualizations"""
    models = list(all_results.keys())
    datasets = list(all_results[models[0]].keys())
    datasets = [d for d in datasets if d != 'model_info']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    for i, model in enumerate(models):
        accs = [all_results[model][ds]['accuracy'] for ds in datasets]
        ax.bar(x + i*width, accs, width, label=model, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Inference Time
    ax = axes[0, 1]
    for i, model in enumerate(models):
        times = [all_results[model][ds]['avg_inference_time_ms'] for ds in datasets]
        ax.bar(x + i*width, times, width, label=model, alpha=0.8)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Inference Time')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Memory Usage
    ax = axes[0, 2]
    for i, model in enumerate(models):
        mem = [all_results[model][ds]['peak_memory_mb'] for ds in datasets]
        ax.bar(x + i*width, mem, width, label=model, alpha=0.8)
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Peak Memory Usage')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. F1-Score
    ax = axes[1, 0]
    for i, model in enumerate(models):
        f1s = [all_results[model][ds]['f1_score'] for ds in datasets]
        ax.bar(x + i*width, f1s, width, label=model, alpha=0.8)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Model Size
    ax = axes[1, 1]
    sizes = [all_results[m]['model_info']['model_size_mb'] for m in models]
    ax.bar(models, sizes, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Size (MB)')
    ax.set_title('Model Size')
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Valid Samples Ratio
    ax = axes[1, 2]
    for i, model in enumerate(models):
        ratios = [all_results[model][ds]['valid_samples'] / all_results[model][ds]['total_samples'] 
                 for ds in datasets]
        ax.bar(x + i*width, ratios, width, label=model, alpha=0.8)
    ax.set_ylabel('Ratio')
    ax.set_title('Valid Samples Ratio')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = Config.RESULTS_DIR / 'plots' / f'comparison_all_{Config.TIMESTAMP}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plots saved to: {filepath}")

# ==================== DATASET LOADERS ====================

def load_dataset_from_folders(dataset_path, dataset_name, max_samples):
    """Generic loader for folder-based datasets - FIXED VERSION"""
    if not dataset_path.exists():
        print(f"✗ Dataset not found at {dataset_path}")
        return None, None, None
    
    # Find data directory
    possible_paths = [
        dataset_path / 'asl_alphabet_train' / 'asl_alphabet_train',
        dataset_path / 'Train',
        dataset_path / 'train',
        dataset_path,
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists() and any(path.iterdir()):
            data_path = path
            break
    
    if data_path is None:
        print(f"✗ Could not find data in {dataset_path}")
        return None, None, None
    
    print(f"Loading {dataset_name} from: {data_path}")
    
    # FIXED: Build complete class list FIRST
    class_names = []
    class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for class_folder in class_folders:
        class_name = class_folder.name.upper()
        if class_name in Config.COMMON_CLASSES:
            class_names.append(class_name)
    
    class_names = sorted(class_names)  # Consistent ordering
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} valid classes: {class_names}")
    
    # Now load images with correct labels
    images, labels = [], []
    
    for class_folder in tqdm(class_folders, desc=f"Loading {dataset_name}"):
        class_name = class_folder.name.upper()
        
        if class_name not in class_to_idx:
            continue
        
        class_idx = class_to_idx[class_name]
        
        image_files = (list(class_folder.glob('*.jpg')) + 
                      list(class_folder.glob('*.jpeg')) + 
                      list(class_folder.glob('*.png')))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)
            except:
                continue
    
    return np.array(images), np.array(labels), np.array(class_names)

# ==================== RANDOM FOREST EVALUATOR ====================

class RandomForestEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.scaler = None
        self.pca = None
        self.mp_hands = None
        
    def load_model(self):
        if not self.model_path.exists():
            print(f"✗ RF model not found at {self.model_path}")
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.class_names = sorted([str(c) for c in model_data['class_names']])
            self.scaler = model_data.get('scaler')
            self.pca = model_data.get('pca')
            
            if MEDIAPIPE_AVAILABLE:
                mp_hands = mp.solutions.hands
                self.mp_hands = mp_hands.Hands(
                    static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.5, model_complexity=0
                )
            return True
        except Exception as e:
            print(f"✗ Error loading RF model: {e}")
            return False
    
    def extract_features(self, image):
        if self.mp_hands is None:
            return None
        try:
            results = self.mp_hands.process(image)
            if not results.multi_hand_landmarks:
                return None
            
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)
            
            return self._extract_enhanced_features(landmarks)
        except:
            return None
    
    def _extract_enhanced_features(self, landmarks):
        points = landmarks.reshape(21, 3)
        centered = points - points[0]
        scale = np.linalg.norm(points[0] - points[9]) + 1e-6
        normalized = (centered / scale).flatten()
        
        v1, v2 = points[5] - points[0], points[17] - points[0]
        palm_normal = np.cross(v1, v2)
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)
        
        angles = []
        for finger in [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]:
            for i in range(len(finger) - 2):
                p1, p2, p3 = points[finger[i]], points[finger[i+1]], points[finger[i+2]]
                v1, v2 = p1 - p2, p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        
        thumb_tip = points[4]
        pinch_dists = [np.linalg.norm(thumb_tip - points[f]) / scale for f in [8, 12, 16, 20]]
        spread_dists = [np.linalg.norm(points[i] - points[i+1]) / scale for i in [4, 8, 12, 16]]
        
        return np.concatenate([normalized, palm_normal, angles, pinch_dists, spread_dists]).astype(np.float32)
    
    def evaluate(self, images, labels, class_names, dataset_name):
        print(f"\n{'='*60}\nEvaluating RandomForest on {dataset_name}\n{'='*60}")
        
        if not MEDIAPIPE_AVAILABLE:
            print("✗ MediaPipe not available")
            return None
        
        tracemalloc.start()
        start_memory = get_memory_usage()
        
        # Extract features
        X_features, y_labels = [], []
        
        for img, label in zip(tqdm(images, desc="Feature extraction"), labels):
            if label >= len(class_names):
                continue
                
            features = self.extract_features(img)
            if features is not None:
                X_features.append(features)
                y_labels.append(label)
        
        if not X_features:
            print("✗ No valid features extracted")
            return None
        
        X_features = np.array(X_features)
        y_labels = np.array(y_labels)
        
        if self.scaler:
            X_features = self.scaler.transform(X_features)
        if self.pca:
            X_features = self.pca.transform(X_features)
        
        # Find common classes
        dataset_classes = set(class_names)
        model_classes = set(self.class_names)
        common_classes = sorted(list(dataset_classes & model_classes))
        
        if not common_classes:
            print("✗ No common classes")
            return None
        
        common_to_eval = {cls: idx for idx, cls in enumerate(common_classes)}
        
        # Filter to common classes
        valid_mask = np.array([class_names[lbl] in common_to_eval for lbl in y_labels])
        X_features = X_features[valid_mask]
        y_labels = y_labels[valid_mask]
        y_true = np.array([common_to_eval[class_names[lbl]] for lbl in y_labels])
        
        print(f"Valid: {len(X_features)}/{len(images)}, Classes: {len(common_classes)}")
        
        # Inference
        inference_times, predictions = [], []
        for features in tqdm(X_features, desc="Inference"):
            start = time.perf_counter()
            pred = str(self.model.predict([features])[0])
            inference_times.append((time.perf_counter() - start) * 1000)
            predictions.append(common_to_eval.get(pred, -1))
        
        predictions = np.array(predictions)
        valid = predictions >= 0
        y_pred, y_true = predictions[valid], y_true[valid]
        
        print(f"Valid predictions: {len(y_pred)}/{len(predictions)}")
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        
        peak_memory = get_memory_usage()
        current, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(common_classes)))
        save_confusion_matrix(cm, common_classes, f'RandomForest - {dataset_name}',
                            f'rf_cm_{dataset_name.lower().replace(" ", "_")}')
        
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        report = classification_report(y_true, y_pred, 
                                      labels=unique_labels,
                                      target_names=[common_classes[i] for i in unique_labels],
                                      zero_division=0, output_dict=True)
        
        return {
            'dataset': dataset_name,
            'total_samples': len(images),
            'valid_samples': len(X_features),
            'num_classes': len(common_classes),
            'common_classes': common_classes,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'memory_used_mb': float(peak_memory - start_memory),
            'peak_memory_mb': float(peak_traced / 1024 / 1024),
            'classification_report': report
        }
    
    def get_model_info(self):
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        
        return {
            'model_size_mb': float(size_mb),
            'n_estimators': int(self.model.n_estimators),
            'total_nodes': int(total_nodes),
            'n_features': int(self.model.n_features_in_),
            'model_classes': self.class_names,
            'requires_mediapipe': True
        }
    
    def cleanup(self):
        if self.mp_hands:
            self.mp_hands.close()

# ==================== RESNET EVALUATOR ====================

class ResNetEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.mp_hands = None
        
    def find_model_file(self):
        if self.model_path.is_file():
            return self.model_path
        
        keras_files = list(self.model_path.glob('*.keras'))
        if keras_files:
            for kf in keras_files:
                if 'final' in kf.name.lower() or 'ft' in kf.name.lower():
                    return kf
            return sorted(keras_files)[-1]
        return None
    
    def load_model(self):
        model_file = self.find_model_file()
        if model_file is None:
            print(f"✗ ResNet model not found")
            return False
        
        try:
            self.model = tf.keras.models.load_model(model_file)
            output_shape = self.model.output_shape[-1]
            
            if output_shape == 24:
                self.class_names = sorted(list('ABCDEFGHIKLMNOPQRSTUVWXY'))
            else:
                self.class_names = sorted([f'Class_{i}' for i in range(output_shape)])
            
            if MEDIAPIPE_AVAILABLE:
                mp_hands = mp.solutions.hands
                self.mp_hands = mp_hands.Hands(
                    static_image_mode=True, max_num_hands=1,
                    min_detection_confidence=0.3, model_complexity=0
                )
            
            print(f"✓ ResNet loaded, classes: {output_shape}")
            return True
        except Exception as e:
            print(f"✗ Error loading ResNet: {e}")
            return False
    
    def detect_and_extract_hand(self, image):
        if self.mp_hands is None:
            return image, False
        
        try:
            results = self.mp_hands.process(image)
            if not results.multi_hand_landmarks:
                return image, False
            
            h, w = image.shape[:2]
            landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            
            pad_w, pad_h = int((x_max - x_min) * 0.2), int((y_max - y_min) * 0.2)
            x_min, x_max = max(0, x_min - pad_w), min(w, x_max + pad_w)
            y_min, y_max = max(0, y_min - pad_h), min(h, y_max + pad_h)
            
            hand_region = image[y_min:y_max, x_min:x_max]
            return hand_region if hand_region.size > 0 else image, hand_region.size > 0
        except:
            return image, False
    
    def preprocess_image(self, image, extract_hand=True):
        detected = False
        if extract_hand:
            image, detected = self.detect_and_extract_hand(image)
        
        img = cv2.resize(image, (Config.IMG_SIZE_RESNET, Config.IMG_SIZE_RESNET))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return tf.keras.applications.resnet50.preprocess_input(img.astype('float32')), detected
    
    def evaluate(self, images, labels, class_names, dataset_name):
        print(f"\n{'='*60}\nEvaluating ResNet50 on {dataset_name}\n{'='*60}")
        
        tracemalloc.start()
        start_memory, start_gpu = get_memory_usage(), get_gpu_memory()
        
        # Find common classes
        common_classes = sorted(list(set(class_names) & set(self.class_names)))
        if not common_classes:
            print("✗ No common classes")
            return None
        
        common_to_eval = {cls: idx for idx, cls in enumerate(common_classes)}
        model_to_eval = {cls: common_to_eval[cls] for cls in common_classes}
        
        # Filter to valid samples
        valid_mask = np.array([
            label < len(class_names) and class_names[label] in common_to_eval
            for label in labels
        ])
        
        images, labels = images[valid_mask], labels[valid_mask]
        y_true = np.array([common_to_eval[class_names[lbl]] for lbl in labels])
        
        print(f"Valid: {len(images)}, Classes: {len(common_classes)}")
        
        # Preprocess
        X_processed, hands_detected = [], 0
        for img in tqdm(images, desc="Preprocessing"):
            processed, detected = self.preprocess_image(img, MEDIAPIPE_AVAILABLE)
            X_processed.append(processed)
            hands_detected += detected
        
        X_processed = np.array(X_processed)
        if MEDIAPIPE_AVAILABLE:
            print(f"Hand detection: {hands_detected}/{len(images)} images")
        
        # Inference
        inference_times, all_preds = [], []
        for i in tqdm(range(0, len(X_processed), Config.BATCH_SIZE), desc="Inference"):
            batch = X_processed[i:i+Config.BATCH_SIZE]
            start = time.perf_counter()
            pred = self.model.predict(batch, verbose=0)
            per_sample = ((time.perf_counter() - start) * 1000) / len(batch)
            inference_times.extend([per_sample] * len(batch))
            all_preds.append(pred)
        
        all_preds = np.concatenate(all_preds)
        model_indices = np.argmax(all_preds, axis=1)
        
        # Map to eval indices
        y_pred = np.array([
            model_to_eval.get(self.class_names[idx], -1)
            for idx in model_indices
        ])
        
        valid = y_pred >= 0
        y_pred, y_true = y_pred[valid], y_true[valid]
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        
        peak_memory = get_memory_usage()
        current, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(common_classes)))
        save_confusion_matrix(cm, common_classes, f'ResNet50 - {dataset_name}',
                            f'resnet_cm_{dataset_name.lower().replace(" ", "_")}')
        
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        report = classification_report(y_true, y_pred,
                                      labels=unique_labels,
                                      target_names=[common_classes[i] for i in unique_labels],
                                      zero_division=0, output_dict=True)
        
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Inference: {np.mean(inference_times):.2f}ms")
        
        return {
            'dataset': dataset_name,
            'total_samples': len(images),
            'valid_samples': len(X_processed),
            'hands_detected': hands_detected if MEDIAPIPE_AVAILABLE else None,
            'num_classes': len(common_classes),
            'common_classes': common_classes,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_inference_time_ms': float(np.mean(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'memory_used_mb': float(peak_memory - start_memory),
            'peak_memory_mb': float(peak_traced / 1024 / 1024),
            'gpu_memory_used_mb': float(get_gpu_memory() - start_gpu),
            'classification_report': report
        }
    
    def get_model_info(self):
        model_file = self.find_model_file()
        size_mb = model_file.stat().st_size / (1024 * 1024)
        total_params = self.model.count_params()
        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        return {
            'model_size_mb': float(size_mb),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable),
            'model_classes': self.class_names
        }
    
    def cleanup(self):
        if self.mp_hands:
            self.mp_hands.close()

# ==================== MAIN ====================

def main():
    print("="*70)
    print("ASL MODEL EVALUATION SUITE")
    print("="*70)
    print(f"Timestamp: {Config.TIMESTAMP}")
    print(f"Results: {Config.RESULTS_DIR}")
    print("="*70)
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    datasets = {}
    
    kapil_data = load_dataset_from_folders(Config.DATASET_KAPIL, "Kapil Dataset", Config.MAX_SAMPLES_PER_CLASS)
    if kapil_data[0] is not None:
        datasets['Kapil'] = kapil_data
        print(f"✓ Kapil: {len(kapil_data[0])} samples, {len(kapil_data[2])} classes")
    
    ayuraj_data = load_dataset_from_folders(Config.DATASET_AYURAJ, "Ayuraj Dataset", Config.MAX_SAMPLES_PER_CLASS)
    if ayuraj_data[0] is not None:
        datasets['Ayuraj'] = ayuraj_data
        print(f"✓ Ayuraj: {len(ayuraj_data[0])} samples, {len(ayuraj_data[2])} classes")
    
    if not datasets:
        print("✗ No datasets loaded. Exiting.")
        return
    
    # Initialize models
    print("\n[2/4] Loading models...")
    all_results = {}
    
    # Random Forest
    if MEDIAPIPE_AVAILABLE:
        rf_eval = RandomForestEvaluator(Config.RF_MODEL_PATH)
        if rf_eval.load_model():
            print("✓ RandomForest loaded")
            all_results['RandomForest'] = {
                'model_info': rf_eval.get_model_info()
            }
            
            # Evaluate on each dataset
            for name, (images, labels, classes) in datasets.items():
                result = rf_eval.evaluate(images, labels, classes, name)
                if result:
                    all_results['RandomForest'][name] = result
            
            rf_eval.cleanup()
    
    # ResNet50
    resnet_eval = ResNetEvaluator(Config.RESNET_MODEL_PATH)
    if resnet_eval.load_model():
        print("✓ ResNet50 loaded")
        all_results['ResNet50'] = {
            'model_info': resnet_eval.get_model_info()
        }
        
        for name, (images, labels, classes) in datasets.items():
            result = resnet_eval.evaluate(images, labels, classes, name)
            if result:
                all_results['ResNet50'][name] = result
        
        resnet_eval.cleanup()
    
    # Save results
    print("\n[3/4] Generating reports...")
    
    # JSON results
    save_results_to_file(all_results, 'evaluation_results')
    
    # Text summary
    summary = []
    summary.append("="*70)
    summary.append("MODEL EVALUATION SUMMARY")
    summary.append("="*70)
    summary.append(f"Timestamp: {Config.TIMESTAMP}\n")
    
    for model_name in all_results:
        summary.append(f"\n{'='*70}")
        summary.append(f"{model_name} Model")
        summary.append(f"{'='*70}")
        
        info = all_results[model_name]['model_info']
        summary.append(f"\nModel Info:")
        summary.append(f"  Size: {info['model_size_mb']:.2f} MB")
        for key, val in info.items():
            if key != 'model_size_mb':
                summary.append(f"  {key}: {val}")
        
        for ds_name in all_results[model_name]:
            if ds_name == 'model_info':
                continue
            
            results = all_results[model_name][ds_name]
            summary.append(f"\n{ds_name} Dataset:")
            summary.append(f"  Samples: {results['valid_samples']}/{results['total_samples']}")
            summary.append(f"  Classes: {results['num_classes']}")
            summary.append(f"  Accuracy: {results['accuracy']:.4f}")
            summary.append(f"  Precision: {results['precision']:.4f}")
            summary.append(f"  Recall: {results['recall']:.4f}")
            summary.append(f"  F1-Score: {results['f1_score']:.4f}")
            summary.append(f"  Avg Inference: {results['avg_inference_time_ms']:.2f} ms")
            summary.append(f"  Memory Used: {results['memory_used_mb']:.2f} MB")
    
    summary_text = "\n".join(summary)
    save_text_report(summary_text, 'evaluation_summary')
    print(summary_text)
    
    # Comparison plots
    print("\n[4/4] Creating visualizations...")
    if len(all_results) > 1:
        create_comparison_plots(all_results)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {Config.RESULTS_DIR}")
    print(f"  - JSON: {Config.RESULTS_DIR / 'reports'}")
    print(f"  - Plots: {Config.RESULTS_DIR / 'plots'}")
    print(f"  - Text: {Config.RESULTS_DIR / 'reports'}")

if __name__ == "__main__":
    main()