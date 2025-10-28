from modules.common.paths import get_cleaned_frames, get_cleaned_reps, DATASETS_DIR

import numpy as np
import pandas as pd
import os
import json

from typing import Dict, Tuple, List
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.interpolate import interp1d


CONFIG = {
    'sequence_length': 40,          
    'noise_std': 0.02,              
    'augmentation_factor': 2,       
    'minority_augmentation_factor': 10,  
    'minority_classes': ['stance_width', 'back_rounding'],  
    'no_augmentation_classes': ['low_depth'],  
    'random_seed': 42,              
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'feature_cols': [
        'sq_knee_angle_L', 'sq_knee_angle_R',
        'sq_torso_incline', 'sq_pelvis_drop',
        'sq_stance_ratio', 'sq_elbow_angle_L',
        'sq_elbow_angle_R'
    ],
    'augmentation_config': {
        'noise_std': 0.02,
        'time_warp_range': (0.85, 1.15),  
        'magnitude_scale_range': (0.88, 1.12),  
        'angle_shift_range': (-5, 5),  
        'smote_alpha_range': (0.2, 0.8),  
        'smote_probability': 0.4  
    }
}

np.random.seed(CONFIG['random_seed'])

def loading_and_preparing_data(frames_path: str, reps_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames_df = pd.read_csv(frames_path)
    reps_df = pd.read_csv(reps_path)

    frames_df = frames_df.drop(columns=["pose_confidence"])
    frames_df = frames_df.drop(columns=["sq_knee_angle_mean"])
    frames_df = frames_df.drop(columns=["sq_bar_present"])
    frames_df = frames_df.dropna(subset=["rep_id"])

    reps_df["reason"] = reps_df["reason"].replace("", pd.NA)
    reps_df["reason"] = reps_df["reason"].fillna("Correct")
    reps_df = reps_df.drop(columns=["label"])
    reps_df = reps_df.rename(columns={"reason": "label"})

    return frames_df, reps_df

def extract_sequences(frames_df: pd.DataFrame, reps_df: pd.DataFrame) -> List[Dict]:
    sequences = []
    
    for _, rep in reps_df.iterrows():
        video = rep['video_name']
        rep_id = rep['rep_id']
        label = rep['label']
        
        rep_frames = frames_df[(frames_df['video_name'] == video) & (frames_df['rep_id'] == rep_id)].sort_values('frame_idx')
        
        features = rep_frames[CONFIG['feature_cols']].values
        
        if len(features) < CONFIG['sequence_length']:
            padding = np.zeros((CONFIG['sequence_length'] - len(features), 7))
            features = np.vstack([features, padding])
        else:
            features = features[:CONFIG['sequence_length']]
        
        sequences.append({
            'video_name': video, 
            'rep_id': rep_id, 
            'sequence': features, 
            'label': label, 
            'original_length': len(rep_frames)
        })
    
    return sequences

def time_warp_augmentation(sequence: np.ndarray, factor_range: Tuple[float, float]) -> np.ndarray:
    factor = np.random.uniform(*factor_range)
    original_len = len(sequence)
    warped_len = int(original_len * factor)
    
    if warped_len < 2:
        warped_len = 2
    
    indices = np.linspace(0, original_len - 1, warped_len)
    
    warped_sequence = np.zeros((warped_len, sequence.shape[1]))
    for feat_idx in range(sequence.shape[1]):
        interpolator = interp1d(range(original_len), sequence[:, feat_idx], kind='linear')
        warped_sequence[:, feat_idx] = interpolator(indices)
    
    if len(warped_sequence) >= original_len:
        return warped_sequence[:original_len]
    else:
        padding = np.zeros((original_len - len(warped_sequence), sequence.shape[1]))
        return np.vstack([warped_sequence, padding])

def magnitude_scaling_augmentation(sequence: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    return sequence * scale

def angle_shift_augmentation(sequence: np.ndarray, shift_range: Tuple[float, float]) -> np.ndarray:
    shift = np.random.uniform(*shift_range)
    return sequence + shift

def smote_interpolation(seq1: np.ndarray, seq2: np.ndarray, alpha_range: Tuple[float, float]) -> np.ndarray:
    alpha = np.random.uniform(*alpha_range)
    return alpha * seq1 + (1 - alpha) * seq2

def add_noise_augmentation(sequence: np.ndarray, noise_std: float) -> np.ndarray:
    noise = np.random.normal(0, noise_std, sequence.shape)
    return sequence + noise

def apply_random_augmentation(sequence: np.ndarray, aug_config: Dict) -> np.ndarray:
    aug_type = np.random.choice(['noise', 'time_warp', 'magnitude', 'angle_shift', 'combined'])
    
    if aug_type == 'noise':
        return add_noise_augmentation(sequence, aug_config['noise_std'])
    elif aug_type == 'time_warp':
        return time_warp_augmentation(sequence, aug_config['time_warp_range'])
    elif aug_type == 'magnitude':
        return magnitude_scaling_augmentation(sequence, aug_config['magnitude_scale_range'])
    elif aug_type == 'angle_shift':
        return angle_shift_augmentation(sequence, aug_config['angle_shift_range'])
    else:  
        augmented = add_noise_augmentation(sequence, aug_config['noise_std'] * 0.5)
        augmented = magnitude_scaling_augmentation(augmented, (0.95, 1.05))
        return augmented

def augment_sequences(sequences: List[Dict]) -> List[Dict]:
    augmented = []
    
    sequences_by_class = {}

    for seq_dict in sequences:
        label = seq_dict['label']
        if label not in sequences_by_class:
            sequences_by_class[label] = []
        sequences_by_class[label].append(seq_dict)
    
    for seq_dict in sequences:
        augmented.append(seq_dict.copy())
        
        label = seq_dict['label']
        
        if label in CONFIG['no_augmentation_classes']:
            continue
        
        is_minority = label in CONFIG['minority_classes']
        aug_factor = CONFIG['minority_augmentation_factor'] if is_minority else CONFIG['augmentation_factor']
        
        for aug_idx in range(aug_factor - 1):
            smote_prob = CONFIG['augmentation_config']['smote_probability'] if is_minority else 0.2
            
            if is_minority and np.random.random() < smote_prob and len(sequences_by_class[label]) > 1:
                other_seq = np.random.choice([s for s in sequences_by_class[label] if s != seq_dict])
                augmented_sequence = smote_interpolation(
                    seq_dict['sequence'], 
                    other_seq['sequence'],
                    CONFIG['augmentation_config']['smote_alpha_range']
                )
            else:
                augmented_sequence = apply_random_augmentation(
                    seq_dict['sequence'].copy(),
                    CONFIG['augmentation_config']
                )
            
            noisy_seq = seq_dict.copy()
            noisy_seq['sequence'] = augmented_sequence
            noisy_seq['is_augmented'] = True
            noisy_seq['augmentation_id'] = aug_idx + 1
            
            augmented.append(noisy_seq)
    
    return augmented

def prepare_dataset(sequences: List[Dict]) -> Dict:
    original_seqs = [s for s in sequences if not s.get('is_augmented', False)]
    
    labels = [s['label'] for s in original_seqs]
    
    label_counts = Counter(labels)
    
    min_samples = min(label_counts.values())

    can_stratify = min_samples >= 2

    train_seqs, test_seqs = train_test_split(
        original_seqs, 
        test_size=CONFIG['val_ratio'] + CONFIG['test_ratio'], 
        random_state=CONFIG['random_seed'], 
        stratify=labels if can_stratify else None
    )
    
    test_labels = [s['label'] for s in test_seqs]

    test_label_counts = Counter(test_labels)

    can_stratify_test = min(test_label_counts.values()) >= 2
    
    val_seqs, test_seqs = train_test_split(
        test_seqs, 
        test_size=CONFIG['test_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio']), 
        random_state=CONFIG['random_seed'], 
        stratify=test_labels if can_stratify_test else None
    )
    
    augmented_seqs = [s for s in sequences if s.get('is_augmented', False)]

    train_seqs.extend(augmented_seqs)
    
    X_train = np.array([s['sequence'] for s in train_seqs])
    y_train = np.array([s['label'] for s in train_seqs])
    
    X_val = np.array([s['sequence'] for s in val_seqs])
    y_val = np.array([s['label'] for s in val_seqs])
    
    X_test = np.array([s['sequence'] for s in test_seqs])
    y_test = np.array([s['label'] for s in test_seqs])
    
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    
    scaler = StandardScaler()
    
    n_train_samples, n_frames, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train = X_train_scaled.reshape(n_train_samples, n_frames, n_features)
    
    X_val_flat = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_flat)
    X_val = X_val_scaled.reshape(len(val_seqs), n_frames, n_features)
    
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test = X_test_scaled.reshape(len(test_seqs), n_frames, n_features)

    return {
        'X_train': X_train, 
        'y_train': y_train, 
        'X_val': X_val, 
        'y_val': y_val, 
        'X_test': X_test, 
        'y_test': y_test, 
        'scaler': scaler, 
        'label_encoder': le, 
        'train_seqs': train_seqs, 
        'val_seqs': val_seqs, 
        'test_seqs': test_seqs
    }

def save_dataset(data: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_train.npy', data['X_train'])
    np.save(f'{output_dir}/y_train.npy', data['y_train'])
    np.save(f'{output_dir}/X_val.npy', data['X_val'])
    np.save(f'{output_dir}/y_val.npy', data['y_val'])
    np.save(f'{output_dir}/X_test.npy', data['X_test'])
    np.save(f'{output_dir}/y_test.npy', data['y_test'])
    
    class_distribution = {}

    for label_idx, label in enumerate(data['label_encoder'].classes_):
        count = np.sum(data['y_train'] == label_idx)
        class_distribution[label] = int(count)
    
    metadata = {
        'config': CONFIG,
        'label_mapping': {
            int(i): label 
            for i, label in enumerate(data['label_encoder'].classes_)
        },
        'feature_names': CONFIG['feature_cols'],
        'normalization_params': {
            'mean': data['scaler'].mean_.tolist(),
            'std': data['scaler'].scale_.tolist()
        },
        'dataset_stats': {
            'train_size': len(data['X_train']),
            'val_size': len(data['X_val']),
            'test_size': len(data['X_test']),
            'num_features': data['X_train'].shape[-1],
            'sequence_length': data['X_train'].shape[1],
            'num_classes': len(data['label_encoder'].classes_),
            'class_distribution': class_distribution
        }
    }
    
    with open(f'{output_dir}/squat_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    frames_df, reps_df = loading_and_preparing_data(
        get_cleaned_frames("squat"), 
        get_cleaned_reps("squat")
    )
    
    sequences = extract_sequences(frames_df, reps_df)
    augmented_sequences = augment_sequences(sequences)
    dataset = prepare_dataset(augmented_sequences)
    metadata = save_dataset(dataset, DATASETS_DIR / "squat")

    return dataset, metadata

if __name__ == "__main__":
    dataset, metadata = main()