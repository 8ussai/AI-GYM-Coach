from modules.common.paths import get_cleaned_frames, get_cleaned_reps, DATASETS_DIR

import numpy as np
import pandas as pd
import os
import json

from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


CONFIG = {
    'sequence_length': 40,          
    'noise_std': 0.02,              
    'augmentation_factor': 2,       
    'random_seed': 42,              
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'feature_cols': [
        'sq_knee_angle_L', 'sq_knee_angle_R',
        'sq_torso_incline', 'sq_pelvis_drop',
        'sq_stance_ratio', 'sq_elbow_angle_L',
        'sq_elbow_angle_R'
    ]
}

np.random.seed(CONFIG['random_seed'])

def loading_and_preparing_data(frames_path: str, reps_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames_df = pd.read_csv(frames_path)
    reps_df = pd.read_csv(reps_path)

    # حذف الأعمدة غير المستخدمة
    frames_df = frames_df.drop(columns=["pose_confidence"], errors='ignore')
    frames_df = frames_df.drop(columns=["sq_knee_angle_mean"], errors='ignore')
    frames_df = frames_df.drop(columns=["sq_bar_present"], errors='ignore')  # ✅ YOLO منفصل
    frames_df = frames_df.dropna(subset=["rep_id"])

    reps_df["reason"] = reps_df["reason"].replace("", pd.NA)
    reps_df["reason"] = reps_df["reason"].fillna("Correct")
    reps_df = reps_df.drop(columns=["label"])
    reps_df = reps_df.rename(columns={"reason": "label"})

    return frames_df, reps_df

def extract_sequences(frames_df: pd.DataFrame, reps_df: pd.DataFrame) -> List[Dict]:
    """
    ✅ استخراج التسلسلات مع Padding/Truncation محسّن
    - إذا أقل من 40: نكرر آخر فريم (بدل zeros)
    - إذا أكثر من 40: نقتطع للـ 40 الأولى
    """
    sequences = []
    
    for _, rep in reps_df.iterrows():
        video = rep['video_name']
        rep_id = rep['rep_id']
        label = rep['label']
        
        rep_frames = frames_df[(frames_df['video_name'] == video) & (frames_df['rep_id'] == rep_id)].sort_values('frame_idx')
        
        features = rep_frames[CONFIG['feature_cols']].values
        original_length = len(features)
        
        # ✅ Padding/Truncation Strategy
        if len(features) < CONFIG['sequence_length']:
            # Padding: كرر آخر فريم بدل zeros
            gap = CONFIG['sequence_length'] - len(features)
            last_frame = features[-1:].repeat(gap, axis=0)
            features = np.vstack([features, last_frame])
        else:
            # Truncation: خذ أول 40 فريم
            features = features[:CONFIG['sequence_length']]
        
        sequences.append({
            'video_name': video, 
            'rep_id': rep_id, 
            'sequence': features, 
            'label': label, 
            'original_length': original_length
        })
    
    return sequences

def add_noise_augmentation(sequences: List[Dict]) -> List[Dict]:
    augmented = []
    
    for seq_dict in sequences:
        augmented.append(seq_dict.copy())
        
        for aug_idx in range(CONFIG['augmentation_factor'] - 1):
            noisy_seq = seq_dict.copy()
            noise = np.random.normal(0, CONFIG['noise_std'], seq_dict['sequence'].shape)
            noisy_seq['sequence'] = seq_dict['sequence'] + noise
            noisy_seq['is_augmented'] = True
            noisy_seq['augmentation_id'] = aug_idx + 1
            augmented.append(noisy_seq)
    
    return augmented

def prepare_dataset(sequences: List[Dict]) -> Dict:
    original_seqs = [s for s in sequences if not s.get('is_augmented', False)]
    
    train_seqs, test_seqs = train_test_split(
        original_seqs, 
        test_size=CONFIG['val_ratio'] + CONFIG['test_ratio'], 
        random_state=CONFIG['random_seed'], 
        stratify=[s['label'] for s in original_seqs]
    )
    
    val_seqs, test_seqs = train_test_split(
        test_seqs, 
        test_size=CONFIG['test_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio']), 
        random_state=CONFIG['random_seed'], 
        stratify=[s['label'] for s in test_seqs]
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
        'X_train': X_train, 'y_train': y_train, 
        'X_val': X_val, 'y_val': y_val, 
        'X_test': X_test, 'y_test': y_test, 
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
            'num_classes': len(data['label_encoder'].classes_)
        }
    }
    
    with open(f'{output_dir}/squat_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset saved successfully!")
    print(f"📊 Stats: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")
    print(f"📐 Shape: ({CONFIG['sequence_length']} frames, {data['X_train'].shape[-1]} features)")
    
    return metadata

def main():
    print("="*60)
    print("🏋️  Building Squat Dataset")
    print("="*60)
    
    frames_df, reps_df = loading_and_preparing_data(
        get_cleaned_frames("squat"), 
        get_cleaned_reps("squat")
    )
    
    print(f"✅ Loaded: {len(frames_df)} frames, {len(reps_df)} reps")
    print(f"✅ Labels: {reps_df['label'].value_counts().to_dict()}")
    
    sequences = extract_sequences(frames_df, reps_df)
    print(f"✅ Extracted: {len(sequences)} sequences")
    
    augmented_sequences = add_noise_augmentation(sequences)
    print(f"✅ Augmented: {len(augmented_sequences)} total sequences")
    
    dataset = prepare_dataset(augmented_sequences)
    metadata = save_dataset(dataset, DATASETS_DIR / "squat")
    
    print("="*60)
    return dataset, metadata

if __name__ == "__main__":
    dataset, metadata = main()