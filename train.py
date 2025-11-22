from __future__ import annotations

import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from collections import Counter
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Any

from preprocessing import normalization

DATASET_PATH = Path("datasets")
CLASSES = ["angry", "happy", "neutral", "surprise",]

def load_dataset_from_folders(dataset_path: Path) -> list[tuple[Path, int]]:
    """
    datasets/
        angry/
            image1.jpg
            image2.jpg
        happy/
            image1.jpg
        ...
    """
    data = []
    for class_name in CLASSES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        class_id = CLASSES.index(class_name)
        image_files = (
            list(class_dir.glob("*.jpg")) + 
            list(class_dir.glob("*.jpeg")) + 
            list(class_dir.glob("*.png"))
        )
        for img_path in image_files:
            data.append((img_path, class_id))
    return data


def stratified_split(data: list[tuple[Path, int]], train_ratio: float = 0.7, 
                     valid_ratio: float = 0.15, seed: int = 42) -> tuple[list, list, list]:
    random.seed(seed)
    
    class_data = {i: [] for i in range(len(CLASSES))}
    for img_path, class_id in data:
        class_data[class_id].append((img_path, class_id))
    
    train_data = []
    valid_data = []
    test_data = []
    
    for class_id, samples in class_data.items():
        random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_valid = int(n_samples * valid_ratio)
        
        train_data.extend(samples[:n_train])
        valid_data.extend(samples[n_train:n_train + n_valid])
        test_data.extend(samples[n_train + n_valid:])
    
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    return train_data, valid_data, test_data


class FacialEmotionDataset(Dataset):
    def __init__(self, image_label_pairs: list[tuple[Path, int]], is_train: bool = False) -> None:
        self.data: list[tuple[NDArray[np.float32], int]] = []
        self.is_train = is_train
        
        print(f"Loading {len(image_label_pairs)} images...")
        for img_path, emotion_class in tqdm(image_label_pairs):
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            
            normalized_data = normalization(image_bgr)
            if normalized_data is None:
                continue
            
            features: Any = normalized_data["normalized_landmarks"]

            self.data.append((features, emotion_class))
        
        print(f"Loaded {len(self.data)} samples (failed: {len(image_label_pairs) - len(self.data)})")
        
        label_counts = Counter([label for _, label in self.data])
        print("Class distribution:")
        for class_id in sorted(label_counts.keys()):
            count = label_counts[class_id]
            percentage = (count / len(self.data)) * 100
            print(f"  {CLASSES[class_id]:10s}: {count:4d} ({percentage:5.1f}%)")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        features, label = self.data[idx]
        features_tensor = torch.tensor(features.flatten(), dtype=torch.float32)
        
        if self.is_train:
            noise = torch.randn_like(features_tensor) * 0.01
            features_tensor += noise
        
        return features_tensor, label

class ExpressionMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 7) -> None:
        super().__init__()

        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_stack(x)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for landmarks, labels in tqdm(dataloader, desc="Training"):
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for landmarks, labels in tqdm(dataloader, desc="Validating"):
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate_with_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    
    with torch.no_grad():
        for landmarks, labels in tqdm(dataloader, desc="Evaluating"):
            landmarks = landmarks.to(device)
            outputs = model(landmarks)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    
    all_predictions_arr = np.array(all_predictions)
    all_labels_arr = np.array(all_labels)
    
    num_classes = len(CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(all_labels_arr, all_predictions_arr):
        confusion_matrix[true_label, pred_label] += 1
    
    print("\nConfusion Matrix:")
    print("Predicted ->")
    header = "True |" + "".join([f"{CLASSES[i][:6]:>8s}" for i in range(num_classes)])
    print(header)
    print("-" * len(header))
    
    for i in range(num_classes):
        row = f"{CLASSES[i][:6]:>5s}|"
        for j in range(num_classes):
            row += f"{confusion_matrix[i, j]:>8d}"
        print(row)
    
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        if class_total > 0:
            class_acc = 100.0 * class_correct / class_total
            print(f"  {CLASSES[i]:10s}: {class_correct:3d}/{class_total:3d} = {class_acc:5.1f}%")
        else:
            print(f"  {CLASSES[i]:10s}: No samples")
    
    total_correct = np.trace(confusion_matrix)
    total_samples = confusion_matrix.sum()
    overall_acc = 100.0 * total_correct / total_samples
    print(f"\nOverall accuracy: {total_correct}/{total_samples} = {overall_acc:.2f}%")
    
    return overall_acc

def main() -> None:
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("Facial Emotion Recognition Training")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASSES}")
    print(f"Dataset path: {DATASET_PATH}")
    print("="*70)
    
    print("\nLoading dataset from folders...")
    all_data = load_dataset_from_folders(DATASET_PATH)
    print(f"Total images found: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No data found! Please check the dataset path.")
        return
    
    print("\nSplitting dataset (stratified)...")
    train_data, valid_data, test_data = stratified_split(
        all_data,
        train_ratio=0.7,
        valid_ratio=0.15,
        seed=42
    )
    
    print(f"   Train: {len(train_data)} samples ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"   Valid: {len(valid_data)} samples ({len(valid_data)/len(all_data)*100:.1f}%)")
    print(f"   Test:  {len(test_data)} samples ({len(test_data)/len(all_data)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Creating datasets...")
    print("="*70)
    train_dataset = FacialEmotionDataset(train_data, is_train=True)
    print()
    valid_dataset = FacialEmotionDataset(valid_data, is_train=False)
    print()
    test_dataset = FacialEmotionDataset(test_data, is_train=False)
    
    if len(train_dataset) == 0:
        print("No training samples after preprocessing. Check normalization/face detection.")
        return
    sample_features, _ = train_dataset[0]
    input_dim = 478 * 2
    print(f"\nFeature dimension: {input_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    model = ExpressionMLP(input_dim=input_dim, num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_acc = 0
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        val_loss, val_acc = validate(
            model, valid_loader, criterion, DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "input_dim": input_dim,
                },
                "best_model.pth",
            )
            print(f"Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*70)
    
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\n" + "="*70)
    print("Detailed Test Set Analysis")
    print("="*70)
    evaluate_with_confusion_matrix(model, test_loader, DEVICE)
    
    print("\n" + "="*70)
    print("All Done!")
    print("="*70)

if __name__ == "__main__":
    main()
