import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import random as rn
import imagehash
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def is_duplicate(img_path, existing_hashes, threshold=5):
    img = cv2.imread(img_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    current_hash = imagehash.phash(pil_image)
    
    for prev_hash in existing_hashes:
        if (current_hash - prev_hash) < threshold:
            return True, current_hash
    
    return False, current_hash


def load_and_split_dataset(path, test_ratio=0.15, valid_ratio=0.15):

    train = []
    test = []
    valid = []
    class_names = []
    
    folders = [f for f in os.listdir(path) 
               if os.path.isdir(os.path.join(path, f))]
    folders.sort()
    
    for class_id, folder in enumerate(folders):
        class_names.append(folder)
        folder_path = os.path.join(path, folder)
        
        images = []
        for f in os.listdir(folder_path):
            if Path(f).suffix.lower() == ".jpg":
                images.append(os.path.join(folder_path, f))
        
        images.sort()
        
        good_images = []
        existing_hashes = []
        
        for img_path in images:
            is_dup, current_hash = is_duplicate(img_path, existing_hashes)
            
            if not is_dup:
                good_images.append(img_path)
                existing_hashes.append(current_hash)
        
        rn.shuffle(good_images)
        
        n_total = len(good_images)
        n_test = int(n_total * test_ratio)
        n_valid = int(n_total * valid_ratio)
        
        for i, img_path in enumerate(good_images):
            if i < n_test:
                test.append((img_path, class_id))
            elif i < n_test + n_valid:
                valid.append((img_path, class_id))
            else:
                train.append((img_path, class_id))
        
        print(f"{folder}: {n_total} уникальных из {len(images)} всего")
    
    rn.shuffle(train)
    rn.shuffle(test)
    rn.shuffle(valid)

    train_dict = {}
    test_dict = {}
    valid_dict = {}
    
    for img_path, class_id in train:
        if class_id not in train_dict:
            train_dict[class_id] = []
        train_dict[class_id].append(img_path)
    
    for img_path, class_id in test:
        if class_id not in test_dict:
            test_dict[class_id] = []
        test_dict[class_id].append(img_path)
    
    for img_path, class_id in valid:
        if class_id not in valid_dict:
            valid_dict[class_id] = []
        valid_dict[class_id].append(img_path)
    
    return train_dict, test_dict, valid_dict, class_names

class SimpsonsDataset(Dataset):
    def __init__(self, char_dict, names):
        self.char_dict = char_dict
        self.names = names
        self.transform = transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.4634, 0.4084, 0.3535], std=[0.2120, 0.1912, 0.2206])
                         ])
        
        self.image_paths = []
        self.labels = []
        
        for idx, images in char_dict.items():
            self.image_paths.extend(images)
            self.labels.extend([idx] * len(images))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        label = self.labels[idx]
        return image, label


class MyCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(MyCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100. * correct / total)
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100. * correct / total)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc={train_accs[-1]:.2f}%, Val Acc={val_accs[-1]:.2f}%, Val Loss={val_losses[-1]:.2f}' )
        
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            torch.save(model.state_dict(), 'best_model_w.pth')

    return train_losses, val_losses, train_accs, val_accs


def show_predictions(model, test_images, class_names, num_images=100, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4634, 0.4084, 0.3535], std=[0.2120, 0.1912, 0.2206])
    ])
    
    model.eval()
    
    if len(test_images) > num_images:
        selected_images = rn.sample(test_images, num_images)
    else:
        selected_images = test_images
        num_images = len(test_images)

    for i, image_path in enumerate(selected_images):
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_name = class_names[predicted_class]
        
        plt.figure(figsize=(8, 6))
        
        plt.imshow(image)
        plt.axis('off')
        
        title = f'Prediction: {predicted_name}\nConfidence: {confidence:.2%}'
        plt.title(title, fontsize=14, pad=20)
        
        plt.suptitle(f'File: {os.path.basename(image_path)}', y=0.95, fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Image {i+1}/{num_images}: {os.path.basename(image_path)} -> {predicted_name} ({confidence:.2%})")

def evaluate_on_test(model, test_dict, class_names, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4634, 0.4084, 0.3535], std=[0.2120, 0.1912, 0.2206])
    ])

    model.eval()
    total_correct = 0
    total_samples = 0
    
    class_stats = {}

    for class_id, img_list in test_dict.items():
        if not img_list: 
            continue
        
        class_correct = 0
        class_total = 0
        class_name = class_names[class_id]
        
        for img_path in img_list:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                
                if predicted.item() == class_id:
                    class_correct += 1
                    total_correct += 1
                total_samples += 1
                class_total += 1
        
        acc = class_correct / class_total if class_total > 0 else 0
        class_stats[class_name] = {
            'acc': acc,
            'correct': class_correct,
            'total': class_total
        }

    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"{'Class Name':<25} | {'Accuracy':<10}")
    print("-" * 40)
    for name in sorted(class_stats.keys()):
        stats = class_stats[name]
        print(f"{name:<25} | {stats['acc']:>8.2%}")
    print("="*30)
    
    return overall_acc, class_stats

def main():
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'archive', 'simpsons_dataset')
    
    print("Devisce",  DEVICE)
    
    train, test, valid, class_names = load_and_split_dataset(train_path) 

    train_dataset = SimpsonsDataset(train, class_names)
    val_dataset = SimpsonsDataset(valid, class_names)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = MyCNN(num_classes=len(class_names)).to(DEVICE)

    model_path = 'best_model_w.pth'
    if os.path.exists(model_path) and not (input("Do you want to retrain the model? (y/n) ")).lower() == "y":
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Pre-trained model loaded")

    else:
        counts = []
        for i in range(len(class_names)):
            count = len(train.get(i, []))
            counts.append(count)
        
        counts_tensor = torch.tensor(counts, dtype=torch.float)
        
        weights = 1.0 / (counts_tensor**0.5)
        
        weights = weights / weights.sum() * len(class_names)
        weights = weights.to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if (input("Maybe you want to Fine-tun the model? (y/n) ")).lower() == "y":

            model.load_state_dict(torch.load('best_model_w.pth'))
        
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
        
        model.load_state_dict(torch.load('best_model_w.pth'))

    test_accuracy, _ = evaluate_on_test(model, test, class_names, DEVICE)
    print(f"Accuracy on test: {test_accuracy:.2%}")

    all_test_images = []
    for imgs in test.values():
        all_test_images.extend(imgs)
    
    show_predictions(model, all_test_images, class_names, num_images=10, device=DEVICE)


if __name__ == "__main__":

    rn.seed(13)
    torch.manual_seed(13)
    main()