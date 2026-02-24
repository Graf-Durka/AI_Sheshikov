import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import random as rn
import matplotlib.pyplot as plt

def data_load(path, flag):
    char_dict = {}
    names = []

    if flag == 1:
        folders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        rn.shuffle(folders)

        for idx, folder in enumerate(folders):
            person_name = os.path.basename(folder)
            names.append(person_name)
            folder_pics = [os.path.join(folder, f) for f in os.listdir(folder)
                           if not os.path.isdir(os.path.join(folder, f))]
            rn.shuffle(folder_pics)
            char_dict[idx] = folder_pics
        return char_dict, names

    else:
        test_dict = {}
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            name_without_ext = os.path.splitext(img_file)[0]
            
            parts = name_without_ext.split('_')
            if parts[-1].isdigit():
                person_name = "_".join(parts[:-1])
            else:
                person_name = name_without_ext
            
            full_path = os.path.join(path, img_file)

            if person_name not in test_dict:
                test_dict[person_name] = []
            test_dict[person_name].append(full_path)

        idx_to_name = sorted(list(test_dict.keys()))
        name_to_idx = {name: i for i, name in enumerate(idx_to_name)}
        
        indexed_dict = {name_to_idx[name]: paths for name, paths in test_dict.items()}

        return indexed_dict, idx_to_name

class SimpsonsDataset(Dataset):
    def __init__(self, char_dict, names):
        self.char_dict = char_dict
        self.names = names
        self.transform = transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc={train_accs[-1]:.2f}%, Val Acc={val_accs[-1]:.2f}%')
        
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses, train_accs, val_accs


def show_predictions(model, test_images, class_names, num_images=100, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def evaluate_on_test(model, test_dict, test_class_names, train_class_names, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    train_name_to_idx = {name: idx for idx, name in enumerate(train_class_names)}

    total_correct = 0
    total_samples = 0
    class_correct = {}
    class_total = {}

    for test_name, img_list in zip(test_class_names, test_dict.values()):
        if test_name in train_name_to_idx:
            true_label = train_name_to_idx[test_name]
            class_correct[test_name] = 0
            class_total[test_name] = len(img_list)

            for img_path in img_list:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                if predicted.item() == true_label:
                    class_correct[test_name] += 1
                    total_correct += 1
                total_samples += 1

    overall_acc = total_correct / total_samples
    return overall_acc


def main():
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'archive', 'simpsons_dataset')
    test_path = os.path.join(base_dir, 'archive', 'kaggle_simpson_testset/kaggle_simpson_testset')
    
    print("Devisce",  DEVICE)
    
    train_dict, train_class_names = data_load(train_path, 1)
    
    test_dict, test_class_names = data_load(test_path, 0)
    
    
    full_dataset = SimpsonsDataset(train_dict, train_class_names)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = MyCNN(num_classes=len(train_class_names)).to(DEVICE)

    model_path = 'best_model.pth'
    if os.path.exists(model_path) and not (input("Do you want to retrain the model? (y/n)")).lower() == "y":
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Pre-trained model loaded")

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
        
        model.load_state_dict(torch.load('best_model.pth'))

    test_accuracy = evaluate_on_test(model, test_dict, test_class_names, train_class_names, DEVICE)
    print(f"Accuracy on test: {test_accuracy:.2%}")

    all_test_images = []
    for imgs in test_dict.values():
        all_test_images.extend(imgs)
    
    show_predictions(model, all_test_images, train_class_names, num_images=10, device=DEVICE)


if __name__ == "__main__":

    rn.seed(13)
    torch.manual_seed(13)
    main()