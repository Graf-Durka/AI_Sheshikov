import torch.nn as nn
import torch
import random as rn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from torchvision import transforms
from torchvision.transforms import functional
import tifffile
import numpy as np
import matplotlib.pyplot as plt

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

class DataLoad(Dataset):
    def __init__(self, path_img, path_mask, crop_size = 512, is_train = 1):
        self.path_img = path_img
        self.path_mask = path_mask
        self.crop_size = crop_size
        self.is_train = is_train

        self.img = sorted([f for f in os.listdir(path_img) 
               if os.path.isfile(os.path.join(path_img, f))])
        
        self.masc = sorted([f for f in os.listdir(path_mask) 
               if os.path.isfile(os.path.join(path_mask, f))])
        
    def __len__(self):
        return len(self.img)
    
    def _process(self, img, mask):
            img_t = functional.to_tensor(img)
            mask_t = functional.to_tensor(mask)

            if self.is_train:
                i, j, h, w = transforms.RandomCrop.get_params(img_t, output_size=(self.crop_size, self.crop_size))
                img_t = functional.crop(img_t, i, j, h, w)
                mask_t = functional.crop(mask_t, i, j, h, w)

                if rn.random() > 0.5:
                    img_t = functional.hflip(img_t)
                    mask_t = functional.hflip(mask_t)

                angle = rn.choice([0, 90, 180, 270])
                if angle != 0:
                    img_t = functional.rotate(img_t, angle)
                    mask_t = functional.rotate(mask_t, angle)
            else:
                img_t = functional.center_crop(img_t, (self.crop_size, self.crop_size))
                mask_t = functional.center_crop(mask_t, (self.crop_size, self.crop_size))

            img_t = functional.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask_t = (mask_t > 0.5).float()

            return img_t, mask_t
    
    def __getitem__(self, idx):
        img_name = self.img[idx]
        img_path = os.path.join(self.path_img, img_name)
        mask_name = img_name.replace('.tiff', '.tif')
        mask_path = os.path.join(self.path_mask, mask_name)

        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        return self._process(image, mask)

def get_dataloader(img_dir, mask_dir, batch_size=4, is_train=True):
    dataset = DataLoad(img_dir, mask_dir, is_train=is_train)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train
    )

class FBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3,1), padding=(1, 0)),
            nn.Conv2d(out_c, out_c, kernel_size=(1,3), padding=(0, 1)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_c, out_c, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Custom_UNet(nn.Module):
    def __init__(self):
        super(Custom_UNet, self).__init__()

        #ENCODER
        self.enc0 = FBlock(3, 64)
        self.p0 = nn.MaxPool2d(2)

        self.enc1 = FBlock(64, 128)
        self.p1 = nn.MaxPool2d(2)

        self.enc2 = FBlock(128, 256)
        self.p2 = nn.MaxPool2d(2)

        #BOTTLENECK
        self.bottlenck = FBlock(256, 512)

        #DECODER
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = FBlock(512, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = FBlock(256, 128)
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = FBlock(128, 64)

        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.p0(e0))
        e2 = self.enc2(self.p1(e1))

        b = self.bottlenck(self.p2(e2))

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        u0 = self.up0(d1)
        d0 = self.dec0(torch.cat([u0, e0], dim=1))

        return torch.sigmoid(self.head(d0))
    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.BCE = nn.BCELoss()
    
    def forward(self, pred, targ):
        bce_loss = self.BCE(pred, targ)

        smooth = 1e-6
        pred_f = pred.view(-1)
        targ_f = targ.view(-1)

        intersection = (pred_f * targ_f).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (pred_f.sum() + targ_f.sum() + smooth)) 

        return 0.5 * bce_loss + 0.5 * dice_loss

    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    torch.cuda.empty_cache()

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0
    
    model.to(device)

    for epoch in range(num_epochs):
        #TRAIN
        model.train()
        train_loss = 0
        train_iou_sum = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            train_iou_sum += calculate_iou(predicted, labels)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou_sum / len(train_loader)
        
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        
        #VALIDATION
        model.eval()
        val_loss = 0
        val_iou_sum = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_iou_sum += calculate_iou(predicted, labels)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss={avg_train_loss:.4f}, Train IoU={avg_train_iou:.4f} | '
              f'Val Loss={avg_val_loss:.4f}, Val IoU={avg_val_iou:.4f}')
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses, train_ious, val_ious

def calculate_iou(pred, target, smooth=1e-7):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def evaluate_on_test(model, test_loader, criterion, device='cuda'):
    model.eval()
    
    test_loss_sum = 0
    test_iou_sum = 0
    
    model.to(device)
        
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss_sum += loss.item()
            
            predicted = (outputs > 0.5).float()
            test_iou_sum += calculate_iou(predicted, labels) 
    
    avg_test_loss = test_loss_sum / len(test_loader)
    avg_test_iou = test_iou_sum / len(test_loader)
    
    print(f'Test Results: Loss={avg_test_loss:.4f}, IoU={avg_test_iou:.4f}')
    
    return avg_test_loss, avg_test_iou

def visualize_prediction(model, image_tensor, mask_tensor, device='cuda'):

    model.eval()
    model.to(device)
    
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output.squeeze(0) > 0.5).float().cpu().numpy()

    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    gt_mask = mask_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image (Satellite)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth (Target)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.title("Model Prediction (Output)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Devisce",  DEVICE)

    model = Custom_UNet()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'archive/tiff')
    folders = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

    train = get_dataloader(folders[2], folders[3], is_train=True) 
    val = get_dataloader(folders[4], folders[5], is_train=False)
    test = get_dataloader(folders[0], folders[1], is_train=False)

    model_path = 'best_model.pth'
    criterion = Loss()

    if os.path.exists(model_path) and not (input("Do you want to retrain the model? (y/n) ")).lower() == "y":
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Pre-trained model loaded")

    else:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        if (input("Maybe you want to Fine-tun the model? (y/n) ")).lower() == "y":
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        train_model(model, train, val, criterion, optimizer, NUM_EPOCHS, DEVICE)
    
    evaluate_on_test(model, test, criterion)

    for images, masks in test:
        for i in range(len(images)):
            visualize_prediction(model, images[i], masks[i], device=DEVICE)

if __name__ == "__main__":
    main()







