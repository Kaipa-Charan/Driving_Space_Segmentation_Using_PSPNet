import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import argparse

from torchvision.models import resnet50
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import namedtuple

class pyramid_pooling_module(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes):
        super(pyramid_pooling_module, self).__init__()

        self.pyramid_pool_layers = nn.ModuleList()
        for bin_sz in bin_sizes:
            self.pyramid_pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sz),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for layer in self.pyramid_pool_layers:
            out.append(F.interpolate(layer(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_aux=False):
        super(PSPNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.initial = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        ppm_in_channels = int(backbone.fc.in_features)
        self.ppm = pyramid_pooling_module(in_channels=ppm_in_channels, out_channels=512, bin_sizes=[1, 2, 3, 6])
        self.cls = nn.Sequential(
            nn.Conv2d(ppm_in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

        self.main_branch = nn.Sequential(self.ppm, self.cls)

        self.use_aux = False
        if self.training and use_aux:
            self.use_aux = True
            # Modify the input channels for the aux branch
            self.aux_branch = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.initial(x)
        x = self.layer1(x)
        x_aux = self.layer2(x)
        x = self.layer3(x_aux)
        x = self.layer4(x)

        main_output = self.main_branch(x)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=True)

        if self.training and self.use_aux:
            aux_output = F.interpolate(self.aux_branch(x_aux), size=input_size, mode='bilinear', align_corners=True)
            output = {'aux': aux_output, 'main': main_output}
            return output

        return main_output

class DrivingDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        label_name = os.path.join(self.label_dir, self.image_list[idx].replace('.jpg', '.png'))

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name)

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(np.array(label), dtype=torch.long)

        return image, label

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing input images (.jpg)')
parser.add_argument('--label_dir', type=str, required=True, help='Path to the directory containing label masks (.png)')
args = parser.parse_args()

# Create the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Repeat single channel for RGB
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dataset = DrivingDataset(args.image_dir, args.label_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the PSPNet model
model = PSPNet(in_channels=3, num_classes=3, use_aux=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        # Resize labels to match the spatial dimensions of the output
        labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs['aux'].shape[-2:], mode='nearest').squeeze(1).long()

        loss = criterion(outputs['aux'], labels)  # Use 'aux' branch for loss calculation

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Training Loss: {running_loss / len(dataloader)}')

# Save the trained weights
torch.save(model.state_dict(), 'PSPNet_drivingspace_weights.pt')
print('Weights saved successfully.')

