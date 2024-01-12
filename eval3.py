import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from collections import namedtuple
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the pyramid pooling module
class pyramid_pooling_module(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes):
        super(pyramid_pooling_module, self).__init__()

        # Create pyramid pool layers
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
            # Upsample pyramid pool layers and concatenate them
            out.append(F.interpolate(layer(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

# Define the PSPNet model
class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_aux=False):
        super(PSPNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Use resnet50 from torchvision without deprecated arguments
        backbone = resnet50(pretrained=False)
        self.initial = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        ppm_in_channels = int(backbone.fc.in_features)
        # Create pyramid pooling module and classifier
        self.ppm = pyramid_pooling_module(in_channels=ppm_in_channels, out_channels=512, bin_sizes=[1, 2, 3, 6])
        self.cls = nn.Sequential(
            nn.Conv2d(ppm_in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

        # Combine pyramid pooling module and classifier into the main branch
        self.main_branch = nn.Sequential(self.ppm, self.cls)

        self.use_aux = False
        if use_aux:
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

        # Pass input through the layers of ResNet
        x = self.initial(x)
        x = self.layer1(x)
        x_aux = self.layer2(x)
        x = self.layer3(x_aux)
        x = self.layer4(x)

        # Pass through the main branch
        main_output = self.main_branch(x)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=True)

        if self.use_aux:
            # Pass through the aux branch
            aux_output = F.interpolate(self.aux_branch(x_aux), size=input_size, mode='bilinear', align_corners=True)
            output = {'aux': aux_output, 'main': main_output}
            return output

        return main_output

# Define the dataset class
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

# Define the label information
Label = namedtuple("Label", ["name", "train_id", "color"])
drivables = [
    Label("direct", 0, (255, 0, 0)),        # green
    Label("alternative", 1, (0, 0, 255)),  # blue
    Label("background", 2, (0, 0, 0)),        # black          
]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)

# Function to process video and save segmented output as .avi
def process_video(input_video_path, output_video_path, model, transform):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    # Create a figure for displaying the segmented frames
    plt.figure()

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(frame_rgb)

            # Apply transformations
            input_tensor = transform(input_image).unsqueeze(0)

            # Inference
            output = model(input_tensor)

            # Post-process the output
            prediction = torch.argmax(output['aux'], dim=1).squeeze().numpy()

            # Ensure prediction has a valid range for indexing into train_id_to_color
            prediction = np.clip(prediction, 0, len(train_id_to_color) - 1)

            # Index into train_id_to_color and convert to uint8
            prediction_color = train_id_to_color[prediction].astype(np.uint8)

            # Convert RGB to BGR
            prediction_bgr = cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR)

            # Save segmented frame to video
            out.write(prediction_bgr)

            # Display the segmented frame using matplotlib
            plt.imshow(prediction_bgr)
            plt.pause(0.01)

    cap.release()
    out.release()
    plt.show()  # Display the matplotlib plot

# Define the transformation for the input images
transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Specify the input video path
input_video_path = '/home/annotationguest/SegmentationDrivable/inst_lane.mp4'  # Replace with the actual path to your input video

# Specify the output video path
output_video_path = 'output_video.avi'

# Load the pre-trained PSPNet model
model = PSPNet(in_channels=3, num_classes=3, use_aux=True)
model.load_state_dict(torch.load('/home/annotationguest/SegmentationDrivable/PSPNet_drivingspace_weights.pt', map_location=torch.device('cpu')))
model.eval()

# Process the video and save the segmented output
process_video(input_video_path, output_video_path, model, transform)

