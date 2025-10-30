import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from memory_tracking import (
    get_nvidia_smi_gpu_memory,
    get_memory_breakdown,
    print_memory_stats,
    plot_memory_usage
)

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),    # [B, 1, 28, 28] -> [B, 256, 28, 28]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [B, 512, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B, 512, 14, 14]
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # -> [B, 1024, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                                 # -> [B, 1024, 7, 7]
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),  # [B, 50176] -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 2048),          # -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 10)             # -> [B, 10]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def main():
    global model, optimizer
    
    # Setup device
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model
    model = LargeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Started")
    print("Memory and Time Statistics")
    print("-" * 50)

    # For memory tracking
    stages = []
    gpu_mems = []

    for epoch in range(3):
        epoch_start = time.time()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                stages.append('Before Forward')
                gpu_mems.append(get_memory_breakdown(model, optimizer, device_id))
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if epoch == 0 and batch_idx == 0:
                outputs = model(images)
                stages.append('After Forward')
                gpu_mems.append(get_memory_breakdown(model, optimizer, device_id))
                
                loss = criterion(outputs, labels)
                loss.backward()
                stages.append('After Backward')
                gpu_mems.append(get_memory_breakdown(model, optimizer, device_id))
                
                optimizer.step()
                stages.append('After Optimizer')
                gpu_mems.append(get_memory_breakdown(model, optimizer, device_id))
                
                # Plot memory usage
                plot_memory_usage(stages, gpu_mems, base_filename='single_gpu_memory')
                plot_memory_usage(stages, gpu_mems, base_filename='single_gpu_nocuda_memory', include_cuda_overhead=False)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

        # Time and memory statistics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        mem = get_memory_breakdown(model, optimizer, device_id)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Time: {epoch_time:.2f} seconds")
        print(f"Loss: {avg_loss:.4f}")
        print_memory_stats(mem)

if __name__ == "__main__":
    main()
