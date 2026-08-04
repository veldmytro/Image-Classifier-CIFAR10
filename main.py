import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import ImageClassifier
import torch.nn as nn

def main():
    load_checkpoint = True
    train_model = True
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # print(f"Using device: {device}")

    transform = transforms.ToTensor()
    
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=False
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True
    )

    model = ImageClassifier().to(device)
    if load_checkpoint:
        checkpoint_path = "checkpoints/image_classifier450.pth"
        model.load_state_dict(
            torch.load(checkpoint_path)
        )
        print(f"Checkpoint loaded: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )
    num_epochs = 50

    if train_model:
        train(model,device,train_loader,criterion,optimizer,num_epochs)
    train_loss, train_accuracy = evaluate(model, train_loader, criterion, device)
    test_loss, test_accuracy = evaluate(model,test_loader,criterion,device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    # torch.save(model.state_dict(), "checkpoints/image_classifier280.pth")
    

def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += ((predictions == labels).sum()).item()
            total += len(labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Average loss: {running_loss / len(train_loader):.4f}")
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

def evaluate(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    average_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy

if __name__ == "__main__":
    main()