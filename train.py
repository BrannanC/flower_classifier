import argparse
import torch
from torchvision import datasets, transforms
from time import time, sleep

from get_input_args import get_training_args
from get_model import get_model


def main():
    s_time = time()
    train_args = get_training_args()
    data_dir = train_args.dir
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomRotation(15),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    model, criterion, optimizer, in_size, out_size = get_model(
        train_args.arch, train_args.learning_rate, train_args.hidden_units, train_args.hidden_layers)

    if train_args.gpu and not torch.cuda.is_available():
        raise Exception('GPU not available')

    device = torch.device("cuda" if torch.cuda.is_available()
                          and train_args.gpu else "cpu")
    model.to(device)

    epochs = int(train_args.epochs)
    count = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            if count % 10 == 0:
                print('running loss', running_loss/len(trainloader))
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
        else:
            print(f"Epoch {epoch+1}/{epochs}.. ")
            print(f"    Training loss: {running_loss/len(trainloader)}")
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)

                    loss = criterion(logps, labels)

                    test_loss = loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(
                    f"    Validation Loss: {test_loss/len(validloader):.3f} ")
                print(
                    f"    Validation accuracy: {accuracy/len(validloader):.3f}")
                model.train()

    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    model.train()

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'input_size': in_size,
        'output_size': out_size,
        'arch': train_args.arch,
        'learning_rate': train_args.learning_rate,
        'hidden_units': train_args.hidden_units,
        'hidden_layers': train_args.hidden_layers,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, train_args.save_dir + 'checkpoint.pth')

    print('Training Time:', time() - s_time)


if __name__ == "__main__":
    main()
