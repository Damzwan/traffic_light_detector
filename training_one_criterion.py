import numpy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from net import Net
from dataset import TrafficLightDataset

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()

    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    INIT_LR = 0.0001
    patience = 2

    running_loss = 0.0
    val_loss = 0.0
    loss_history = []
    val_loss_history = []

    training_path = 'ModelDataSplit/train'
    val_path = 'ModelDataSplit/val'

    train_dataloader = DataLoader(torch.load(training_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(torch.load(val_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    net = Net()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=INIT_LR, weight_decay=0.000005)
    box_criterion = nn.MSELoss()

    if cuda_available:
        net = net.cuda()

    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        val_loss = 0

        for data in train_dataloader:
            images, class_labels, box_labels = data

            if cuda_available:
                box_labels = box_labels.cuda()
                images = images.float().cuda()
                class_labels = class_labels.cuda()

            optimizer.zero_grad()

            # predict and calculate loss
            box_output = net(images)
            loss = box_criterion(box_output, box_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        for data in val_dataloader:
            images, _, box_labels = data

            if cuda_available:
                box_labels = box_labels.cuda()
                images = images.float().cuda()

            box_output = net(images)
            loss = box_criterion(box_output, box_labels)
            val_loss += loss.item()

        loss_history.append(running_loss / (len(train_dataloader) * BATCH_SIZE))
        val_loss_history.append(val_loss / (len(val_dataloader) * BATCH_SIZE))

        print(f'[{epoch + 1}] loss: {loss_history[-1]:.3f} val: {val_loss_history[-1]:.3f}')

    numpy.save('saved/train_history2', np.array(loss_history))
    numpy.save('saved/val_history2', np.array(val_loss_history))
    torch.save(net.state_dict(), 'saved/model2.pth')
