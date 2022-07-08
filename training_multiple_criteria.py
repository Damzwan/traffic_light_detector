import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TrafficLightDataset
from net import Net

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()

    BATCH_SIZE = 8
    MAX_EPOCHS = 50
    INIT_LR = 0.001
    WEIGHT_DECAY = 0.00005
    LR_DROP_MILESTONES = [400, 600]
    alpha = 0.5

    running_loss = 0.0  # stores the total loss for the epoch
    running_loss_MSE = 0.0  # stores the total MSE loss for the epoch
    running_loss_cross_entropy = 0.0  # store the total cross entropy loss for the epoch

    training_path = 'filtered.csv'
    train_accuracies = []  # stores the training accuracy of the network at each epoch

    img_dir = 'dataset'

    train_dataset = TrafficLightDataset(csv_file=training_path, img_dir=img_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    net = Net()

    class_criterion = nn.BCELoss()
    box_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=INIT_LR, weight_decay=0.000005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DROP_MILESTONES)

    if cuda_available:
        net = net.cuda()

    for epoch in range(MAX_EPOCHS):
        for i, data in enumerate(train_dataloader, 0):
            images, class_labels, box_labels = data

            if cuda_available:
                images = images.cuda().float()
                class_labels = class_labels.cuda()
                box_labels = box_labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            class_output, box_output = net(images)
            _, predicted = torch.max(class_output, 1)  # finds index of largest probability

            binary_cross_entropy = class_criterion(class_output, class_labels)
            MSE = box_criterion(box_output, box_labels)
            loss = binary_cross_entropy * alpha + MSE * (1 - alpha)  # combine losses

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_MSE += MSE.item()
            running_loss_cross_entropy += binary_cross_entropy.item()

        print('Epoch: ' + str(epoch + 1))
        print('Average training loss: ' + str(running_loss / (i + 1)))
        print('Average training MSE loss: ' + str(running_loss_MSE / (i + 1)))
        print('Average training cross entropy loss: ' + str(running_loss_cross_entropy / (i + 1)))

        lr_scheduler.step(epoch + 1)  # decrease learning rate if at desired epoch
