import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from dataset import TrafficLightDataset

from net import Net

if __name__ == '__main__':
    BATCH_SIZE = 1

    model = Net(2)
    model.load_state_dict(torch.load('saved/model2.pth'))
    model.eval()

    test_path = 'ModelDataSplit/test'

    test_dataloader = DataLoader(torch.load(test_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for data in test_dataloader:
        images, class_labels, box_labels = data
        box_output = model(images.float())

        box_criterion = nn.MSELoss()
        loss = box_criterion(box_output, box_labels)
        print(loss)

        img = images[0]
        w = img.size(dim=2)
        h = img.size(dim=1)

        box_output[0] = torch.tensor(
            [box_output[0][0] * w, box_output[0][1] * h, box_output[0][0] * w + box_output[0][2] * w,
             box_output[0][1] * h + box_output[0][3] * h])

        img = draw_bounding_boxes(img, box_output, width=5,
                                  colors="green",
                                  fill=True)

        img = torchvision.transforms.ToPILImage()(img)
        img.show()

        # npimg = img.numpy()
        # plt.imsave('test.png', np.transpose(npimg, (1, 2, 0)))

        quit()
        print(box_output)
