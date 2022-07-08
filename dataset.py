import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms

original_training_path = 'annotations/original_training_file.csv'
original_testing_path = 'annotations/original_testing_file.csv'
original_ = 'annotations/original_validation_file.csv'


# Filter the dataset to only contain items with a 0 (green) or a 1 (red)
def filter_dataset():
    for dataset in [original_training_path, original_testing_path, original_]:
        with open(dataset, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = reader.__next__()
            filtered_rows = [item for item in reader if int(item[1]) <= 1]

            with open(dataset.replace('original_', ''), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(filtered_rows)


def filter_ayes_dataset():
    with open('dataset_challenge/coding_challenge_AYES_bbox.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.__next__()

        filtered_rows = []

        for item in reader:
            if item[1][0] == 'I' and int(item[1].rpartition('_')[-1].rpartition('.')[0]) < 1750:
                continue

            filtered_rows.append(item)

    with open('filtered.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)


def create_data_split():
    dataset = TrafficLightDataset(csv_file='filtered.csv', img_dir='dataset')

    train_size = int(0.7 * len(dataset))
    test_size = val_size = (len(dataset) - train_size) // 2

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_size, test_size, val_size + (
                                                                                         len(dataset) - train_size) % 2])

    torch.save(train_dataset, 'ModelDataSplit/train')
    torch.save(test_dataset, 'ModelDataSplit/test')
    torch.save(val_dataset, 'ModelDataSplit/val')


class TrafficLightDataset(Dataset):

    def __init__(self, csv_file, img_dir):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = self.labels.iloc[index]

        img_path = os.path.join(self.img_dir, item[1])
        img = Image.open(img_path)
        img = torch.tensor(np.transpose(img, (2, 0, 1)))

        class_label = torch.Tensor([0, 0])
        class_label[item[-1]] = 1

        box_label = torch.Tensor(item[2:6])

        return img, class_label, box_label


if __name__ == '__main__':
    # filter_ayes_dataset()
    create_data_split()
