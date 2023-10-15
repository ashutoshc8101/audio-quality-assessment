import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class AudioFeatureDataset(Dataset):
    def __init__(self, annotations_file, mode='train'):
        self.data = pd.read_csv(annotations_file)
        self.data = self.data.drop(['Unnamed: 0'], axis=1)

        # Splitting the dataset into train and validation sets
        total_samples = len(self.data)
        train_size = int(0.8 * total_samples)
        valid_size = total_samples - train_size

        if mode == 'train':
            self.data = self.data.iloc[:train_size]
        else:
            self.data = self.data.iloc[train_size:]

        self.features = torch.Tensor(self.data.drop(['class'], axis=1).values)
        self.labels = torch.Tensor(self.data['class'].values)
        self.mode = mode

    def __len__(self):
        # print(self.features.shape)
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# # Example usage
# dataset = AudioFeatureDataset('./working_dataset.csv', mode='train')
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # for batch in dataloader:
# #     x, y = batch
# #     # Your training code here
# # x_train = dataset[0][0]
# # x_target = dataset[0][1]

# # print(len(dataset))
# # print(len(dataset[0][0]))

# dataset_val = AudioFeatureDataset('./working_dataset.csv', mode="val")
# dataloader_val = DataLoader(dataset, batch_size=1, shuffle=True)
# # print(len(dataset_val))