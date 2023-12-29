from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, original_data, server_labels):
        self.data = original_data
        self.labels = server_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        label = self.labels[index]
        return img, label