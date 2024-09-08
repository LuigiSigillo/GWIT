import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, eegs, images, labels):
        self.eegs   = eegs
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        eeg   = self.eegs[index]
        image = self.images[index]
        label = self.labels[index]
        return eeg, image, label

    def __len__(self):
        return len(self.eegs)

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False):
        super().__init__(dataset, batch_size=1, shuffle=shuffle)

    def __iter__(self):
        batch_eeg       = []
        batch_image     = []
        labels_in_batch = []
        for eeg, image, label in super().__iter__():
            # if the batch is empty, add the first data point and label
            eeg = torch.squeeze(eeg, dim=0)
            image = torch.squeeze(image, dim=0)
            label = torch.squeeze(label, dim=0)
            if not batch_eeg:
                batch_eeg.append(eeg)
                batch_image.append(image)
                labels_in_batch.append(label)
                continue

            # if the current label is already in the batch, skip it
            if label in labels_in_batch:
                continue

            # add the data and label to the batch
            batch_eeg.append(eeg)
            batch_image.append(image)
            labels_in_batch.append(label)

            # if the batch is full, yield it
            if len(batch_eeg) == self.batch_size:
                yield torch.stack(batch_eeg), torch.stack(batch_image), torch.tensor(labels_in_batch)
                batch_eeg   = []
                batch_image = []
                labels_in_batch = []
        
        # yield the last batch, if it is not full
        if batch_eeg:
            yield torch.stack(batch_eeg), torch.stack(batch_image), torch.tensor(labels_in_batch)