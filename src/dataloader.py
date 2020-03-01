import torch 
from torch.utils.data import DataLoader 
from dataset import SquidDataset


def get_dataloader():
    """get dataloader from datasetss
    
    Returns
    -------
    train_dataloader : DataLoader
    test_dataloader : DataLoader
    """
    dataset = SquidDataset()

    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    test_size = n_samples - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    
    return train_dataloader, test_dataloader 


if __name__ == '__main__':
    # check
    train_loader, test_loader = get_dataloader()
    
    for data in test_loader:
        from IPython import embed; embed(); exit()
        print(len(data))
