from torch.utils.data import DataLoader, Dataset


def get_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """
    Create dataloader.
    """
    return DataLoader(dataset, batch_size=batch_size)