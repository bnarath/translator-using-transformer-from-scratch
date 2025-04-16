from torch.utils.data import DataLoader


def display_first_n_batch(dataloader: DataLoader, n=5):
    for batch_id, batch in enumerate(dataloader):
        print(batch_id, batch)
        if batch_id >= n:
            break
