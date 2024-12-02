from numpy import random
from torch.utils.data import SubsetRandomSampler, DataLoader


def get_data_loaders(data, batch_size=32, val_size=0.1, test_size=0.1, seed=0):
    """
    Split data in data loaders: train, val, test
    :param data: list, [[adj_matrices[i], node_features[i], edge_features[i]], Y[i]]]
    :param batch_size: int, size of batches
    :param val_size: float, portion of data to be used for validation
    :param test_size: float, portion of data to be used for testing
    :param seed: int, seed for reproducibility
    :return: train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    # train test val split with seed
    indices = [i for i in range(len(data) // 2)]
    random.seed(seed)
    random.shuffle(indices)
    split1 = int(len(data) // 2 * (1 - val_size - test_size))
    split2 = int(len(data) // 2 * (1 - test_size))

    # split evenly for the 3 classes
    train_indices = indices[:split1] + [indices[i] + len(data) // 2 for i in range(0, split1)]
    val_indices = indices[split1:split2] + [indices[i] + len(data) // 2 for i in range(split1, split2)]
    test_indices = indices[split2:] + [indices[i] + len(data) // 2 for i in range(split2, len(data) // 2)]

    # FIXME:
    # print(train_indices)
    # print(val_indices)
    # print(test_indices)

    # use torch's dataloader
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
    return train_loader, val_loader, test_loader

def get_sequential_data_loader(data):
    # train test val split with seed

    # FIXME:
    # print(train_indices)
    # print(val_indices)
    # print(test_indices)

    # use torch's dataloader
    all_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    return all_loader