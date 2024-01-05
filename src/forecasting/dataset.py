import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.abba import ABBA


class ForecastingDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int = 10):
        if data.dtype == np.int64:
            self.data = torch.tensor(data, dtype=torch.long)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.sequence_length],
            self.data[idx + self.sequence_length],
        )


def get_datasets_and_loaders(
    standardized_time_series: np.ndarray,
    test_split_ratio: float,
    sequence_length: int,
    abba: ABBA | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, DataLoader, DataLoader]:
    raw_train_data = standardized_time_series[
        : -int(len(standardized_time_series) * test_split_ratio)
    ]
    raw_test_data = standardized_time_series[
        -int(len(standardized_time_series) * test_split_ratio) :
    ]
    train_data = raw_train_data
    test_data = raw_test_data

    if abba is not None:
        _, _, centroid_sequence = abba.learn_transform(train_data)
        train_data = abba.apply_transform(train_data)
        test_data = abba.apply_transform(test_data)
        if verbose:
            print(
                "Average time series length per symbol:", centroid_sequence[:, 0].mean()
            )

    # Add the last sequence_length elements of the training data to the beginning of the test data
    test_data = np.concatenate((train_data[-sequence_length:], test_data), axis=0)

    train_dataset = ForecastingDataset(train_data, sequence_length)
    test_dataset = ForecastingDataset(test_data, sequence_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return raw_train_data, raw_test_data, train_loader, test_loader
