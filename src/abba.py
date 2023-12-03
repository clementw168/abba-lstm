import numpy as np
from sklearn.cluster import KMeans


class ABBA:
    def __init__(
        self,
        increment_threshold: float = 0.1,
        max_length: int | None = None,
        min_cluster_size: int = 1,
        max_cluster_size: int = 20,
        increment_scale: float = 1.0,
    ):
        self.increment_threshold = increment_threshold
        self.max_length = max_length
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.increment_scale = increment_scale

    def standardize(self, time_series: np.ndarray) -> np.ndarray:
        return (time_series - np.mean(time_series)) / np.std(time_series)

    def get_linear_pieces(self, time_series: np.ndarray) -> np.ndarray:
        current_start = 0
        current_end = 1
        linear_pieces = []  # list of lists [x_increment, y_increment]
        index_range = np.arange(len(time_series))

        while current_end < len(time_series):
            current_slope = (time_series[current_end] - time_series[current_start]) / (
                current_end - current_start
            )

            linear_time_series = (
                time_series[current_start]
                + current_slope * index_range[: current_end - current_start + 1]
            )
            error = np.linalg.norm(
                linear_time_series - time_series[current_start : current_end + 1]
            )

            if error <= self.increment_threshold * (
                current_end - current_start + 1
            ) and (
                self.max_length is None
                or current_end - current_start + 1 <= self.max_length
            ):
                current_end += 1
            else:
                linear_pieces.append(
                    [
                        current_end - 1 - current_start,
                        time_series[current_end - 1] - time_series[current_start],
                    ]
                )
                current_start = current_end - 1

        linear_pieces.append(
            [
                len(time_series) - current_start,
                time_series[len(time_series) - 1] - time_series[current_start],
            ]
        )

        return np.array(linear_pieces)

    def unfold_linear_pieces(self, linear_pieces: np.ndarray) -> np.ndarray:
        times_series = np.zeros((int(np.sum(linear_pieces[:, 0])),))
        current_index = 0
        current_y = 0

        for piece in linear_pieces:
            slope = piece[1] / piece[0]
            times_series[
                current_index : current_index + int(piece[0])
            ] = current_y + slope * np.arange(int(piece[0]))
            current_index += int(piece[0])
            current_y += piece[1]

        return times_series

    def _get_max_cluster_variance(
        self, linear_pieces: np.ndarray, labels: np.ndarray, centers: np.ndarray
    ) -> float:
        max_variance = 0.0
        for label in np.unique(labels):
            cluster = linear_pieces[labels == label] - centers[label]
            max_variance = max(
                max_variance, np.var(cluster[:, 0]), np.var(cluster[:, 1])
            )

        return max_variance

    def _order_clusters(
        self, labels: np.ndarray, centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        unique, counts = np.unique(labels, return_counts=True)

        order = np.argsort(counts)[::-1]

        labels_ = np.empty_like(labels)
        for i, label in enumerate(unique[order]):
            labels_[labels == label] = i

        return labels_, centers[order]

    def cluster_pieces(
        self, linear_pieces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear_pieces = np.copy(linear_pieces)

        # Standardize the increments
        x_increment_std = np.std(linear_pieces[:, 0])
        y_increment_std = np.std(linear_pieces[:, 1])

        linear_pieces[:, 0] /= x_increment_std
        linear_pieces[:, 1] /= y_increment_std
        linear_pieces[:, 0] *= self.increment_scale

        n = len(linear_pieces)
        N = np.sum(linear_pieces[:, 0])
        s = 0.2
        lower_bound = (self.increment_threshold / s) ** 2 * 6 * (N - n) / N / n

        variance = np.inf
        k = self.min_cluster_size
        centers = np.empty((0, 2))
        labels = np.empty((0,))

        while k <= self.max_cluster_size - 1 and variance > lower_bound:
            k += 1
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(linear_pieces)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            variance = self._get_max_cluster_variance(linear_pieces, labels, centers)

        centers[:, 0] *= x_increment_std
        centers[:, 0] /= self.increment_scale
        centers[:, 1] *= y_increment_std

        labels, centers = self._order_clusters(labels, centers)

        return labels, centers, centers[labels]

    def stringify_representation(self, labels: np.ndarray, centers: np.ndarray) -> str:
        start_letter = ord("a")
        self.language = {
            chr(start_letter + label): center for label, center in enumerate(centers)
        }

        string = [chr(start_letter + label) for label in labels]
        string = "".join(string)

        return string

    def reverse_digitalization(self, string: str) -> np.ndarray:
        return np.array([self.language[letter] for letter in string])

    def quantize(self, linear_pieces: np.ndarray) -> np.ndarray:
        linear_pieces = np.copy(linear_pieces)

        error = linear_pieces[0, 0] - np.round(linear_pieces[0, 0])
        linear_pieces[0, 0] = np.round(linear_pieces[0, 0])

        if len(linear_pieces) == 1:
            return linear_pieces

        for i in range(1, len(linear_pieces)):
            linear_pieces[i, 0] += error
            error = linear_pieces[i, 0] - np.round(linear_pieces[i, 0])
            linear_pieces[i, 0] = np.round(linear_pieces[i, 0])

        return linear_pieces

    def apply_transform(self, time_series: np.ndarray) -> str:
        time_series = self.standardize(time_series)
        linear_pieces = self.get_linear_pieces(time_series)
        labels, centers, _ = self.cluster_pieces(linear_pieces)

        return self.stringify_representation(labels, centers)

    def apply_inverse_transform(self, string: str) -> np.ndarray:
        linear_approx = self.reverse_digitalization(string)
        linear_approx = self.quantize(linear_approx)

        return self.unfold_linear_pieces(linear_approx)
