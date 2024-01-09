# abba-lstm
An implementation of ABBA symbolic representation of Time series and LSTM training on top of that.

References:

Elsworth, S., & Güttel, S. (2020). ABBA: Adaptive Brownian bridge-based symbolic aggregation of time series. Data Mining and Knowledge Discovery, 34(4), 1175-1200. [Link](https://arxiv.org/abs/2003.12469)

Elsworth, S., & Güttel, S. (2020). Time series forecasting using LSTM networks: A symbolic approach. arXiv preprint arXiv:2003.05672. [Link](https://arxiv.org/abs/2003.05672)


## Set up

The code was written in Python 3.11.4. Poetry was used to manage the dependencies. And a makefile was used to create shortcuts for the most common commands.

To install the project:

```bash
make install
```

## Usage

There are two notebooks:
- `toy_dataset.ipynb`: This notebook shows how to use the ABBA representation and how to train a LSTM model on top of it. It uses a toy dataset which is a simple sine wave.
- `main.ipynb`: This notebook evaluates the ABBA representation and the LSTM model the sunspot dataset.

A report of the project is available at the root of the repository: `report.pdf`.
