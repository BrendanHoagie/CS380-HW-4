import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model
from ae1 import AE1


class AE2(_Autoencoder):
    """AutoEncoder 2 - single convolution + max-pooling + single deconvolution w/ ReLu"""

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, 64, 16, 32),
            layers=[
                nn.Conv2d(
                    in_channels=64,
                    out_channels=n_kernels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ],
        )

        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, 64, 32, 16),
            layers=[
                nn.ConvTranspose2d(
                    in_channels=n_kernels,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
            ],
        )

        self.model = Model(
            input_shape=self.encoder.input_shape, layers=[self.encoder, self.decoder]
        )


if __name__ == "__main__":

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load("data", image_size=64)
    data.shuffle()

    ae1 = AE1("models/ae1.pt")
    ae1.load()
    data_thru_ae1 = ae1.encode(data)

    ae2 = AE2("models/ae2.pt")
    ae2.print()

    if not epochs:
        print(f"\nLoading {ae2.path}...")
        ae2.load()
    else:
        print(f"\nTraining...")
        ae2.train(epochs, data_thru_ae1)
        print(f"\nSaving {ae2.path}...")
        ae2.save()

    print(f"\nGenerating samples...")
    samples = ae2.generate(data_thru_ae1)
    data_thru_ae1.display(32)
    samples.display(32)
