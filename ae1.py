import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model


class AE1(_Autoencoder):
    """AutoEncoder 1 - single convolution + max-pooling + single deconvolution w/ sigmoid"""

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            # input_shape = input_size, output_size, num_params, kernel size
            input_shape=(self.BATCH_SIZE, 3, 64, 64),
            layers=[
                nn.Conv2d(
                    in_channels=3,
                    out_channels=n_kernels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ],
        )

        self.decoder = Model(
            # these numbers were chosen arbitrarily via experimentation
            input_shape=(self.BATCH_SIZE, 64, 16, 32),
            layers=[
                nn.ConvTranspose2d(
                    in_channels=n_kernels,
                    out_channels=3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Sigmoid(),
            ],
        )

        self.model = Model(
            input_shape=self.encoder.input_shape, layers=[self.encoder, self.decoder]
        )


if __name__ == "__main__":

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load("data", image_size=64)
    data.shuffle()

    ae = AE1("models/ae1.pt")
    ae.print()

    if not epochs:
        print(f"\nLoading {ae.path}...")
        ae.load()
    else:
        print(f"\nTraining...")
        ae.train(epochs, data)
        print(f"\nSaving {ae.path}...")
        ae.save()

    print(f"\nGenerating samples...")
    samples = ae.generate(data)
    data.display(32)
    samples.display(32)
