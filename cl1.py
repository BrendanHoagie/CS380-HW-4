import sys

import torch.nn as nn
from classifier import _Classifier
from autoencoder import _Autoencoder
from data import Data
from model import Model
from ae1 import AE1
from ae2 import AE2
from ae3 import AE3


class CL1(_Classifier):
    """Classifier 1 - stack AE1 + AE2 + AE3 encoders, flatten to 4 units. Tested to 98.6% accuracy"""

    def __init__(
        self,
        path,
        l1_autoencoder: _Autoencoder,
        l2_autoencoder: _Autoencoder,
        l3_autoencoder: _Autoencoder,
    ):
        super().__init__(path)
        self._l1_encoder = l1_autoencoder.encoder
        self._l2_encoder = l2_autoencoder.encoder
        self._l3_encoder = l3_autoencoder.encoder

        self.model = Model(
            input_shape=(self.BATCH_SIZE, 3, 64, 64),
            layers=[
                self._l1_encoder,
                self._l2_encoder,
                self._l3_encoder,
                nn.Flatten(),
                nn.Dropout(p=0.1),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            ],
        )


if __name__ == "__main__":

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load("data", image_size=64)
    data.shuffle()

    ae1 = AE1("models/ae1.pt").load()
    ae2 = AE2("models/ae2.pt").load()
    ae3 = AE3("models/ae3.pt").load()

    cl = CL1("models/cl1.pt", l1_autoencoder=ae1, l2_autoencoder=ae2, l3_autoencoder=ae3)
    cl.print()

    if not epochs:
        print(f"\nLoading {cl.path}...")
        cl.load()
    else:
        train_data, test_data = data.split(0.8)
        print(f"\nTraining...")
        cl.train(epochs, train_data, test_data)
        print(f"\nSaving {cl.path}...")
        cl.save()

    results = cl.classify(data)
    print(f"\nAccuracy: {results.accuracy(data):.1f}%")
    print(f"\nConfusion Matrix:\n\n{results.confusion_matrix(data)}")
    results.display(32, data)
