import numpy as np


class Iterator:

    def __init__(self, dataset, batch_size=32):
        self.datset = dataset
        self.max = len(dataset)
        self.batch_size = batch_size
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx + + self.batch_size >= self.max - 1:
            np.random.shuffle(self.datset)
            self.idx = 0
        self.idx += self.batch_size
        return self.datset[self.idx:self.idx + self.batch_size]
