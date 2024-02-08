import enum

import numpy as np


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.prev_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        if (validation_loss - self.prev_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
        self.prev_loss = validation_loss


class Architecture(enum.Enum):
    XFORMER = "xformer"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
