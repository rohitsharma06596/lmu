import pytest
import numpy as np
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.initializers import Constant
from tensorflow.keras import initializers
from tensorflow.keras.layers import RNN
from tensorflow.keras.models import Sequential
from lmu import LMUCell


def test_trainable_A():

    X_train = np.array([[[1]], [[2]], [[3]], [[4]], [[5]]])
    Y_train = np.array([[3], [3], [5], [4], [8]])

    def lmu_layer(train, **kwargs):
        return RNN(
            LMUCell(
                units=10,
                order=1,
                theta=3,
                trainable_A=train,
            ),
            return_sequences=False,
            **kwargs
        )

    model = Sequential()
    model.add(lmu_layer(True, input_shape=[1, 1]))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    A_before = model.layers[0].cell._A

    model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    A_after = model.layers[0].cell._A

    print(A_before)
    print(A_after)

    model = Sequential()
    model.add(lmu_layer(False, input_shape=X_train.shape[1:]))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    A_before = model.layers[0].cell._A

    model.fit(
        X_train,
        Y_train,
        epochs=1,
        batch_size=1,
    )

    A_after = model.layers[0].cell._A

    print(A_before)
    print(A_after)


test_trainable_A()

def test_trainable_B():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        trainable_B=True
    )

    # do test

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        trainable_B=False
    )

    # do test
    


def test_method():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="zoh",
    )


def test_method():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="",
    )
