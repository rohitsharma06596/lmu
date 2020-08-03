import pytest

from tensorflow.keras.layers import RNN, Bidirectional, StackedRNNCells
from lmu import LMUCell


def test_keras_rnn():
    
    layer = RNN(
        LMUCell(
            units=212,
            order=256,
            theta=784,
        )
    )

    assert layer.cell.__class__.__name__ == 'LMUCell'
    assert layer.cells[0].__class__.__name__ == 'LMUCell'


def test_keras_bidirectional():
    
    forward_layer = RNN(
        LMUCell(
            units=212,
            order=256,
            theta=784,
        )
    )

    layer = Bidirectional(
        forward_layer
    )
    print(dir(layer))
    assert layer.cells[0].__class__.__name__ == 'LMUCell'


def test_keras_stacked_rnn():

    cells = [LMUCell(units=212, order=256, theta=784) for _ in range(10)]
    layer = StackedRNNCells(cells)

    assert layer.cells[0].__class__.__name__ == 'LMUCell'


test_keras_bidirectional()
