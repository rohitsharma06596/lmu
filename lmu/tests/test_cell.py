import pytest

from tensorflow.keras.layers import RNN, Bidirectional, StackedRNNCells
from tensorflow.keras.initializers import Constant
from tensorflow.keras import activations, initializers
from lmu import LMUCell


def test_create_cell():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        input_encoders_initializer=Constant(1),
        hidden_encoders_initializer=Constant(0),
        memory_encoders_initializer=Constant(0),
        input_kernel_initializer=Constant(0),
        hidden_kernel_initializer=Constant(0),
        memory_kernel_initializer="glorot_normal",
    )
    print(initializers.get(Constant(1)))
    assert cell.units == 212
    assert cell.order == 256
    assert cell.theta == 784
    assert cell.input_encoders_initializer == initializers.get(Constant(1))
    #assert cell.hidden_encoders_initializer == Constant(0)
    #assert cell.memory_encoders_initializer == Constant(0)
    #assert cell.input_kernel_initializer == Constant(0)
    #assert cell.hidden_kernel_initializer == Constant(0)
    assert cell.memory_kernel_initializer == "glorot_normal"


test_create_cell()
