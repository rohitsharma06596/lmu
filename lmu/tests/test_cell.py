import pytest

from tensorflow.keras.initializers import Constant
from tensorflow.keras import initializers
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
    
    assert cell.units == 212
    assert cell.order == 256
    assert cell.theta == 784


test_create_cell()
