import pytest

from tensorflow.keras.initializers import Constant
from tensorflow.keras import activations
from lmu import LMUCell
from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
import numpy as np


def test_default_param_assignment():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
    )

    assert cell.units == 212
    assert cell.order == 256
    assert cell.theta == 784
    assert cell.method == "zoh"
    assert cell.trainable_input_encoders == True
    assert cell.trainable_hidden_encoders == True
    assert cell.trainable_memory_encoders == True
    assert cell.trainable_input_kernel == True
    assert cell.trainable_hidden_kernel == True
    assert cell.trainable_memory_kernel == True
    assert cell.trainable_A == False
    assert cell.trainable_B == False
    assert cell.input_encoders_initializer.distribution == "uniform"
    assert cell.hidden_encoders_initializer.distribution == "uniform"
    assert cell.memory_encoders_initializer.value == 0
    assert cell.input_kernel_initializer.distribution == "truncated_normal"
    assert cell.hidden_kernel_initializer.distribution == "truncated_normal"
    assert cell.memory_kernel_initializer.distribution == "truncated_normal"
    assert cell.hidden_activation == activations.get("tanh")


def test_custom_param_assignment():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="euler",
        trainable_input_encoders=False,
        trainable_hidden_encoders=False,
        trainable_memory_encoders=False,
        trainable_input_kernel=False,
        trainable_hidden_kernel=False,
        trainable_memory_kernel=False,
        trainable_A=True,
        trainable_B=True,
        input_encoders_initializer=Constant(1),
        hidden_encoders_initializer=Constant(0),
        memory_encoders_initializer=Constant(0),
        input_kernel_initializer=Constant(0),
        hidden_kernel_initializer=Constant(0),
        memory_kernel_initializer="glorot_normal",
        hidden_activation="relu",
    )

    assert cell.units == 212
    assert cell.order == 256
    assert cell.theta == 784
    assert cell.method == "euler"
    assert cell.trainable_input_encoders == False
    assert cell.trainable_hidden_encoders == False
    assert cell.trainable_memory_encoders == False
    assert cell.trainable_input_kernel == False
    assert cell.trainable_hidden_kernel == False
    assert cell.trainable_memory_kernel == False
    assert cell.trainable_A == True
    assert cell.trainable_B == True
    assert cell.input_encoders_initializer.value == 1
    assert cell.hidden_encoders_initializer.value == 0
    assert cell.memory_encoders_initializer.value == 0
    assert cell.input_kernel_initializer.value == 0
    assert cell.hidden_kernel_initializer.value == 0
    assert cell.memory_kernel_initializer.distribution == "truncated_normal"
    assert cell.hidden_activation == activations.get("relu")


def test_attr_assignment():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
    )
    
    realizer = Identity()
    factory = LegendreDelay

    tmp = cell._realizer_result == realizer(factory(theta=784, order=256))
    tmp = tmp.all()
    print(tmp)

    # assert (cell._realizer_result == realizer(factory(theta=784, order=256))).all()
    assert tmp
    assert cell._ss == cont2discrete(cell._realizer_result.realization, dt=1.0, method="zoh")
    assert cell._A == cell._ss.A - np.eye(256)
    assert cell._B == cell._ss.B
    assert cell._C == cell._ss.C
    assert cell.state_size == (212, 256)
    assert cell.output_size == 212


def test_build():
    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
    )

    cell.build([2, 4, 6])

    assert cell.input_encoders.shape == [6, 1]
    assert cell.hidden_encoders.shape == [212, 1]
    assert cell.memory_encoders.shape == [256, 1]
    assert cell.input_kernel.shape == [6, 212]
    assert cell.hidden_kernel.shape == [212, 212]
    assert cell.memory_kernel.shape == [256, 212]
    assert cell.AT.shape == [256, 256]
    assert cell.BT.shape == [1, 256]
    assert cell.built == True


def test_call():
    cell = LMUCell(
        units=10,
        order=5,
        theta=10,
    )

    cell.build([1, 10])

    inputs = np.array([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]])
    states = [
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[2, 2, 2, 2, 2]]
    ]

    print(cell.call(inputs, states))


def test_config():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
    )

    config = cell.get_config()
    
    assert config["units"] == 212
    assert config["order"] == 256
    assert config["theta"] == 784
    assert config["method"] == "zoh"
    assert config["trainable_input_encoders"] == True
    assert config["trainable_hidden_encoders"] == True
    assert config["trainable_memory_encoders"] == True
    assert config["trainable_input_kernel"] == True
    assert config["trainable_hidden_kernel"] == True
    assert config["trainable_memory_kernel"] == True
    assert config["trainable_A"] == False
    assert config["trainable_B"] == False
    assert cell.input_encoders_initializer.distribution == "uniform"
    assert cell.hidden_encoders_initializer.distribution == "uniform"
    assert cell.memory_encoders_initializer.value == 0
    assert cell.input_kernel_initializer.distribution == "truncated_normal"
    assert cell.hidden_kernel_initializer.distribution == "truncated_normal"
    assert cell.memory_kernel_initializer.distribution == "truncated_normal"
    assert config["hidden_activation"] == activations.get("tanh")


#test_default_param_assignment()
#test_custom_param_assignment()
#test_attr_assignment()
#test_build()
test_call()
#test_config()
