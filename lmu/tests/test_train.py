import pytest
import numpy as np
import nengo
from nengo.processes import WhiteSignal

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras import initializers
from tensorflow.keras.layers import RNN
from tensorflow.keras.models import Sequential
from lmu import LMUCell

####################################################################################

def generate_signal():

    model = nengo.Network()
    with model:
        inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
        pre = nengo.Ensemble(60, dimensions=2)
        nengo.Connection(inp, pre)
        post = nengo.Ensemble(60, dimensions=2)
        conn = nengo.Connection(pre, post, function=lambda x: np.random.random(2))
        inp_p = nengo.Probe(inp)
        pre_p = nengo.Probe(pre, synapse=0.01)
        post_p = nengo.Probe(post, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(10.0)

    return sim.data[inp_p].T[0]


def lmu_layer(trainable_A, trainable_B, **kwargs):
        return RNN(
            LMUCell(
                units=10,
                order=1,
                theta=9999,
                trainable_A=trainable_A,
                trainable_B=trainable_B,
            ),
            return_sequences=False,
            **kwargs
        )


def get_A(layer):
    return layer.cell._A


def get_B(layer):
    return layer.cell._B

####################################################################################


def test_trainable_A():

    unfiltered_data = generate_signal()
    filt = nengo.Lowpass(5, default_dt=1)
    filtered_data = filt.filt(unfiltered_data)

    X_train = np.array([filtered_data[0:-1]])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array([filtered_data[-1]])

    model = Sequential()
    model.add(lmu_layer(True, False, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    A_before = get_A(model.layers[0])

    result = model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    A_after = get_A(model.layers[0])

    print(A_before)
    print(A_after)


def test_untrainable_A():

    unfiltered_data = generate_signal()
    filt = nengo.Lowpass(5, default_dt=1)
    filtered_data = filt.filt(unfiltered_data)

    X_train = np.array([filtered_data[0:-1]])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array([filtered_data[-1]])

    model = Sequential()
    model.add(lmu_layer(False, False, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    A_before = get_A(model.layers[0])

    result = model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    A_after = get_A(model.layers[0])

    print(A_before)
    print(A_after)


def test_trainable_B():

    unfiltered_data = generate_signal()
    filt = nengo.Lowpass(5, default_dt=1)
    filtered_data = filt.filt(unfiltered_data)

    X_train = np.array([filtered_data[0:-1]])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array([filtered_data[-1]])

    model = Sequential()
    model.add(lmu_layer(False, True, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    B_before = get_B(model.layers[0])

    result = model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    B_after = get_B(model.layers[0])

    print(B_before)
    print(B_after)


def test_untrainable_B():

    unfiltered_data = generate_signal()
    filt = nengo.Lowpass(5, default_dt=1)
    filtered_data = filt.filt(unfiltered_data)

    X_train = np.array([filtered_data[0:-1]])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array([filtered_data[-1]])

    model = Sequential()
    model.add(lmu_layer(False, False, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    B_before = get_B(model.layers[0])

    result = model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    B_after = get_B(model.layers[0])

    print(B_before)
    print(B_after)


def test_method():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="zoh",
    )


def test_hidden_activation():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="",
    )


test_trainable_A()
test_untrainable_A()
