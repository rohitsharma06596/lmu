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


def lmu_layer(
    trainable_input_encoders=True,
    trainable_hidden_encoders=True,
    trainable_memory_encoders=True,
    trainable_input_kernel=True,
    trainable_hidden_kernel=True,
    trainable_memory_kernel=True,
    trainable_A=False,
    trainable_B=False,
    **kwargs
):
    return RNN(
        LMUCell(
            units=10,
            order=1,
            theta=9999,
            trainable_input_encoders=trainable_input_encoders,
            trainable_hidden_encoders=trainable_hidden_encoders,
            trainable_memory_encoders=trainable_memory_encoders,
            trainable_input_kernel=trainable_input_kernel,
            trainable_hidden_kernel=trainable_hidden_kernel,
            trainable_memory_kernel=trainable_memory_kernel,
            trainable_A=trainable_A,
            trainable_B=trainable_B,
        ),
        return_sequences=False,
        **kwargs
    )


def create_mock_data():
    unfiltered_data = generate_signal()
    filt = nengo.Lowpass(5, default_dt=1)
    filtered_data = filt.filt(unfiltered_data)

    X_train = np.array([filtered_data[0:-1]])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array([filtered_data[-1]])

    return X_train, Y_train


def get_model_param(model, param):
    return model.layers[0].cell.__dict__[param]


def tmp():
    from nengo.utils.filter_design import cont2discrete
    tau = 5
    dt = 1
    A = np.array([[1 / tau]])
    B = np.array([[1 / tau]])
    C = np.array([[1]])
    D = np.array([[0]])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt=dt)
    print(Ad)
    print(Bd)

####################################################################################


def test_untrainability():
    X_train, Y_train = create_mock_data()

    model = Sequential()
    model.add(lmu_layer(input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    A_before = get_model_param(model, 'AT')
    B_before = get_model_param(model, 'BT')
    IE_before = get_model_param(model, 'input_encoders')
    HE_before = get_model_param(model, 'hidden_encoders')
    ME_before = get_model_param(model, 'memory_encoders')
    IK_before = get_model_param(model, 'input_kernel')
    HK_before = get_model_param(model, 'hidden_kernel')
    MK_before = get_model_param(model, 'memory_kernel')

    result = model.fit(
        X_train,
        Y_train,
        epochs=1,
        batch_size=1,
    )

    A_after = get_model_param(model, 'AT')
    B_after = get_model_param(model, 'BT')
    IE_after = get_model_param(model, 'input_encoders')
    HE_after = get_model_param(model, 'hidden_encoders')
    ME_after = get_model_param(model, 'memory_encoders')
    IK_after = get_model_param(model, 'input_kernel')
    HK_after = get_model_param(model, 'hidden_kernel')
    MK_after = get_model_param(model, 'memory_kernel')

    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight.shape)

    assert A_before == A_after
    assert B_before == B_after
    assert IE_before == IE_after
    assert HE_before == HE_after
    assert ME_before == ME_after
    assert IK_before == IK_after
    assert HK_before == HK_after
    assert MK_before == MK_after


def test_trainability():
    X_train, Y_train = create_mock_data()

    model = Sequential()
    model.add(lmu_layer(input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    A_before = get_model_param(model, 'AT')
    B_before = get_model_param(model, 'BT')
    IE_before = get_model_param(model, 'input_encoders')
    HE_before = get_model_param(model, 'hidden_encoders')
    ME_before = get_model_param(model, 'memory_encoders')
    IK_before = get_model_param(model, 'input_kernel')
    HK_before = get_model_param(model, 'hidden_kernel')
    MK_before = get_model_param(model, 'memory_kernel')

    result = model.fit(
        X_train,
        Y_train,
        epochs=1,
        batch_size=1,
    )

    A_after = get_model_param(model, 'AT')
    B_after = get_model_param(model, 'BT')
    IE_after = get_model_param(model, 'input_encoders')
    HE_after = get_model_param(model, 'hidden_encoders')
    ME_after = get_model_param(model, 'memory_encoders')
    IK_after = get_model_param(model, 'input_kernel')
    HK_after = get_model_param(model, 'hidden_kernel')
    MK_after = get_model_param(model, 'memory_kernel')

    # assert A_before == A_after
    # assert B_before == B_after
    # assert IE_before == IE_after
    # assert HE_before == HE_after
    # assert ME_before == ME_after
    # assert IK_before == IK_after
    # assert HK_before == HK_after
    # assert MK_before == MK_after


def test_trainable_A():
    X_train, Y_train = create_mock_data()

    model = Sequential()
    model.add(lmu_layer(trainable_A=True, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    A_before = get_model_param(model, 'AT')

    model.fit(
        X_train,
        Y_train,
        epochs=1,
        batch_size=1,
    )

    A_after = get_model_param(model, 'AT')


def test_trainable_B():
    X_train, Y_train = create_mock_data()
    
    model = Sequential()
    model.add(lmu_layer(trinable_B=True, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    B_before = get_B(model.layers[0])

    result = model.fit(
        X_train,
        to_categorical(Y_train),
        epochs=1,
        batch_size=1,
    )

    B_after = get_B(model.layers[0])


def test_trainable_IE():
    X_train, Y_train = create_mock_data()
    
    model = Sequential()
    model.add(lmu_layer(trainable_input_encoders=True, input_shape=[9999, 1]))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    IE_before = get_model_param(model, 'input_encoders')

    result = model.fit(
        X_train,
        Y_train,
        epochs=1,
        batch_size=1,
    )

    IE_after = get_model_param(model, 'input_encoders')


test_untrainability()
#tmp()
#test_trainable_A()
#test_untrainable_A()
# test_trainable_B()
# test_untrainable_B()
