import pytest

from tensorflow.keras.initializers import Constant
from tensorflow.keras import initializers
from lmu import LMUCell


def test_trainable_A():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="", # fill in
    )


def test_trainable_B():

    cell = LMUCell(
        units=212,
        order=256,
        theta=784,
        method="", # fill in
    )


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
