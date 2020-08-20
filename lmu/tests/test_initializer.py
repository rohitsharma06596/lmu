import pytest
from lmu import Legendre

def test_legendre_init():
    initializer = Legendre()
    values = initializer.__call__([2, 4])

    print(values)
    assert values == [[ 1., 1., 1., 1.], [-1., -0.33333333, 0.33333333, 1.]]
