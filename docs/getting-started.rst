Getting started
***************

Installation
============

To install NengoLMU, we recommend using ``pip``.

.. code:: bash

   pip install lmu
   pip install -e .

Installing other packages
-------------------------

The above steps will only install the hard dependencies.
The optional NengoLMU features require other packages.
These can be installed either through
Python's own package manager, ``pip``.

- Additional decoder solvers and other speedups
  require SciPy and scikit-learn.
- Running the test suite requires
  pytest, Matplotlib, and Jupyter.
- Building the documentation requires
  Sphinx, NumPyDoc and guzzle_sphinx_theme.

These additional dependencies can be installed
through ``pip`` when installing NengoLMU.

.. code-block:: bash

   pip install lmu[docs]  # Needed to run example notebooks
   pip install lmu[optional]  # Needed to use the Legendre initializer 

Usage
=====

The base class in NengoLMU is the cell contained in
`lmu.LMUCell`. To create a new batch of ``LMUCell``:

.. testcode::

   import lmu
   
   cells = lmu.LMUCell(
       units=10,
       order=256,
       theta=784,
   )

Note these are arbitrary values for ``units``, ``order``, 
and ``theta``.

Creating LMU Layers
-------------------

We can connect cells to layers
in order to integrate the lmu 
within larger models.

.. testcode::

   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import RNN

   model = Sequential()
   layer = RNN(cells)
   model.add(layer)

Next steps
==========

* If you're wondering how this works and you're not
  familiar with the Neural Engineering Framework,
  we recommend reading
  `this technical overview <http://compneuro.uwaterloo.ca/files/publications/stewart.2012d.pdf>`_.
