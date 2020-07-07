.. NengoLMU documentation master file, created by
   sphinx-quickstart on Mon Jul  6 09:47:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

********
NengoLMU
********

We propose a novel memory cell for recurrent neural networks that dynamically maintains information across long windows of time using relatively few resources. The Legendre Memory Unit (LMU) is mathematically derived to orthogonalize its continuous-time history – doing so by solving d coupled ordinary differential equations (ODEs), whose phase space linearly maps onto sliding windows of time via the Legendre polynomials up to degree d − 1.

.. toctree::
   :maxdepth: 2

   getting-started
   user-guide
   examples
   contributing
   project

* :ref:`genindex`
