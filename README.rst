.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Thu 30 Jan 08:46:53 2014 CET

======================
 Tensorflow Examples
======================

This example package implements simple CNN trainings with tensorflow using the MNIST database.
It is implemented two CNNs:

* Lenet using softmax
* Siamease network (Lenet as a base)

Installation
------------

With tensorflow installed in your environment just run::

  $ python bootstrap.py
  $ ./bin/buildout

Documentation
-------------

For the time being has just one script that trains the MNIST using the Lenet architecture using the softmax
cross entropy loss. Run the command bellow to have some help::

  $ ./bin/train_mnist.py --help # Lenet using softmax
  $ ./bin/train_mnist_siamese.py --help # Lenet using siamese net


It is just the classical stuff.
I haven't added any regularization (L2, L1, dropout, etc )
