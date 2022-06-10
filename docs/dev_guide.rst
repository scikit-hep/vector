Developer guide
-----------------
If you are planning to develop ``vector``, or if you want to use the latest commit of ``vector`` on your local machine,
you might want to install it from the source. This installation is not recommended for the users who want to use
the stable version of ``vector``. The steps below describe the installation process of ``vector``'s latest commit. It also
describes how to test ``vector``'s codebase and build ``vector``'s documentation.

Installing vector
===================
We recommend using a virtual environment to install ``vector``. This would isolate the library from your global ``Python``
environment which would be beneficial for reproducing bugs, and the overall development of ``vector``.

Creating a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A virtual environment can be set up and activated using ``venv`` in both ``UNIX`` and ``Windows`` systems.

**UNIX**:

.. code-block::

    python3 -m venv .env
    . .env/bin/activate

**Windows**:

.. code-block::

    python -m venv .env
    .env\bin\activate

Installation
~~~~~~~~~~~~

The developer installation of ``vector`` comes with a lot of options -

* awkward: installs `awkward <https://github.com/scikit-hep/awkward>`_ along with ``vector``
* test: the test dependencies
* test_extras: extra dependencies to run tests on a specific Python version and Operating System
* docs: extra dependencies to build and develop ``vector``'s documentation
* dev: installs the ``awkward`` option + the ``test`` option + `numba <https://github.com/numba/numba>`_
* all: installs dependencies from every option

These options can be used with ``pip`` in the editable (``-e``) mode of installation in the following ways -

.. code-block::

    pip install -e .[dev,test]

For example, if you want to install the ``docs`` dependencies along with the dependencies included above, use -

.. code-block::

    pip install -e .[dev,test,docs]

Furthermore, ``vector`` can also be installed using ``conda``, and this installation also requires using a virtual
environment. ``Vector`` can be installed by executing the following commands -

.. code-block::

    conda env create
    conda activate vector
    conda config --env --add channels conda-forge  # Optional

Adding vector for notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Vector`` can be added to the notebooks using the following commands -

.. code-block::

    python -m ipykernel install --user --name vector # For notebooks

Activating pre-commit
=====================


Testing vector
==============

Documenting vector
==================
``Vector``'s documentation is mainly written in the form of `docstrings <https://peps.python.org/pep-0257/>`_ and
`reStructurredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_. The docstrings include the description,
arguments, examples, return values, and attributes of a class or a function. The ``.rst`` files enable us to render
this documentation on ``vector``'s documentation website.

``Vector`` primarily uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for rendering documentation on its
website. The configuration file (``conf.py``) for ``sphinx`` can be found `here <https://github.com/scikit-hep/vector/blob/main/docs/conf.py>`_.

Ideally, documentation in the form of comments, docstrings, and ``.rst`` files should be added with addition of every
new feature to ``vector``.

Building documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The documentation is located in the ``docs`` folder of the main repository. This documentation can be built using
the ``docs`` dependencies of ``vector`` in the following way -

.. code-block::

    cd docs/
    make clean
    make html

The commands executed above will clean any existing documentation build, and create a new build under the ``docs/_build``
folder. This build can be viewed in any browser by opening up the ``index.html`` file in that browser.

Example notebooks
=================

CI/CD pipeline of vector
========================

