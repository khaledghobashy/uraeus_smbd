uraeus.smbd
===========
**Symbolic Multi-Body Dynamics in Python** | A python package for symbolic creation and analysis of constrained multi-body systems

----

.. contents::

Please visit `uraeus.mbd <https://github.com/khaledghobashy/uraeus_mbd>`_ for more information about **audience** , **fields of applications** and **background**.

----

Features
--------
- Creation of symbolic **template-based** and **standalone** multi-body systems using minimal API via python scripting.
- Convenient and easy creation of complex multi-body assemblies.
- Convenient visualization of the system topology as a network graph.
- Viewing the system's symbolic equations in a natural mathematical format using Latex printing.
- Optimization of the system symbolic equations by performing common sub-expressions elimination.
- Creation of symbolic configuration files to facilitate the process of numerical simulation data entry.

Installation
------------
Using the tool on `Colab <https://colab.research.google.com>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Google Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud [2]. 
So, if you do not have an up and running python environment, you still can check out the tool and create multi-body systems seamlessly. 

The *examples* section below, has several ready-to-use Colab notebooks that walks you through a typical modeling process flow.


Using the tool on your machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool needs a valid python 3.6+ environment. If new to scientific computing in python, 
`Anaconda <https://www.anaconda.com/download/>`_ is a recommended free python distribution from Continuum 
Analytics that includes SymPy, SciPy, NumPy, Matplotlib, and many more useful packages for scientific computing, 
which provides a nice coherent platform with most of the tools needed.

Git
'''
As the tool is still under continuous development, cloning this repository is a more versatile way to test and
play with it, until a more stable first release is released. 
This can be done via the following **git** command using the terminal (Linux and Mac) or powershell (Windows).

.. code:: bash

    git clone https://github.com/khaledghobashy/uraeus_smbd.git

This will download the repository locally on your machine. To install the package locally and use it as other
python packages, using the same terminal/powershell run the following command

.. code:: bash

    pip install -e uraeus_smbd
