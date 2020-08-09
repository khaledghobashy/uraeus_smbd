.. uraeus.smbd documentation master file, created by
   sphinx-quickstart on Thu Aug  6 08:44:30 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

    <br>

uraeus.smbd
===========
.. note:: *Documentation is still under construction ...*

------------------------------------------------------------------------------------------------------

**Symbolic Multi-Body Dynamics in Python** | 
A python package for the creation and analysis of symbolic constrained multi-body systems.

------------------------------------------------------------------------------------------------------

Multi-Body Systems
------------------
In modern literature, multi-body systems refer to modern mechanical systems that are often very complex
and consist of many components interconnected by joints and force elements such as springs, dampers, and 
actuators. Examples of multi-body systems are machines, mechanisms, robotics, vehicles, space structures, 
and bio-mechanical systems. The dynamics of such systems are often governed by complex relationships 
resulting from the relative motion and joint forces between the components of the system. [1]_

Therefore, a multi-body system is hereby defined as *a finite number of material bodies connected in an 
arbitrary fashion by mechanical joints that limit the relative motion between pairs of bodies*.
Practitioners of multi-body dynamics study the generation and solution of the equations governing the 
motion of such systems [2]_.

------------------------------------------------------------------------------------------------------


Audience and Fields of Application
----------------------------------
Initially, the main targeted audience was the Formula Student community. The motive was to encourage a 
deeper understanding of the modeling processes and the underlying theories used in other commercial 
software packages, which is a way of giving back to the community, and supporting the concept of 
"**knowledge share**" adopted there by exposing it to the open-source community as well.

Currently, the tool aims to serve a wider domain of users with different usage goals and different 
backgrounds, such as students, academic researchers and industry professionals.

Fields of application include any domain that deals with the study of interconnected bodies, such as:

- Ground Vehicles' Systems.
- Construction Equipment.
- Industrial Mechanisms.
- Robotics.
- Biomechanics.
- etc.


------------------------------------------------------------------------------------------------------

Installation
------------
The package needs a valid python 3.6+ environment. If new to scientific computing in python, 
`Anaconda <https://www.anaconda.com/download/>`_ is a recommended free python distribution from 
Continuum Analytics that includes SymPy, SciPy, NumPy, Matplotlib, and many more useful packages for 
scientific computing, which provides a nice coherent platform with most of the tools needed.

Pip
^^^
.. code:: bash

    pip install uraeus.smbd


Git
^^^
As the package is still under continuous development, cloning this repository is a more versatile way 
to test and play with it, until a more stable first release is released. 
This can be done via the following **git** command using the terminal (Linux and Mac) or powershell 
(Windows).

.. code:: bash

    git clone https://github.com/khaledghobashy/uraeus_smbd.git

This will download the repository locally on your machine. To install the package locally and use it 
as other python packages, using the same terminal/powershell run the following command:

.. code:: bash

    pip install -e uraeus_smbd

------------------------------------------------------------------------------------------------------

Guide
-----

.. toctree::
   :maxdepth: 2

   background
   license

------------------------------------------------------------------------------------------------------

References
----------
.. [1] Shabana, A.A., Computational Dynamics, Wiley, New York, 2010.
.. [2] McPhee, J.J. Nonlinear Dyn (1996) 9: 73. https://doi.org/10.1007/BF01833294


------------------------------------------------------------------------------------------------------


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

