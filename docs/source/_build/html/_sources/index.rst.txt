.. |br| raw:: html

    <br>

uraeus.smbd
===========

-------------------------------------------------------------------------------

**Symbolic Multi-Body Dynamics in Python** | 
A python package for the symbolic creation and analysis of constrained 
multi-body systems.

.. note:: *The documentation is still under construction ...*

-------------------------------------------------------------------------------

Multi-Body Systems
------------------
In modern literature, multi-body systems refer to modern mechanical systems 
that are often very complex and consist of many components interconnected by 
joints and force elements such as springs, dampers, and actuators. Examples of 
multi-body systems are machines, mechanisms, robotics, vehicles, space 
structures, and bio-mechanical systems. The dynamics of such systems are often 
governed by complex relationships resulting from the relative motion and joint 
forces between the components of the system. [1]_

Therefore, a multi-body system is hereby defined as *a finite number of 
material bodies connected in an arbitrary fashion by mechanical joints that 
limit the relative motion between pairs of bodies*.
Practitioners of multi-body dynamics study the generation and solution of the 
equations governing the motion of such systems [2]_.

-------------------------------------------------------------------------------


Audience and Fields of Application
----------------------------------
Initially, the main targeted audience was the Formula Student community. The 
motive was to encourage a deeper understanding of the modeling processes and 
the underlying theories used in other commercial software packages, which is a 
way of giving back to the community, and supporting the concept of 
"**knowledge share**" adopted there by exposing it to the open-source community
as well.

Currently, the tool aims to serve a wider domain of users with different usage 
goals and different backgrounds, such as students, academic researchers and 
industry professionals.

Fields of application include any domain that deals with the study of 
interconnected bodies, such as:

- Ground Vehicles' Systems.
- Construction Equipment.
- Industrial Mechanisms.
- Robotics.
- Biomechanics.
- etc.


-------------------------------------------------------------------------------

Features
--------
Currently, **uraeus.smbd** provides:

- Creation of symbolic **template-based** and **standalone** multi-body systems
  using minimal API via python scripting.
- Convenient and easy creation of complex multi-body assemblies.
- Convenient visualization of the system topology as a network graph.
- Viewing the system's symbolic equations in a natural mathematical format 
  using Latex printing.
- Optimization of the system symbolic equations by performing common 
  sub-expressions elimination.
- Creation of symbolic configuration files to facilitate the process of 
  numerical simulation data entry.


-------------------------------------------------------------------------------


Guide
-----

.. toctree::
   :maxdepth: 1

   installation
   background
   license

-------------------------------------------------------------------------------

References
----------
.. [1] Shabana, A.A., Computational Dynamics, Wiley, New York, 2010.
.. [2] McPhee, J.J. Nonlinear Dyn (1996) 9: 73. 
       https://doi.org/10.1007/BF01833294


-------------------------------------------------------------------------------

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

