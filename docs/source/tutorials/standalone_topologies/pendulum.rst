Pendulum
========

Model Description
^^^^^^^^^^^^^^^^^
A pendulum is a weight suspended from a pivot so that it can swing freely. 
It consists of two bodies, the pendulum and the ground, attached together 
through a pin/revolute joint. More general information about the pendulum can 
be found `here <https://en.wikipedia.org/wiki/Pendulum>`_.

Topology Layout
^^^^^^^^^^^^^^^
The mechanism consists of 1 Body + 1 Ground. Therefore, total system 
coordinates -including the ground- is 
:math:`n=n_b\times7 = 2\times7 = 14`, where :math:`n_b` is the total number of 
bodies in the system. [1]_

The list of bodies is given below:

- Pendulum body :math:`body`.

The system connectivity is as follows:

- Pendulum :math:`body` is connected to the **ground** by a revolute joint, 
  resulting in constraint equations :math:`n_{c,rev} = 5`.


.. list-table:: Connectivity Table
   :widths: 25 25 50 25 25
   :header-rows: 1

   * - Joint Name
     - Body i
     - Body j
     - Joint Type
     - :math:`n_c`

   * - a
     - :math:`ground`
     - :math:`body`
     - Revolute
     - :math:`5`

The degrees of freedom of the system can be calculated as:

.. math::

    n-( n_{c,rev}+n_{c,P}+n_{c,g}) = 14 - (5 + (1 \times 1) + 7) = 14 - 13 = 1

Where:

- :math:`n_{c,P}` represents the constraints due to euler-parameters 
  normalization equations.
- :math:`n_{c,g}` represents the constraints due to the ground constraints.
- :math:`n_{c,rev}` represents the constraints due to the revolute joint.

-------------------------------------------------------------------------------

.. [1] The tool uses `euler-parameters 
       <https://en.wikibooks.org/wiki/Multibody_Mechanics/Euler_Parameters>`_ 
       -which is a 4D unit quaternion- to represents bodies orientation in 
       space. This makes the generalized coordinates used to fully define a 
       body in space to be **7,** instead of **6**, it also adds an algebraic 
       equation to the constraints that ensures the unity/normalization of the 
       body quaternion. This is an important remark as the calculations of the 
       degrees-of-freedom depends on it.