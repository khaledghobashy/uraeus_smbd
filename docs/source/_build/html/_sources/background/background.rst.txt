.. |br| raw:: html

    <br>

Background and Approach
=======================

The Problem
-----------

What is Multi-Body Dynamics?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As mentioned earlier, a multi-body system is hereby defined as *a finite number
of material bodies connected in an arbitrary fashion by mechanical joints that 
limit the relative motion between pairs of bodies*, where practitioners of 
multi-body dynamics study the generation and solution of the equations 
governing the motion of such systems.

What is the problem to be solved?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One of the primary interests in multi-body dynamics is to analyze the behavior 
of a given multi-body system under the effect of some inputs. In analogy with
control systems; a multi-body system can be thought as a system subjected to 
some inputs producing some outputs. 
These three parts of the problem are dependent on the analyst end goal of the 
analysis and simulation.

How is the system physics abstracted mathematically?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An unconstrained body in space is normally defined using 6 generalized 
coordinates defining its location and orientation in space. 
For example, a system of 10 bodies requires 60 generalized coordinates to be 
fully defined, which in turn requires 60 independent equations to be solved for
these unknown generalized coordinates.

The way we achieve a solution for the system is dependent on the type of study 
we are performing. 
Mainly we have four types of analysis that are of interest for a given 
multi-body system. These are:

- **Kinematic Analysis** |br| 
  "How does the whole system move if we moved this particular body ?"

- **Inverse Dynamic Analysis** |br| 
  "What are the forces needed to achieve this motion we just did ?"

- **Equilibrium Analysis** |br| 
  "How does the system look if we did nothing ?"

- **Dynamic Analysis** |br| 
  "Now we gave it a force, how does it behave ?"

Each analysis type -or question- can be modeled by a set of algebraic and/or 
differential equations that can be solved for the system generalized states 
(positions, velocities and accelerations).

.. note:: 
  A more detailed discussion of each analysis type will be provided in another 
  part of the documentation and linked here.

-------------------------------------------------------------------------------

The Approach
------------

The philosophy of the **uraeus** framework is to isolate the model creation 
process form the actual numerical and computational representation of the 
system that will be used in the numerical simulation process. 
This is done through the concepts of **symbolic computing** and 
**code-generation**. 
The uraeus.smbd package is responsible for the symbolic creation of multi-body
systems.

Symbolic Topology
^^^^^^^^^^^^^^^^^
The System Topology is a description of the connectivity relationships between 
the bodies in a given multi-body system. These relationships represent the 
system constraints that limit the relative motion between the system bodies 
and produce the desired kinematic behavior.

The package abstracts the topology of a given system as a multi-directed graph, 
where each **node** represents a **body** and each **edge** represents a 
**connection** between the end nodes, where this 
connection may represents a **joint**, an **actuator** or a **force element**. 
|br|
No numerical inputs is needed at that step, the focus is only on the validity 
of the topological design of the system, **not** how it is configured in space.

This problem statement and approach leads to the following important landmarks:

- 

