.. |br| raw:: html

    <br>

Background and Approach
=======================

The Problem
-----------

**What is the problem to be solved?** |br|
One of the primary interests in multi-body dynamics is to analyze the behavior of a given multi-body 
system under the effect of some inputs. In analogy with control systems; a multi-body system can be 
thought as a system subjected to some inputs producing some outputs. 
These three parts of the problem are dependent on the analyst end goal of the analysis and simulation.

**How is the system physics abstracted mathematically?** |br|
An unconstrained body in space is normally defined using 6 generalized coordinates defining its location
and orientation in space. For example, a system of 10 bodies requires 60 generalized coordinates to be 
fully defined, which in turn requires 60 independent equations to be solved for these unknown generalized 
coordinates.

The way we achieve a solution for the system is dependent on the type of study we are performing. 
Mainly we have four types of analysis that are of interest for a given multi-body system. These are:

- **Kinematic Analysis** |br| "How does the whole system move if we moved this particular body ?"

- **Inverse Dynamic Analysis** |br| "What are the forces needed to achieve this motion we just did ?"

- **Equilibrium Analysis** |br| "How does the system look if we did nothing ?"

- **Dynamic Analysis** |br| "Now we gave it a force, how does it behave ?"

Each analysis type -or question- can be modeled by a set of algebraic and/or differential equations 
that can be solved for the system generalized states (positions, velocities and accelerations).

A more detailed discussion of each analysis type will be provided in another documentation.

The Approach
------------

