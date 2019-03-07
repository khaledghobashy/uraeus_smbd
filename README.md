# ASURT - CDT

**ASU Racing Team - Computational Dynamics Tool**

## Description

**asurt_cdt** is a python package developed for the creation, simulation and visualization of multibody systems. Those are systems that consist of various interconnected bodies, such as ground and space vehicles, as well as robotics and industrial mechanisms. The analysis of such systems leads to highly nonlinear mathematical formulations that requires the use of advanced numerical, and computational methods.

### Background
#### What is Computational Dynamics?
Computational Dynamics is a rapidly developing field that brings together applied mathematics (especially numerical analysis), computer science, and scientific or engineering applications. Multibody Dynamics can be considered as a sub-field of computational dynamics.
#### What is the problem to be solved?
The primary interest in multibody dynamics is to analyze the system behavior for given inputs. In analogy with control systems; a multi-body system can be thought as a **_system_** subjected to some **_inputs_** producing some **_outputs_**. These three parts of the problem are dependent on the analyst end goal of the analysis and simulation. 
#### How is the system physics abstracted mathematically and computationally?
To keep it as simple as possible. An unconstrained body in space is normally defined using six generalized coordinates defining its location and orientation. For example, a system of 10 bodies will require 60 unknown generalized coordinates to be fully defined which in turn requires 60 independent algebraic equations to be solved for these generalized coordinates. These equations are constructed form the joints existing in the system.

### Approach

