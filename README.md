# ASURT - CDT

**ASU Racing Team - Computational Dynamics Tool**

## Description
**asurt_cdt** is a python package developed for the creation, simulation and visualization of multibody systems. Those are systems that consist of various interconnected bodies, such as ground vehicles, space structures, robotics and industrial mechanisms. The analysis of such systems leads to highly nonlinear mathematical formulations that requires the use of advanced numerical, and computational methods.

### Background
#### What is Computational Dynamics?
Computational Dynamics is a rapidly developing field that brings together applied mathematics (especially numerical analysis), computer science, and scientific or engineering applications. Multibody Dynamics can be considered as a sub-field of computational dynamics.
#### What is the problem to be solved?
The primary interest in multibody dynamics is to analyze the system behavior for given inputs. In analogy with control systems; a multi-body system can be thought as a **_system_** subjected to some **_inputs_** producing some **_outputs_**. These three parts of the problem are dependent on the analyst end goal of the analysis and simulation. 
#### How is the system physics abstracted mathematically and computationally?
To keep it as simple as possible. An unconstrained body in space is normally defined using 6 generalized coordinates defining its location and orientation. For example, a system of 10 bodies requires 60 generalized coordinates to be fully defined, which in turn requires 60 independent equations to be solved for these  -unknown- generalized coordinates.

### Approach
The philosophy of the tool is to isolate the model creation process form the actual numerical and computational representation of the system that will be used in the simulation process. This is done through the ideas of symbolic computing and code-generation as well be shown below.

#### Model Creation

The topology of the system is represented as a multi-directed graph, where each node represents a body and each edge represents a connection between the end nodes, where this connection may represents a joint, actuator or a force element. This serves mainly two aspects:

1. A natural way to create and represent the topology of a given multibody system.
2. A convenient way to abstract the system programmatically, where all the topological data of the system are stored in a graph.

The tool achieves this by making heavy use the [NetworkX](https://networkx.github.io/documentation/stable/index.html) python package to create topology graphs and to construct the governing equations of the system. The equations themselves are represented symbolically by using [SymPy](https://www.sympy.org/en/index.html), which is a Python library for symbolic mathematics.

The combination of both, NetworkX and SymPy, provides the tool with a very simple, easy-to-use and convenient interface for the process of model creation and topology design, where the user only focuses on the validity of the system topology in hand as he thinks only in terms of the topological components - bodies, joints, actuators and forces-, without the burden of frequent numerical inputs for each component, or how the actual system is configured in space. In short, the tool divide the typical model creation process in halves, the system topology design and the system configuration assignment.

`The tool adopts the *"programming to an interface instead of an implementation"* methodology which abstracts`

#### Code Generation

The process of performing actual simulations on the created model requires the generation of a valid numerical and computational code of the created model. This is done by taking in the created symbolic model and create a valid code files written in the desired programming language with the desired programming paradigm and structure. For now, The tool provides a Python Code Generator that generate an object oriented code structure of the symbolic multibody system

#### Model Simulation



