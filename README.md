# **URAEUS**

**_Symbolic creation and analysis of constrained multi-body systems in python_**

## Description

**uraeus** is a python package developed for the symbolic creation and analysis of constrained [multi-body systems](https://en.wikipedia.org/wiki/Multibody_system).

A multi-body system is hereby defined as *a finite number of material bodies connected in an arbitrary fashion by mechanical joints that limit the relative motion between pairs of bodies*. Practitioners of multi-body dynamics study the generation and solution of the equations governing the motion of such systems [1].

### Audience and Fields of Application

Initially, the main targeted audience was the **Formula Student** community. The motive was *encouraging a deeper understanding of the modelling processes and the underlying theories used in other commercial software packages*, which is a way of giving back to the community, and supporting the concept of *"knowledge share"* adopted there by exposing it to the open-source community as well.

Currently, the tool aims to serve a wider domain of users with different usage goals and different backgrounds, such as students, academic researchers and industry professionals.

Fields of application include any domain that deals with the study of interconnected bodies, such as:

- Ground Vehicles' Systems.
- Construction Equipment.
- Industrial Mechanisms.
- Robotics.
- Biomechanics.
- .. etc.

---------------------------------------------------

### Background

#### What is the problem to be solved?

The primary interest in multi-body dynamics is to analyze the system behaviour for given inputs. In analogy with control systems; a multi-body system can be thought as a **_system_** subjected to some **_inputs_** producing some **_outputs_**. These three parts of the problem are dependent on the analyst end goal of the analysis and simulation. 

#### How is the system physics abstracted mathematically?

An unconstrained body in space is normally defined using 6 generalized coordinates defining its location and orientation. For example, a system of 10 bodies requires 60 generalized coordinates to be fully defined, which in turn requires 60 *independent equations* to be solved for these  _unknown_ generalized coordinates.

The way we achieve a solution for the system is dependent on the type of study we are performing. Mainly we have **four types** of analysis that are of interest for a given multi-body system. These are:

- **Kinematic Analysis**</br>
  *"How does the whole system move if we moved this particular body ?"*
- **Inverse Dynamic Analysis**</br>
  *"What are the forces needed to achieve this motion we just did ?"*
- **Equilibrium Analysis**</br>
  *"How does the system look if we did nothing ?"*
- **Dynamic Analysis**</br>
  *"Now we gave it a force, how does it behave ?"*

Each analysis type -or question- can be modelled by a set of algebraic and/or differential equations that can be solved for the system generalized states (positions, velocities and accelerations). A more detailed discussion of each analysis type will be provided in another documentation.

### Approach

The philosophy of the tool is to isolate the model creation process form the actual numerical and computational representation of the system that will be used in the simulation process. This is done through the ideas of **symbolic computing** and **code-generation** as well be shown below.

#### Model Creation

The topology of the system is represented as a multi-directed graph, where each node represents a body and each edge represents a connection between the end nodes, where this connection may represents a joint, actuator or a force element. This serves mainly two aspects:

1. A natural way to create and represent the topology of a given multi-body system.
2. A convenient way to abstract the system programmatically, where all the topological data of the system are stored in a graph.

The tool achieves this by making heavy use the [NetworkX](https://networkx.github.io/documentation/stable/index.html) python package to create topology graphs and to construct the governing equations of the system. The equations themselves are represented symbolically using [SymPy](https://www.sympy.org/en/index.html), which is a Python library for symbolic mathematics.

The combination of both, NetworkX and SymPy, provides the tool with a very simple, easy-to-use and convenient interface for the process of model creation and topology design, where the user only focuses on the validity of the system topology in hand, as he thinks only in terms of the topological components - bodies, joints, actuators and forces-, without the burden of frequent numerical inputs for each component, or how the actual system is configured in space. In short, the tool divide the typical model creation process in halves, the system topology design and the system configuration assignment.

#### Code Generation and Model Simulation

The process of performing actual simulations on the created model requires the generation of a valid numerical and computational code of the created model. This is done by taking in the symbolic model and create a valid code files written in the desired programming language with the desired programming paradigm and structure.

These numerical environments are decoupled from this package and developed fairly independently, as each numerical environment is responsible for the translation of the developed symbolic models into valid numerical code and what features it aims to provide for the users.

The development of such environments in different languages requires a good grasp of several aspects such as :

- Good knowledge of the symbolic models' interfaces and structure.
- Good knowledge of the target language.
- Appropriate environment architecture/structure that serves the intended usage requirements.
- Good knowledge of the available linear algebra and math libraries for that language.
- Design for minimal dependencies on 3rd parties libraries.
- Simple API for usage and simple build process for compiled languages.

_**Note**: The development of such environments is discussed in a separate documentation for those interested in developing their own._

#### Conclusion

Several benefits of the adopted approach can be stated here, but the major theme here is the flexibility and modularity, in both software usage and software development. These can be summarized as follows:

- The distinction between the topology design phase and the configuration assignment phase, which gives proper focus for each at its' own.
- Natural adoption of the template-based modelling theme that emerges from the use of network-graphs to represent the system, which allows convenient assemblage of several graphs to form a new system. 
- Uncoupled simulation environment, where the symbolic equations generated form the designed topology is free to be written in any programming language with any desired numerical libraries.

---------------------------------------------------

### Features 

Currently, the tool provides:

- Creation of symbolic template-based and stand-alone multi-body systems using minimal API via python scripting.
- Convenient and easy creation of complex multi-body assemblies.
- Convenient visualization of the system topology as a network graph.
- Viewing the system's symbolic equations in a natural mathematical format using Latex printing.
- Optimization of the system symbolic equations by performing common sub-expressions elimination.
- Creation of symbolic configuration files to facilitate the process of numerical simulation data entry.

---------------------------------------------------

---------------------------------------------------
## Installation

### Using the tool on [Colab](https://colab.research.google.com)

Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud [2]. So, if you do not have an up and running python environment, you still can check out the tool and create multi-body systems seamlessly. 

The examples section below, has several ready-to-use Colab notebooks that walks you through a typical modelling process flow.

### Using the tool on your machine.

The tool needs a valid python 3.6+ environment. If new to scientific computing in python, [Anaconda](https://www.anaconda.com/download/) is a recommended free python distribution from Continuum Analytics that includes SymPy, SciPy, NumPy, Matplotlib, and many more useful packages for scientific computing, which provides a nice coherent platform with most of the tools needed.

#### Git

As the tool is still under continuous development, cloning this repository is a more versatile way to test and play with it, until a more stable first release is released. This can be done via the following git commands from the command line.

```bash
git clone https://github.com/khaledghobashy/uraeus.git
```

```bash
git pull origin master
```

Or alternatively, download the repository as a **zip** file and extract it on your machine.

Then, if creating a new python session to use the tool, add the directory path where the tool exists to the python system path.

```python
try:
    import uraeus
except ModuleNotFoundError:
    import sys
	pkg_path = 'path/to/uraeus'
    sys.path.append(pkg_path)

# the package can now be imported as uraeus, e.g. :
# from uraeus.systems import standalone_topology
```



#### Pip

*To Do.*

#### Conda

*To Do*

---------------------------------------------------

---------------------------------------------------
## Usage Examples & Tutorials

### Ready-to-Use Notebooks & Tutorials

This is a list of ready-to-use jupyter notebooks that walks you through the typical flow of the tool modelling process. The [**examples**](https://github.com/khaledghobashy/smbd/tree/master/examples/standalone_models/notebooks/) directory in this repository is planned to include updated versions of working models that can be statically viewed on github, downloaded on your machine or to be ran directly on Colab.

### Standalone Studies

- **Spatial Four-bar**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/spatial_fourbar/spatial_fourbar.ipynb) | [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/spatial_fourbar/spatial_fourbar.ipynb))
- **Spatial Slider-Crank**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/spatial_slider_crank/spatial_slider_crank.ipynb)| [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/spatial_slider_crank/spatial_slider_crank.ipynb))
- **Double-Wishbone Suspension**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_wishbone_suspension/double_wishbone_direct_acting.ipynb)| [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_wishbone_suspension/double_wishbone_direct_acting.ipynb#scrollTo=A5aeLp5S45eh))
- **Double-Four-bar Mechanism**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_fourbar/double_fourbar.ipynb)| [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_fourbar/double_fourbar.ipynb#scrollTo=A5aeLp5S45eh))
- **Simple Pendulum**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/simple_pendulum/simple_pendulum.ipynb)| [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/simple_pendulum/simple_pendulum.ipynb#scrollTo=A5aeLp5S45eh))
- **Double Pendulum**. ([**GitHub**](https://github.com/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_pendulum/double_pendulum.ipynb)| [**Colab**](https://colab.research.google.com/github/khaledghobashy/smbd/blob/master/examples/standalone_models/notebooks/double_pendulum/double_pendulum.ipynb#scrollTo=A5aeLp5S45eh))

### Template-Based Projects

*To be discussed ...*

---------------------------------------------------

---------------------------------------------------

### Detailed Example - Spatial Fourbar Mechanism
----------------------------------
Below is code sample that walks you through the process of building a standalone symbolic topology and configuration as well as the generation of numerical simulation environments.

This model will be created as a **standalone** topology and project. What this means is that model topological data is fully encapsulated in one topology graph and no need for any topological data from other external systems, which is the case for **template-based** topologies.

This also means that the project files/database is self-contained, unlike the **template-based** topologies that need to be organized in a shared database.

#### Initializing Project Structure

Currently, a standalone project is structured using three top-level directories inside a given ```parent_dir```; these are

-  ```/numenv``` : </br>
  Directory of the numerical environments to be generated.
- ``` /results``` :</br>
  Directory to store the results of numerical simulations if needed.
- ``` /config_inputs``` :</br>
  Directory to store the numerical inputs used in numerical simulations.

To create a standalone project :

```python
from smbd.systems import standalone_project

parent_dir = '' # current working directory

project = standalone_project(parent_dir)
project.create()
```



#### Building the Symbolic Topology.
We start by importing the ```standalone_topology``` class from the ```systems``` module to create our symbolic model instance.
```python
from smbd.systems import standalone_topology

model_name = 'fourbar'
sym_model = standalone_topology(model_name)
```
We then start  constructing our system by adding the bodies, joints, actuators and forces.
```python
# Adding Bodies
sym_model.add_body('l1')
sym_model.add_body('l2')
sym_model.add_body('l3')

# Adding Joints
sym_model.add_joint.revolute('a', 'ground', 'rbs_l1')
sym_model.add_joint.spherical('b', 'rbs_l1', 'rbs_l2')
sym_model.add_joint.universal('c', 'rbs_l2', 'rbs_l3')
sym_model.add_joint.revolute('d', 'rbs_l3', 'ground')

# Adding Actuators
sym_model.add_actuator.rotational_actuator('act', 'jcs_a')
```
And that's it; we have just created a symbolic topology that represents our fourbar mechanism. The topology graph of the system can be visualized by the method ```sym_model.topology.draw_constraints_topology()```
Also we can check the number of constraint equations, generalized coordinates and the estimated degrees of freedom of the system.
To finalize this step, we call the ```assemble()``` method to construct the governing equations symbolically.

```python
sym_model.assemble()
```
We can check the system equations by accessing the appropriate topology attributes.
```python
# Position level constraint equations.
sym_model.topology.pos_equations
# System Jacobian of the position level constraint equations.
sym_model.topology.jac_equations
```
#### Building the symbolic configuration.
The next step is to create a symbolic configuration of our symbolic model, but what is this symbolic configuration? </br>
You may have noticed that we did not care explicitly about how our system is configured in space, we did not care about how our bodies or joints are located or oriented or how we can define these configuration parameters, all we cared about is only the topological connectivity. These configuration parameters already got generated automatically based on the used components. For example, the creation of a symbolic body -*body l1* *for example*- generates automatically the following symbolic parameters:

- ```m_rbs_l1```:  body mass.
- ```Jbar_rbs_l1```: inertia tensor.
- ```R_rbs_l1```: body reference point location.
-  ```Rd_rbs_l1```: body translational velocity.
-  ```Rdd_rbs_l1```: body translational acceleration.
- ```P_rbs_l1```: body orientation.
- ```Pd_rbs_l1```: body orientation 1st  rate of change.
- ```Pdd_rbs_l1```: body orientation 2nd  rate of change.

where the ```rbs_``` initial is short for *rigid body single*. If the body is mirrored, the system will create two bodies with the initials ```rbr_``` and ```rbl_``` for right and left respectively.

The same happens for edges' components -joints, actuators and forces- where each component is responsible for creating its own configuration symbolic parameters.

These parameters are extracted from the symbolic topology to form the primary configuration layer that represents the needed user inputs for any given simulation. The benefit of the symbolic configuration is that we can construct our own layer of inputs that we desire to use in the numerical simulation and define the relations between these inputs and the primary parameters extracted from the topology components. This is best shown by example.

Our fourbar mechanism is simply visualized as three links and a ground that are connected at four distinct points, **a**, **b**, **c** and **d**. We can simply get directly the numerical values of these points in space much easier than -for example- getting directly the orientation of the two axes used to define the universal joint used to connect **l2** with **l3**. 

The idea is to construct a directed relational graph that maps the required primary configuration to a set of new configuration parameters that may be easier and more convenient to specify directly.

We start by creating our configuration instance
```python
from smbd.systems import configuration
sym_config = configuration('%s_cfg'%model_name, sym_model)
```
Now we can check the primary configuration parameters extracted the from the symbolic topology by ```sym_config.config.input_nodes``` which returns a list of strings containing the inputs' parameters names.

Now, we create our desired user inputs.

```python
# Adding the desired set of UserInputs
# ====================================

sym_config.add_point.UserInput('a')
sym_config.add_point.UserInput('b')
sym_config.add_point.UserInput('c')
sym_config.add_point.UserInput('d')

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')
```
After that, we set the relations between the primary configuration parameters and our custom configuration inputs.
```python
# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint (a) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_a',))
sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_x',))

# Spherical Joint (b) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_b', ('hps_b',))
sym_config.add_relation.Equal_to('ax1_jcs_b', ('vcs_z',))

# Universal Joint (c) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_c', ('hps_c',))
sym_config.add_relation.Oriented('ax1_jcs_c', ('hps_b', 'hps_c'))
sym_config.add_relation.Oriented('ax2_jcs_c', ('hps_c', 'hps_b'))

# Revolute Joint (d) location and orientation
sym_config.add_relation.Equal_to('pt1_jcs_d', ('hps_d',))
sym_config.add_relation.Equal_to('ax1_jcs_d', ('vcs_y',))

```
*__Note__: The set of configuration parameters of each component and their naming convention will be discussed in a separate documentation*

The first line of the above code-block adds a relation that sets the location of joint ```pt1_jsc_a``` to be ```Equal_to``` the user-input location point ```hps_a```, where the fifth line adds a relation that sets the orientation of the first axis of the universal joint ```ax1_jsc_c``` to be ```Oriented``` along the user-input location points ```hps_b``` and ```hps_c```. The rest of the statements follows the same convention.

An optional and recommended step is to create symbolic geometries and assign these geometries to topology bodies to automatically evaluate the bodies configuration parameters stated earlier. Also this will be used to generate a python-blender script that can be used in blender to create 3D visualizations in blender later.
```python
# links radius
sym_config.add_scalar.UserInput('links_ro')

# Link 1 geometry
sym_config.add_geometry.Cylinder_Geometry('l1', ('hps_a','hps_b','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l1', 'gms_l1')

# Link 2 geometry
sym_config.add_geometry.Cylinder_Geometry('l2', ('hps_b','hps_c','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l2', 'gms_l2')

# Link 3 geometry
sym_config.add_geometry.Cylinder_Geometry('l3', ('hps_c','hps_d','s_links_ro'))
sym_config.assign_geometry_to_body('rbs_l3', 'gms_l3')
```
The last step is to ```assemble``` the symbolic configuration and extract the updated set of inputs to a .csv file.
```python
sym_config.assemble()
sym_config.extract_inputs_to_csv(parent_dir)
```
---------------------------------------------------

---------------------------------------------------
## Roadmap

*To be discussed ...*

## Implementation Details

*The **Implementation Details** will be provided in a separate documentation and linked here*

## Theoretical Basis

*The **Theoretical Basis** will be provided in a separate documentation and linked here*

## Support

As the tool is developed and maintained by one developer for now, if you have any inquiries, do not hesitate to contact me at khaled.ghobashy@live.com or kh.ghobashy@gmail.com

## References
[1] : McPhee, J.J. Nonlinear Dyn (1996) 9: 73. https://doi.org/10.1007/BF01833294

[2] : https://colab.research.google.com/notebooks/welcome.ipynb

## License

**uraeus** is distributed under the 3-clause BSD license. See the [LICENSE](LICENSE) file for details.

