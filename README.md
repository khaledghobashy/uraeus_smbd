# **URAEUS**

# uraeus.smbd

**Symbolic Multi-Body Dynamics in Python** | A python package for symbolic creation and analysis of constrained multi-body systems

*Please visit **[uraeus.mbd]( https://github.com/khaledghobashy/uraeus_mbd )** for more information about **audience** , **fields of applications** and **background**.*

---------------------------------------------------

## Features 

#### Code-Generation

- Creation of symbolic **template-based** and **standalone** multi-body systems using minimal API via python scripting.
- Convenient and easy creation of complex multi-body assemblies.
- Convenient visualization of the system topology as a network graph.
- Viewing the system's symbolic equations in a natural mathematical format using Latex printing.
- Optimization of the system symbolic equations by performing common sub-expressions elimination.
- Creation of symbolic configuration files to facilitate the process of numerical simulation data entry.

---------------------------------------------------

## Installation

### Using the tool on [Colab](https://colab.research.google.com)

Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud [2]. So, if you do not have an up and running python environment, you still can check out the tool and create multi-body systems seamlessly. 

The *examples* section below, has several ready-to-use Colab notebooks that walks you through a typical modeling process flow.

### Using the tool on your machine.

The tool needs a valid python 3.6+ environment. If new to scientific computing in python, [Anaconda](https://www.anaconda.com/download/) is a recommended free python distribution from Continuum Analytics that includes SymPy, SciPy, NumPy, Matplotlib, and many more useful packages for scientific computing, which provides a nice coherent platform with most of the tools needed.

#### Git

As the tool is still under continuous development, cloning this repository is a more versatile way to test and play with it, until a more stable first release is released. This can be done via the following git commands from the command line.

```bash
git clone https://github.com/khaledghobashy/uraeus-smbd.git
```

```bash
git pull origin master
```

Or alternatively, download the repository as a **zip** file and extract it on your machine.

To install the package locally and use it as other python packages, open a terminal or a powershell and `cd` to the package location, then run

```bash
pip install -e uraeus-smbd
```



#### Pip

The package can be also installed from PyPi using

```bash
pip install uraeus.smbd
```



#### Conda

*To Do*

---------------------------------------------------

### Approach

The philosophy of the tool is to isolate the model creation process form the actual numerical and computational representation of the system that will be used in the simulation process. This is done through the ideas of **symbolic computing** and **code-generation** as well be shown below.

#### Model Creation

The topology of the system is represented as a multi-directed graph, where each node represents a body and each edge represents a connection between the end nodes, where this connection may represents a joint, actuator or a force element. This serves mainly two aspects:

1. A natural way to create and represent the topology of a given multi-body system.
2. A convenient way to abstract the system programmatically, where all the topological data of the system are stored in a graph.

The tool achieves this by making heavy use the [NetworkX](https://networkx.github.io/documentation/stable/index.html) python package to create topology graphs and to construct the governing equations of the system. The equations themselves are represented symbolically using [SymPy](https://www.sympy.org/en/index.html), which is a Python library for symbolic mathematics.

The combination of both, NetworkX and SymPy, provides the tool with a very simple, easy-to-use and convenient interface for the process of model creation and topology design, where the user only focuses on the validity of the system topology in hand, as he thinks only in terms of the topological components - bodies, joints, actuators and forces-, without the burden of frequent numerical inputs for each component, or how the actual system is configured in space. In short, the tool divide the typical model creation process in halves, the system topology design and the system configuration assignment.

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

**uraeus.smbd** is distributed under the 3-clause BSD license. See the [LICENSE](LICENSE) file for details.