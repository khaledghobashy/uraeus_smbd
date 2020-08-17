.. |br| raw:: html

    <br>

Pendulum
========

Model Description
^^^^^^^^^^^^^^^^^
A pendulum is a weight suspended from a pivot so that it can swing freely. 
It consists of two bodies, the pendulum and the ground, attached together 
through a pin/revolute joint. More general information about the pendulum can 
be found `here <https://en.wikipedia.org/wiki/Pendulum>`_.

-------------------------------------------------------------------------------

Topology Layout
^^^^^^^^^^^^^^^
The mechanism consists of 1 Body + 1 Ground. Therefore, total system 
coordinates -including the ground- is 
:math:`n=n_b\times7 = 2\times7 = 14`, where :math:`n_b` is the total number of 
bodies in the system. [1]_

List of Bodies
^^^^^^^^^^^^^^
The list of bodies is given below:

- Pendulum body :math:`body`.

Connectivity
^^^^^^^^^^^^
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

Degrees of Freedom
^^^^^^^^^^^^^^^^^^
The degrees of freedom of the system can be calculated as:

.. math::

    n-( n_{c,rev}+n_{c,P}+n_{c,g}) = 14 - (5 + (1 \times 1) + 7) = 14 - 13 = 1

Where:

- :math:`n_{c,P}` represents the constraints due to euler-parameters 
  normalization equations.
- :math:`n_{c,g}` represents the constraints due to the ground constraints.
- :math:`n_{c,rev}` represents the constraints due to the revolute joint.

-------------------------------------------------------------------------------

Symbolic Topology
^^^^^^^^^^^^^^^^^
In this section, we create the symbolic topology that captures the topological 
layout that we discussed earlier. |br|
Defining the topology is very simple. We start by importing the 
``standalone_topology`` class and create a new instance that represents our 
symbolic model, let it be ``sym_model``. 
Then we start adding the components we discussed earlier, starting by the 
bodies, then the joints, actuators and forces, and thats it. |br|
These components will be represented symbolically, and therefore there is no 
need for any numerical inputs at this step.

The system is stored in a form of a network graph that stores all the data 
needed for the assemblage of the system equations later. But even before the 
assemblage process, we can gain helpful insights about our system as well be 
shown.

Directories Construction
''''''''''''''''''''''''

.. code:: python

    # standard library imports
    import os

    # getting directory of current file and specifying the directory
    # where data will be saved
    os.makedirs(os.path.join("model", "symenv", "data"), exist_ok=True)
    
    data_dir = os.path.abspath("model/symenv/data")


Topology Construction
'''''''''''''''''''''

.. code:: python

    # uraeus imports
    from uraeus.smbd.systems import standalone_topology, configuration

    # ============================================================= #
    #                       Symbolic Topology
    # ============================================================= #

    # Creating the symbolic topology as an instance of the
    # standalone_topology class
    project_name = 'pendulum'
    sym_model = standalone_topology(project_name)

    # Adding Bodies
    # =============
    sym_model.add_body('body')

    # Adding Joints
    # =============
    sym_model.add_joint.revolute('a', 'ground', 'rbs_body')


Symbolic Characteristics
''''''''''''''''''''''''

Now, we can gain some insights about our topology using our ``sym_model`` 
instance. By accessing the ``topology`` attribute of the ``sym_model``, we can 
visualize the connectivity of the model as a network graph using the 
``sym_model.topology.draw_constraints_topology()`` method, where the nodes 
represent the bodies, and the edges represent the joints, forces and/or 
actuators between the bodies.

.. code:: python

    sym_model.topology.draw_constraints_topology()

Also, we can check the system's number of generalized coordinates  :math:`n`  
and number of constraints  :math:`nc`.

.. code:: python

    print(sym_model.topology.n, sym_model.topology.nc)


Assembling
''''''''''

This is the last step of the symbolic building process, where we ask the 
system to assemble the governing equations, which will be used then in the 
code generation for the numerical simulation, as well as further symbolic 
manipulations.

Also, we can export/save a *pickled* version of the model.

.. code:: python

    # Assembling and Saving model
    sym_model.save(data_dir)
    sym_model.assemble()

.. note:: The equations' notations will be discussed in another part of the 
          documentation.

-------------------------------------------------------------------------------

Symbolic Configuration
^^^^^^^^^^^^^^^^^^^^^^
In this step we define a symbolic configuration of our symbolic topology. 
As you may have noticed in the symbolic topology building step, we only cared 
about the **topology**, thats is the system bodies and their connectivity, and 
we did not care explicitly with how these components are configured in space.

In order to create a valid numerical simulation session, we have to provide the 
system with its numerical configuration needed, for example, the joints' 
locations and orientations. The symbolic topology in its raw form will require 
you to manually enter all these numerical arguments, which can be cumbersome 
even for smaller systems. This can be checked by checking the configuration 
inputs of the symbolic configuration as ``sym_config.config.input_nodes``

Here we start by stating the symbolic inputs we wish to use instead of the 
default inputs set, and then we define the relation between these newly defined 
arguments and the original ones.

.. code:: python

    # ============================================================= #
    #                     Symbolic Configuration
    # ============================================================= #

    # Symbolic configuration name.
    config_name = "%s_cfg"%project_name

    # Symbolic configuration instance.
    sym_config = configuration(config_name, sym_model)

    # Adding the desired set of UserInputs
    # ====================================
    sym_config.add_point.UserInput('p1')
    sym_config.add_point.UserInput('p2')

    sym_config.add_vector.UserInput('v')

    # Defining Relations between original topology inputs
    # and our desired UserInputs.
    # ===================================================

    # Revolute Joint (a) location and orientation
    sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_p1',))
    sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_v',))

    # Creating Geometries
    # ===================
    sym_config.add_scalar.UserInput('radius')

    sym_config.add_geometry.Sphere_Geometry('body', ('hps_p2', 's_radius'))
    sym_config.assign_geometry_to_body('rbs_body', 'gms_body')

    # Exporting the configuration as a JSON file
    sym_config.export_JSON_file(data_dir)

.. note:: The details of this process will be discussed in another part of the 
          documentation.

-------------------------------------------------------------------------------

.. [1] **uraeus.smbd** uses `euler-parameters 
       <https://en.wikibooks.org/wiki/Multibody_Mechanics/Euler_Parameters>`_ 
       -which is a 4D unit quaternion- to represents bodies orientation in 
       space. This makes the generalized coordinates used to fully define a 
       body in space to be **7,** instead of **6**, it also adds an algebraic 
       equation to the constraints that ensures the unity/normalization of the 
       body quaternion. This is an important remark as the calculations of the 
       degrees-of-freedom depends on it.