Tutorials
=========

This section presents some tutorials illustrating the process of creating 
standalone and template-based symbolic models.

The figure below shows a high-level **activity diagram** of a typical usage 
flow of the **uraeus** framework, where we have three swim-lanes representing 
a main layer of the three activity layers of the framework.

Here we are concerned with the first *swim-lane* that represents the 
**symbolic environment** layer and the **symbolic model creation** activity.

.. image:: uraeus_activity_diagram-Swimlane.png
  :width: 600
  :alt: Alternative text

-------------------------------------------------------------------------------

Symboilc Models
^^^^^^^^^^^^^^^

The **uraeus.smbd** python package provides two main types of symbolic models 
to be constructed, **standalone models**, and **template-based** models.

**Standalone models** stands for symbolic models that are fully described in 
**one** topology graph, where all the bodies and connections are already 
defined in the model graph, and therefore, the model is fully independent and 
does not need any other topological information.

**Template-based models**, on the other hand,  are not fully described in one 
topology, as they need to be assembled with other templates to form a complete 
assembled topology. For example, this is the case for creating a full vehicle 
assembly, where we model the vehicle subsystems as templates, then we assemble 
them together to form the desired vehicle assembly.

The creation of template-based models' database can be found in 
`uraeus.fsae <https://github.com/khaledghobashy/uraeus_fsae>`_, 
which is an under-development multi-body systems database for formula-style 
vehicles implemented in the **uraeus** framework.

-------------------------------------------------------------------------------

Guide
-----

.. toctree::
   :maxdepth: 1

   standalone_topologies/index.rst
   template_based_topologies/index.rst
