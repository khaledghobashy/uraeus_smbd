

from smbd.interfaces.scripting import standalone_topology, configuration
from smbd.numenv.python.codegen import generators
from smbd.numenv.cpp_eigen.codegen import generators as cppgen

model = standalone_topology('grader')

model.add_body('chassis')
model.add_body('arm', mirrored=True)
model.add_body('bar')
model.add_body('drawbar')
model.add_body('upper_cyl', mirrored=True)
model.add_body('bottom_cyl', mirrored=True)
model.add_body('center_shift1_cyl')
model.add_body('center_shift2_cyl')

model.add_body('table')
model.add_body('inc_member')
model.add_body('plate')

model.add_joint.fixed('fix', 'rbs_chassis', 'ground')


# Fourbar Mechanism
model.add_joint.revolute('rev_arm', 'rbs_chassis', 'rbr_arm', mirrored=True)
model.add_joint.universal('uni_arm', 'rbr_arm', 'rbs_bar')
model.add_joint.spherical('sph_arm', 'rbl_arm', 'rbs_bar')

# Cylinders Mechanism
model.add_joint.cylinderical('cyl_1', 'rbr_upper_cyl', 'rbr_bottom_cyl', mirrored=True)
model.add_joint.cylinderical('cyl_centershift', 'rbs_center_shift1_cyl', 
                             'rbs_center_shift2_cyl')


# Parallel Mechanism
model.add_joint.spherical('sph_drawbar', 'rbs_drawbar', 'rbs_chassis')
model.add_joint.universal('uni_drawbar_cyl', 
                          'rbs_drawbar', 'rbr_bottom_cyl', mirrored=True)
model.add_joint.universal('uni_arm_cyl', 
                          'rbr_arm', 'rbr_upper_cyl', mirrored=True)


# CenterShift 
model.add_joint.universal('uni_drawbar_cyl', 
                          'rbs_drawbar', 'rbs_center_shift2_cyl')
model.add_joint.universal('uni_arm_CS1', 
                          'rbr_arm', 'rbs_center_shift1_cyl')


model.add_joint.revolute('rev_table', 'rbs_drawbar', 'rbs_table')
model.add_joint.revolute('rev_table_inc', 'rbs_inc_member', 'rbs_table')

model.add_joint.translational('trns_plate', 'rbs_plate', 'rbs_inc_member')



print(model._mbs.n, model._mbs.nc)
model._mbs.draw_constraints_topology()

model.assemble()

#model.assemble()

