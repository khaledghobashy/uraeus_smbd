# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:17:44 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as dwb
import use_cases.parametric_templates.templates.parallel_link_steering as steer
import use_cases.parametric_templates.templates.front_axle_testrig as testrig


SU_sym = subsystem('SU',dwb.template)
ST_sym = subsystem('ST',steer.template)
TR_sym = subsystem('TR',testrig.template)

assembled = assembly('front_axle')

assembled.add_subsystem(SU_sym)
assembled.add_subsystem(ST_sym)
assembled.add_subsystem(TR_sym)

assembled.assign_virtual_body('SU.vbr_steer','ST.rbr_rocker')
assembled.assign_virtual_body('TR.vbr_upright','SU.rbr_upright')
assembled.assign_virtual_body('TR.vbr_hub','SU.rbr_hub')
assembled.assign_virtual_body('TR.vbs_steer_gear','ST.rbl_rocker')

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()


#from bokeh.io import show, output_file
#from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
#from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
#from bokeh.palettes import Spectral4
#import networkx as nx
#
#G = assembled.constraints_graph.to_undirected(as_view=True)
#
#plot = Plot(plot_width=400, plot_height=400,
#            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
#plot.title.text = "Graph Interaction Demonstration"
#
#plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
#
#graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
#
#graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
#graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
#graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
#
#graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
#graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
#graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
#
#graph_renderer.selection_policy = NodesAndLinkedEdges()
#graph_renderer.inspection_policy = EdgesAndLinkedNodes()
#
#plot.renderers.append(graph_renderer)
#
#output_file("interactive_graphs.html")
#show(plot)

