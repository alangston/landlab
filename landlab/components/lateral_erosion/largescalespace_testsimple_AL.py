# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:33:54 2019

@author: abby
"""

import numpy as np
import matplotlib.pyplot as plt

import pytest
from numpy import testing

from landlab import HexModelGrid, RasterModelGrid, imshow_grid
from landlab.components import DepressionFinderAndRouter, FlowAccumulator, SpaceLargeScaleEroder, LateralEroderSolo, PriorityFloodFlowRouter, ChannelProfiler
print("done importing")
import time

tic=time.time()
# print(frog)
"""
Test that model matches the bedrock-alluvial analytical solution
for slope/area relationship at steady state:
S=((U * v_s * (1 - F_f)) / (K_sed * A^m) + U / (K_br * A^m))^(1/n).

Also test that the soil depth everywhere matches the bedrock-alluvial
analytical solution at steady state:
H = -H_star * ln(1 - (v_s / (K_sed / (K_br * (1 - F_f)) + v_s))).
"""



# z = mg.add_zeros("node", "topographic__elevation")
# br = mg.add_zeros("node", "bedrock__elevation")
# soil = mg.add_zeros("node", "soil__depth")

# mg["node"]["topographic__elevation"] += (
#     mg.node_y / 100000 + mg.node_x / 100000 + np.random.rand(len(mg.node_y)) / 10000
# )


#%%
num_rows = 20
num_columns = 20
node_spacing = 100.0
mg = RasterModelGrid((num_rows, num_columns), xy_spacing=node_spacing)
node_next_to_outlet = num_columns + 1
np.random.seed(seed=5000)
z = mg.add_zeros("topographic__elevation", at="node")
_ = mg.add_zeros("soil__depth", at="node")
mg.at_node["soil__depth"][mg.core_nodes] = 2.0
_ = mg.add_zeros("bedrock__elevation", at="node")
mg.at_node["bedrock__elevation"] += (
    mg.node_y / 10. + mg.node_x / 10. + np.random.rand(len(mg.node_y)) / 10.
)

mg.at_node["bedrock__elevation"][:] = mg.at_node["topographic__elevation"]
mg.at_node["topographic__elevation"][:] += mg.at_node["soil__depth"]
mg.set_closed_boundaries_at_grid_edges(
    bottom_is_closed=True,
    left_is_closed=True,
    right_is_closed=True,
    top_is_closed=True,
)

# fig = plt.figure()
# plot = plt.subplot()
# _ = imshow_grid(
#     mg,
#     "topographic__elevation",
#     plot_name="Topographic Elevation",
#     var_name="Topographic Elevation",
#     var_units=r"m",
#     grid_units=("m", "m"),
#     cmap="terrain",
# )

# print(frog)
mg.set_watershed_boundary_condition_outlet_id(
    0, mg.at_node['topographic__elevation'], -9999.0
)
fr = PriorityFloodFlowRouter(mg, flow_metric='D8', suppress_out = True)
sp = SpaceLargeScaleEroder(
    mg,
    K_sed=0.01,
    K_br=0.001,
    F_f=0.0,
    phi=0.0,
    H_star=1.0,
    v_s=5.0,
    m_sp=0.5,
    n_sp=1.0,
    sp_crit_sed=0,
    sp_crit_br=0,
)
#below is creating the lateral erosion component. Note the Kl/Kv ratio is really high here (3, rather than 1, 1.5, etc.)
le = LateralEroderSolo(mg, Kv=0.001, Kl_ratio = 3)
timestep = 50.0
elapsed_time = 0.0
count = 0
run_time = 3e4
uplift_rate = 0.0002    #mm/year
sed_flux = np.zeros(int(run_time // timestep))
while elapsed_time < run_time:
    fr.run_one_step()
    _ = sp.run_one_step(dt=timestep)
    sedflux_space = np.copy(mg.at_node["sediment__flux"])
    #below is the lateral erosion component
    _ = le.run_one_step_basic(dt=timestep)
    sedflux_lateral = mg.at_node["sediment__flux"]
    ### test for match with sed fluxes before and after lateral erosion component
    # lat ero component should add to sed flux from eroded lateral node to downstream node
    # testing.assert_array_almost_equal(
    #     sedflux_space,
    #     sedflux_lateral,
    #     decimal=8,
    #     err_msg="sediment flux IS being changed in lateral erosion module",
    #     verbose=True,
    # )
    # print(frog)
    sed_flux[count] = mg.at_node["sediment__flux"][node_next_to_outlet]
    mg.at_node["topographic__elevation"][mg.core_nodes] += uplift_rate * timestep
    elapsed_time += timestep
    count += 1

#%% Plot the results.

fig = plt.figure()
plot = plt.subplot()
_ = imshow_grid(
    mg,
    "topographic__elevation",
    plot_name="Topographic Elevation",
    var_name="Topographic Elevation",
    var_units=r"m",
    grid_units=("m", "m"),
    cmap="terrain",
)
_ = plt.figure()
_ = imshow_grid(
    mg,
    "sediment__flux",
    plot_name="Sediment flux",
    var_name="Sediment flux",
    var_units=r"m$^3$/yr",
    grid_units=("m", "m"),
    cmap="terrain",
)
_ = plt.figure()
_ = imshow_grid(
    mg,
    "lateral_erosion__depth_cum",
    plot_name="Cumulative Lateral Erosion",
    var_name="Cumu. lat. ero.",
    var_units=r"m",
    grid_units=("m", "m"),
    cmap="terrain",
)
fig = plt.figure()
sedfluxplot = plt.subplot()
_ = sedfluxplot.plot(np.arange(len(sed_flux)) * timestep, sed_flux, color="k", linewidth=1.0)
_ = sedfluxplot.set_xlabel("Time [yr]")
_ = sedfluxplot.set_ylabel(r"Sediment flux [m$^3$/yr]")

#%%
toc=time.time()
print("elapsed time = ", toc-tic)