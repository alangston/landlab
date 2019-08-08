# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:50:43 2019

@author: abby


Doc tests and unit tests for lateral erosion.
"""

from landlab.components import FlowAccumulator, LateralEroder
from numpy import testing
from landlab import RasterModelGrid, CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
import numpy as np


def test_lateral_erosion_and_node():
    """
    Test that sets up a simple, pre-defined drainage network and compares
    the lateral node that is eroded, the volume of lateral eorsion, and the elevation
    of the landscape after one timestep to known values.
    """
    nr = 5
    nc = 5
    dx = 1
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)
    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

    z = mg.add_zeros('node', 'topographic__elevation')
    loading_vector = np.linspace(1, 4, num=nr)
    ramp = np.repeat(loading_vector, nc)
    # the tweaks to elevation below make lateral node at node 7
    z += ramp
    z[11] -= 0.9
    z[12] -= 0.4
    z[8] -= 0.001
    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)
    latero = LateralEroder(mg, latero_mech="UC", Kv=0.1, Kl_ratio=1.5)
    fa.accumulate_flow()
    (mg, dzlat,) = latero.run_one_step(mg, dt=1.,)

    vlname = mg['node']['volume__lateral_erosion']
    # predicted volume of lateral eorsion
    pred_vollat = 0.00045158164
    # predicted elevation after 1 time step
    pred_zafter = np.array([1., 1., 1., 1., 1., 1.75, 1.675, 1.675, 1.6731154, 1.75, 2.5, 1.66418779,
                            2.06181623, 2.4249, 2.5, 3.25, 3.085, 3.13332738, 3.16868272, 3.25, 4., 4., 4., 4., 4.])

    testing.assert_array_almost_equal(
        mg.at_node['topographic__elevation'],
        pred_zafter,
        decimal=8,
        err_msg="LatEro basic erosion test failed",
        verbose=True,
    )
    testing.assert_array_almost_equal(
        vlname[7],
        pred_vollat,
        decimal=8,
        err_msg="LatEro volume lateral erosion failed",
        verbose=True,
    )


def test_matches_detlim_solution():
    """
    Test that model matches the detachment-limited analytical solution
    for slope/area relationship at steady state: S=(U/K_br)^(1/n)*A^(-m/n).
    """

    U = 0.005
    dt = 10
    nr = 5
    nc = 5
    dx = 10
    Kbr = 0.001
    # instantiate grid
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)
    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

    z = mg.add_zeros('node', 'topographic__elevation')
    ir2 = np.random.uniform(low=0.0, high=0.5, size=(z.size))
    loading_vector = np.linspace(1, 4, num=nr)
    ramp = np.repeat(loading_vector, nc)
    ramp += ir2
    z[:] += ramp
    n_sp = 1.0
    m_sp = 0.5

    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)  # "DepressionFinderAndRouter", router="D8")

    # set alph = 0.0 to make run detachment limited and set Kl_ratio = 1.0 for
    # equal vertical and lateral bedrock erosion coefficients
    latero = LateralEroder(mg, latero_mech="TB", Kv=Kbr,
                           solver="basic", alph=0.0, Kl_ratio=1.0)

    for i in range(2000):
        fa.run_one_step()  # flow accumulator
        # erode the landscape with lateral erosion
        (mg, dzlat) = latero.run_one_step(mg, dt,)
        mg.at_node['topographic__elevation'][mg.core_nodes] += U * dt

    num_slope = mg.at_node["topographic__steepest_slope"][mg.core_nodes]
    analytical_slope = np.power(U / Kbr, 1.0 / n_sp) * np.power(
        mg.at_node["drainage_area"][mg.core_nodes], -m_sp / n_sp
    )

    testing.assert_array_almost_equal(
        num_slope,
        analytical_slope,
        decimal=4,
        err_msg="LatEro detachment-limited steady state test failed",
        verbose=True,
    )


def test_ss_sed_flux():
    """
    Test that sediment flux of lateral erosion model matches steady state
    analytical solution, Qs = U * A
    """
    U = 0.0005  # m/yr
    dt = 100  # years
    nr = 5
    nc = 5
    dx = 10
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)

    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

    z = mg.add_zeros('node', 'topographic__elevation')
    ir2 = np.random.uniform(low=0.0, high=0.5, size=(z.size))
    loading_vector = np.linspace(1, 2.5, num=nr)
    ramp = np.repeat(loading_vector, nc)
    ramp += ir2
    z[:] += ramp
    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)
    latero = LateralEroder(mg, latero_mech="UC", alph=1.5,
                           Kv=0.001, Kl_ratio=1.0, solver="basic")

    for i in range(2000):
        fa.run_one_step()  # flow accumulator
        (mg, dzlat) = latero.run_one_step(mg, dt,)
        mg.at_node['topographic__elevation'][mg.core_nodes] += U * \
            dt  # uplift the landscape

    # compare numerical and analytical sediment flux solutions
    num_sedflux = mg.at_node["qs"][mg.core_nodes]
    analytical_sedflux = U * mg.at_node["drainage_area"][mg.core_nodes]

    # test for match with analytical sediment flux.
    testing.assert_array_almost_equal(
        num_sedflux,
        analytical_sedflux,
        decimal=2,
        err_msg="LatEro transport-limited sediment flux test failed",
        verbose=True,
    )


def test_variable_bedrock_K():
    """
    Test that model matches the detachment-limited analytical solution
    for slope/area relationship at steady state: S=(U/K_br)^(1/n)*A^(-m/n),
    with variable bedrock erodibility.

    """
    U = 0.005  # m/yr
    dt = 1  # years

    nr = 5
    nc = 5
    nnodes = nr * nc
    dx = 10
    # instantiate grid
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)
    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

    z = mg.add_zeros('node', 'topographic__elevation')
    loading_vector = np.linspace(1, 4, num=nr)
    ramp = np.repeat(loading_vector, nc)
    ramp += np.random.random_sample(nnodes) * 0.8
    z[:] += ramp
    n_sp = 1.0
    m_sp = 0.5
    # below, array of different bedrock erodibilities
    Kvar = 0.006 * np.ones(nr * nc)
    Kvar[0:9] = 0.06
    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)  # "DepressionFinderAndRouter", router="D8")

    # ***NOTE, YOU MUST USE ADAPTIVE TIME STEPPER FOR variable K, or you may get strange
    # topography
    latero = LateralEroder(mg, latero_mech="UC", Kv=Kvar,
                           solver="adaptive", alph=0.0, Kl_ratio=0.0)

    for i in range(2000):
        fa.run_one_step()  # flow accumulator
        # erode the landscape with lateral erosion
        (mg, dzlat) = latero.run_one_step(mg, dt,)
        mg.at_node['topographic__elevation'][mg.core_nodes] += U * dt

    num_slope = mg.at_node["topographic__steepest_slope"][mg.core_nodes]
    analytical_slope = np.power(U / Kvar[mg.core_nodes], 1.0 / n_sp) * np.power(
        mg.at_node["drainage_area"][mg.core_nodes], -m_sp / n_sp
    )

    testing.assert_array_almost_equal(
        num_slope,
        analytical_slope,
        decimal=4,
        err_msg="LatEro variable bedrock test failed",
        verbose=True,
    )


def test_latero_steady_inlet():
    """
    Test that time steady inputs of drainage area and sediment at a designated inlet node
    functions in the lateral erosion component.
    """
    U = 0.005  # m/yr
    dt = 10  # years
    nr = 5
    nc = 5
    dx = 10
    # instantiate grid
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)
    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

    z = mg.add_zeros('node', 'topographic__elevation')
    loading_vector = np.linspace(1, 2.5, num=nr)
    ramp = np.repeat(loading_vector, nc)
    ramp += np.random.random_sample(mg.number_of_nodes) * 0.8
    z += ramp

    # set inlet node = true, provide inlet node id, inlet drainage area, and inlet qs.
    latero = LateralEroder(mg, latero_mech="UC", Kv=0.001, Kl_ratio=1.0, inlet_on=True,
                           inlet_node=17, inlet_area=500, qsinlet=2.5)
    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)
    for i in range(2000):
        fa.run_one_step()  # flow accumulator
        # erode the landscape with lateral erosion
        (mg, dzlat) = latero.run_one_step(mg, dt,)
        mg.at_node['topographic__elevation'][mg.core_nodes] += U * \
            dt  # uplift the landscape

    da = mg.at_node['surface_water__discharge'] / dx**2
    num_sedflux = mg.at_node["qs"]
    analytical_sedflux = U * da

    # test for match with analytical sediment flux. note that the values are off a little
    # because of the lateral erosion
    testing.assert_array_almost_equal(
        num_sedflux,
        analytical_sedflux,
        decimal=2,
        err_msg="LatEro inlet transport-limited sediment flux test failed",
        verbose=True,
    )


def test_latero_timevary_inlet():
    """
    Test that time varying drainage area and sediment to inlet function in the 
    lateral erosion component works. Comparison to steady state solution and 
    proscribed change in drainage area at inlet node.

    """
    U = 0.005
    dt = 5
    nr = 5
    nc = 5
    dx = 10
    mg = RasterModelGrid((nr, nc), xy_spacing = dx)
    for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge, mg.nodes_at_right_edge):
        mg.status_at_node[edge] = CLOSED_BOUNDARY
    for edge in (mg.nodes_at_bottom_edge):
        mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY
    z = mg.add_zeros('node', 'topographic__elevation')
    loading_vector = np.linspace(1, 2.5, num=nr)
    ramp = np.repeat(loading_vector, nc)
    ramp += np.random.random_sample(mg.number_of_nodes) * 0.8
    z += ramp

    latero = LateralEroder(mg, latero_mech="UC", alph=0.01, Kv=0.01, Kl_ratio=0.1, solver="adaptive", inlet_on=True,
                           inlet_node=17, inlet_area=500, qsinlet=2.5)

    fa = FlowAccumulator(mg,
                         surface='topographic__elevation',
                         flow_director='FlowDirectorD8',
                         runoff_rate=None,
                         depression_finder=None)
    inareats = None
    sedints = None
    for i in range(1000):
        if i == 200:
            # when i equals 200, increase drainage area and sediment flux at inlet
            inareats = 1000
            sedints = 5
        fa.run_one_step()  # flow accumulator
        (mg, dzlat) = latero.run_one_step(
            mg, dt, inlet_area_ts=inareats, qsinlet_ts=sedints)
        mg.at_node['topographic__elevation'][mg.core_nodes] += U * \
            dt  # uplift the landscape

    da = mg.at_node['surface_water__discharge'] / mg.dx**2
    num_sedflux = mg.at_node["qs"]
    analytical_sedflux = U * da

    # test to make sure drainage area at node changed to 1000m^2 + existing drainage
    # area at inlet node (node 17)
    testing.assert_array_almost_equal(
        da[17],
        mg.at_node['drainage_area'][17] + 1000.,
        decimal=4,
        err_msg="LatEro time varying inlet drainage area test failed",
        verbose=True,
    )

    # test for match with analytical sediment flux. note that the values are off a little
    # because of the lateral erosion
    testing.assert_array_almost_equal(
        num_sedflux,
        analytical_sedflux,
        decimal=3,
        err_msg="LatEro time varying inlet transport-limited sediment flux test failed",
        verbose=True,
    )