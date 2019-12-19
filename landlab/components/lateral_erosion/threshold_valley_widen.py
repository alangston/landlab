# -*- coding: utf-8 -*-
"""
Grid-based simulation of lateral erosion by channels in a drainage network.
ALangston


"""

import numpy as np

from landlab import Component, RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from landlab.components.lateral_erosion.node_finder import node_finder

# Hard coded constants
cfl_cond = 0.3  # CFL timestep condition
wid_coeff = 0.4  # coefficient for calculating channel width
wid_exp = 0.35  # exponent for calculating channel width


class ValleyWiden(Component):
    """
    Laterally erode neighbor node through fluvial erosion.

    """


    def __init__(
        self,
        grid,
        latero_mech="UC",
        alph=0.8,
        Kl=None,
        solver="basic",
        flow_accumulator=None,
    ):
        super(ValleyWiden, self).__init__(grid)

        assert isinstance(
            grid, RasterModelGrid
        ), "LateralEroder requires a sqare raster grid."

        if "flow__receiver_node" in grid.at_node:
            if grid.at_node["flow__receiver_node"].size != grid.size("node"):
                msg = (
                    "A route-to-multiple flow director has been "
                    "run on this grid. The LateralEroder is not currently "
                    "compatible with route-to-multiple methods. Use a route-to-"
                    "one flow director."
                )
                raise NotImplementedError(msg)

        if solver not in ("basic", "adaptive"):
            raise ValueError(
                "value for solver not understood ({val} not one of {valid})".format(
                    val=solver, valid=", ".join(("basic", "adaptive"))
                )
            )

        if latero_mech not in ("UC", "TB"):
            raise ValueError(
                "value for latero_mech not understood ({val} not one of {valid})".format(
                    val=latero_mech, valid=", ".join(("UC", "TB"))
                )
            )


        if Kl is None:
            raise ValueError(
                "Kl must be set as a float, node array, or field name. It was None."
            )

        if solver == "adaptive":
            if not isinstance(flow_accumulator, FlowAccumulator):
                raise ValueError(
                    (
                        "When the adaptive solver is used, a valid "
                        "FlowAccumulator must be passed on "
                        "instantiation."
                    )
                )
            self._flow_accumulator = flow_accumulator

        # Create fields needed for this component if not already existing
        if "volume__lateral_erosion" in grid.at_node:
            self._vol_lat = grid.at_node["volume__lateral_erosion"]
        else:
            self._vol_lat = grid.add_zeros("volume__lateral_erosion", at="node")

        if "sediment__flux" in grid.at_node:
            self._qs_in = grid.at_node["sediment__flux"]
        else:
            self._qs_in = grid.add_zeros("sediment__flux", at="node")

        if "lateral_erosion__depth_increment" in grid.at_node:
            self._dzlat = grid.at_node["lateral_erosion__depth_increment"]
        else:
            self._dzlat = grid.add_zeros("lateral_erosion__depth_increment", at="node")

        # you can specify the type of lateral erosion model you want to use.
        # But if you don't the default is the undercutting-slump model
        if latero_mech == "TB":
            self._TB = True
            self._UC = False
        else:
            self._UC = True
            self._TB = False
        # option use adaptive time stepping. Default is fixed dt supplied by user
        if solver == "basic":
            self.run_one_step = self.run_one_step_basic
        elif solver == "adaptive":
            self.run_one_step = self.run_one_step_adaptive
        self._Kl = Kl  # can be overwritten with spatially variable

        self._dzdt = grid.add_zeros(
            "dzdt", at="node", noclobber=False
        )  # elevation change rate (M/Y)

        # handling Kv for floats (inwhich case it populates an array N_nodes long) or
        # for arrays of Kv. Checks that length of Kv array is good.
        self._Kl = np.ones(self._grid.number_of_nodes, dtype=float) * Kl

    def run_one_step_basic(self, dt=1.0):
        """Calculate vertical and lateral erosion for
        a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]

        """
        grid = self._grid
        UC = self._UC
        TB = self._TB
        Kl = self._Kl
        qs_in = self._qs_in
        dzdt = self._dzdt
        vol_lat = self._grid.at_node["volume__lateral_erosion"]
        depth_at_node = self._grid.at_node["channel__depth"]
#        print("depth in valleywid", depth_at_node)

        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("node", "sediment__flux", noclobber=False)
        qs = grid.add_zeros("node", "qs", noclobber=False)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        da = grid.at_node["drainage_area"]
        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        interior_s = s[np.where((grid.status_at_node[s] == 0))[0]]
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        max_slopes[:] = max_slopes.clip(0)
        #ALL***: below is only for finding the lateral node
        for i in dwnst_nodes:
            # potential lateral erosion initially set to 0
            petlat = 0.0
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                # node_finder picks the lateral node to erode based on angle
                # between segments between three nodes
                [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, da)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                # if the lateral node is not 0 or -1 continue. lateral node may be
                # 0 or -1 if a boundary node was chosen as a lateral node. then
                # radius of curavature is also 0 so there is no lateral erosion
                if lat_node > 0:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative
                    if z[lat_node] > z[i]:
                        petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                        # the calculated potential lateral erosion is mutiplied by the length of the node
                        # and the bank height, then added to an array, vol_lat_dt, for volume eroded
                        # laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                        # each timestep loop. vol_lat_dt is added to itself in case more than one primary
                        # nodes are laterally eroding this lat_node
                        # volume of lateral erosion per timestep
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * depth_at_node[i]
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height at
                        # node i), slumping, and instant removal.
                        if UC == 1:
                            voldiff = (z[i] + depth_at_node[i] - z[flowdirs[i]]) * grid.dx ** 2
                        # TB model: entire lat node must be eroded before lateral erosion
                        # occurs
                        if TB == 1:
                            voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx ** 2
                        # if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already has
                        # timestep in it
                        if vol_lat[lat_node] >= voldiff:
                            self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                            # after the lateral node is eroded, reset its volume eroded to
                            # zero
                            vol_lat[lat_node] = 0.0
            # send sediment downstream. sediment eroded from 
            # and lateral erosion is sent downstream
            #            print("debug before 406")
            qs_in[flowdirs[i]] += (
                qs_in[i] - (petlat * grid.dx * depth_at_node[i])
            )  # qsin to next node
        
        qs[:] = qs_in
        #All***: ^ I don't exactly remember what that is for/why.
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
#        for i in dwnst_nodes:
#            lat_node = lat_nodes[i]
#            depth_at_node[i]
#            if lat_node > 0:  # greater than zero now bc inactive neighbors are value -1
#                if z[lat_node] > z[i]:
#                    # vol_diff is the volume that must be eroded from lat_node so that its
#                    # elevation is the same as node downstream of primary node
#                    # UC model: this would represent undercutting (the water height at
#                    # node i), slumping, and instant removal.
#                    if UC == 1:
#                        voldiff = (z[i] + depth_at_node[i] - z[flowdirs[i]]) * grid.dx ** 2
#                    # TB model: entire lat node must be eroded before lateral erosion
#                    # occurs
#                    if TB == 1:
#                        voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx ** 2
#                    # if the total volume eroded from lat_node is greater than the volume
#                    # needed to be removed to make node equal elevation,
#                    # then instantaneously remove this height from lat node. already has
#                    # timestep in it
#                    if vol_lat[lat_node] >= voldiff:
#                        self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
#                        # after the lateral node is eroded, reset its volume eroded to
#                        # zero
#                        vol_lat[lat_node] = 0.0

        # change height of landscape by just removing laterally eroded stuff.
        z[:] += self._dzlat
        return grid, self._dzlat

