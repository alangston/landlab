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
        Kl=None,
        Dchar=None,
        solver="basic",
        flow_accumulator=None,
        g=9.81,
        sed_density=2700,
        fluid_density=1000,
        shields_thresh=0.05,
        sec_per_year=31557600.0
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

        if Kl is None:
            raise ValueError(
                "Kl must be set as a float, node array, or field name. It was None."
            )
            
        if Dchar is None:
            raise ValueError(
                "Dchar (charactertistic block size) must be set as a float, node array, or field name. It was None."
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

        if "sediment__flux_from_lat" in grid.at_node:
            self._qs_in = grid.at_node["sediment__flux_from_lat"]
        else:
            self._qs_in = grid.add_zeros("sediment__flux_from_lat", at="node")

        if "lateral_erosion__depth_increment" in grid.at_node:
            self._dzlat = grid.at_node["lateral_erosion__depth_increment"]
        else:
            self._dzlat = grid.add_zeros("lateral_erosion__depth_increment", at="node")


        if "block_size" in grid.at_node:
            self._block_size = grid.at_node["block_size"]
        else:
            self._block_size = grid.add_zeros("block_size", at="node")

        # option use adaptive time stepping. Default is fixed dt supplied by user
        if solver == "basic":
            self.run_one_step = self.run_one_step_basic
        elif solver == "adaptive":
            self.run_one_step = self.run_one_step_adaptive
        self._Kl = Kl  # can be overwritten with spatially variable
        self.g = g
        self.sed_density = sed_density
        self.fluid_density = fluid_density
        self.shields_thresh = shields_thresh
        self.sec_per_year = sec_per_year
        self.Dchar = Dchar
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
        Kl = self._Kl
        qs_in = self._qs_in
        vol_lat = self._grid.at_node["volume__lateral_erosion"]
        depth_at_node = self._grid.at_node["channel__depth"]

        channel__bed_shear_stress = self._grid.at_node["channel__bed_shear_stress"]
        block_size = self._grid.at_node["block_size"]
        Dchar = self.Dchar
        rel_sed_flux = self._grid.at_node["channel_sediment__relative_flux"]
        chan_trans_cap = self._grid.at_node["channel_sediment__volumetric_transport_capacity"]
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("node", "sediment__flux_from_lat", noclobber=False)
#        qs = grid.add_zeros("node", "qs", noclobber=False)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        status_lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        dzlat_ts = np.zeros(grid.number_of_nodes, dtype=float)
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
#        print("dzlat beginning", self._dzlat)
        #ALL***: below is only for finding the lateral node
        for i in dwnst_nodes:
            # potential lateral erosion initially set to 0
            petlat = 0.0
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, da)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                if lat_node > 0 and z[lat_node] > z[i]:
                    # ^ if the elevation of the lateral node is higher than primary node, keep going
                    ### v ARE YOU BLOCKS OR BEDROCK?
                    debug3=0
                    if debug3:
                        print(" ")
                        print("lat_node", lat_node)
                    debug=0
                    if block_size[lat_node] > 0.0:
                        tau_crit = block_size[lat_node]*self.g * (self.sed_density - self.fluid_density) * self.shields_thresh
                        #calc, can blocks be transported?
                        if channel__bed_shear_stress[i] > tau_crit:
                            if debug3:
                                print("blocks can transport")
                                print("tau", channel__bed_shear_stress[i])
                                print("taucrit", tau_crit)
    #                            print(frog)
                            #if blocks transported, do it.
                            # volume of pile of stuff
                            pile_volume = (z[lat_node] - z[i]) * grid.dx ** 2
                            # below is the conversion of trans capacity into m^3/model time step
                            transcap_here_ts = chan_trans_cap[i]*dt*self.sec_per_year
                            avail_trans_cap = transcap_here_ts * (1.0-rel_sed_flux[i])
                            if avail_trans_cap >= pile_volume:
                                #if all sediment from lateral erosion can be transported
                                # by teh channel, send it all down stream
                                qs_in[flowdirs[i]] += pile_volume 
                                # then calculate how much elevation will be lost on teh lateral node
                                # from that downstream transport
                                dzlat_ts[lat_node] = z[i] - z[lat_node]
                                #finally, reset block size to reflect fresh bedrock
                                block_size[lat_node] = 0.0
                                status_lat_nodes[lat_node] = 4
                                if debug3:
                                    print("entire pile transported")
                            elif avail_trans_cap < pile_volume and rel_sed_flux[i] < 1:
                                #**Note here I found that if avail trans capacity is 0,
                                # model will still go through this loop. This is not a problem
                                # except it's inefficient. I fixed it by addign the and
                                #statment above.
                                #use all available trans capacity to move as much
                                # pile as possible
                                # note I use negative availtranscap to make dzlat a negative number
                                dzlat_ts[lat_node] = max(-avail_trans_cap / grid.dx **2, z[i] - z[lat_node])
                                # ^ this will give the elevation that can be removed from 
                                # the pile of stuff that is the lateral node.
                                qs_in[flowdirs[i]] += avail_trans_cap
                                status_lat_nodes[lat_node] = 3
                                debug11=0
                                if debug11:
                                    print("qs_in[flowdirs[i]]", qs_in[flowdirs[i]])
                                    print("entire pile NOT transported")
                                    print("")
                                    print("transcap", transcap_here_ts)
                                    print("relsedflux", rel_sed_flux[i])
                                    print("avail_trans_cap", avail_trans_cap)
                                    print("pile_vol", pile_volume)
                                    print("dzlat[latnode]", dzlat_ts[lat_node])
                                    print("z[latnode]",z[lat_node])
                                    print("z[i]",z[i])
#                                    print(frog)
                                # ^ send the sediment downstream
                            debug11 = 0
                            if debug11:
                                print("qs_in[flowdirs[i]]", qs_in[flowdirs[i]])
                                print("transcap", transcap_here_ts)
                                print("relsedflux", rel_sed_flux[i])
                                print("avail_trans_cap", avail_trans_cap)
                                print("pile_vol", pile_volume)
                                print("dzlat[latnode]", dzlat_ts[lat_node])
                                print("z[latnode]",z[lat_node])
                                print("z[i]",z[i])
#                                print(frog)
#                            print("lat ero occurred blocks can move")
#                            print("lat node", lat_node)
#                            print("transcap", transcap_here_ts)
#                            print("relsedflux", rel_sed_flux[i])
#                            print("avail_trans_cap", avail_trans_cap)
#                            print("pile_vol", pile_volume)
#                            print("dzlat[latnode]", dzlat_ts[lat_node])
#                            print("z[latnode]",z[lat_node])
#                            print("z[i]",z[i])
#                            print("depthatnode[i]",depth_at_node[i])
#                            print("dzlatts", dzlat_ts)
                        #if blocks can't be transported: calc Elat, track undercutting
                        else:    # below is for blocks that can't be transported
                            petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                            vol_lat_dt[lat_node] += abs(petlat) * grid.dx * depth_at_node[i]
                            vol_lat[lat_node] += vol_lat_dt[lat_node] * dt
                            # vol_diff is the volume that must be eroded from lat_node so that its
                            # elevation is the same as node downstream of primary node
    #                        voldiff = (z[i] + depth_at_node[i] - z[flowdirs[i]]) * grid.dx ** 2
                            voldiff = depth_at_node[i] * grid.dx ** 2
                            # below, send sediment downstream
                            qs_in[flowdirs[i]] += (abs(petlat) * grid.dx * depth_at_node[i]) * dt
                            status_lat_nodes[lat_node] = 2
                            if debug:
                                print("blocks can't transport")
#                                print("voldiff", voldiff)
#                                print("vol_lat[latnode]", vol_lat[lat_node])
                            #*******WILL VALLEY WALL COLLAPSE again?
                            if vol_lat[lat_node] >= voldiff:
                                dzlat_ts[lat_node] = depth_at_node[i] * -1.0
                                # ^ Change elevation of lateral node by the length undercut
                                vol_lat[lat_node] = 0.0
#                                print("lat ero occurred blocks can't move")
#                                print("lat node", lat_node)
#                                print("z[flowdirs[i]]",z[flowdirs[i]])
#                                print("z[i]",z[i])
#                                print("depthatnode[i]",depth_at_node[i])
#                                print("dzlatts", dzlat_ts)
                                # ^after the lateral node is eroded, reset its volume eroded to
                                # zero
                                ####HAVE ALL BLOCKS BEEN ERODED? compare lat and primary node within 5mm
                                if np.isclose(z[lat_node], z[i], atol=0.005):
                                    block_size[lat_node] = 0.0

                    else:    # below is for fresh bedrock valley walls
                        petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * depth_at_node[i]
                        vol_lat[lat_node] += vol_lat_dt[lat_node] * dt
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as primary node
                        voldiff = (depth_at_node[i]) * grid.dx ** 2
                        status_lat_nodes[lat_node] = 1
                        if debug:
                            print("fresh bedrock")
#                            print("voldiff", voldiff)
#                            print("vol_lat[latnode]", vol_lat[lat_node])
                        #*******WILL VALLEY WALL COLLAPSE for the first time?
                        if vol_lat[lat_node] >= voldiff:
                            #ALL***: ^now this line is just telling me: will this
                            # valley wall collapse?
                            dzlat_ts[lat_node] = depth_at_node[i] * -1.0
                            # ^ Change elevation of lateral node by the length undercut
                            vol_lat[lat_node] = 0.0
                            # ^after the lateral node is eroded, reset its volume eroded to
                            # zero
                            ####change block size status from bedrock to blocks
                            block_size[lat_node] = Dchar
                            debug1=0
                            if debug1:
#                                print("qs_in[flowdirs[i]]", qs_in[flowdirs[i]])
                                print("lat ero occurred BR")
                                print("lat node", lat_node)
                                print("z[flowdirs[i]]",z[flowdirs[i]])
                                print("z[i]",z[i])
                                print("depthatnode[i]",depth_at_node[i])
#                                print("block_size", block_size[lat_node])
#                                print("petlat", petlat)
                                print("dzlatts", dzlat_ts)
#                                print(frog)
                        # send sediment downstream. for bedrock erosion only
                        qs_in[flowdirs[i]] += (abs(petlat) * grid.dx * depth_at_node[i]) * dt
#                        print("qs_in[flowdirs[i]] AFTER", qs_in[flowdirs[i]])
#        qs[:] = qs_in
        debug2=0
        if debug2:
            print(" ")
            print("qs_in", qs_in)
#            print("status latnodes", status_lat_nodes)
            print("dzlat_ts", dzlat_ts)
#            print(frog)
        
        #All***: ^ I don't exactly remember what that is for/why.
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
        z[:] += dzlat_ts
#        print("z in lat", z)
        return grid

