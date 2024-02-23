# -*- coding: utf-8 -*-
"""
Grid-based simulation of lateral erosion by channels in a drainage network.
ALangston


"""

import numpy as np

from landlab import Component, RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from landlab.components.lateral_erosion.node_finder import node_finder
from landlab.utils.return_array import return_array_at_node


# Hard coded constants
cfl_cond = 0.3  # CFL timestep condition
wid_coeff = 0.4  # coefficient for calculating channel width
wid_exp = 0.35  # exponent for calculating channel width


class ValleyWiden(Component):
    """
    Laterally erode neighbor node through fluvial erosion AND add volume of
    collapsed bedrock to channel as sediment of a specified size (Dchar).


    Parameters
    ----------
    grid : ModelGrid
        A Landlab square cell raster grid object
    Kl : float
        Lateral bedrock erodibility, units  1/years
    Dchar : float
        Characteristic sediment grain size bedrock breaks up into, units m
    solver : string
        Solver options:
            (1) 'basic' (default): explicit forward-time extrapolation.
                Simple but will become unstable if time step is too large or
                if bedrock erodibility is vry high.
            (2) 'adaptive': subdivides global time step as needed to
                prevent slopes from reversing.
    flow_accumulator : Instantiated Landlab FlowAccumulator, optional
        When solver is set to "adaptive", then a valid Landlab FlowAccumulator
        must be passed. It will be run within sub-timesteps in order to update
        the flow directions and drainage area.
        
    Special note on valley widen parameters below: {b_sde, Qs_thresh_prefactor, Qs_power_onA,
    Qs_power_onAthresh, Qs_prefactor}. These come from the SedFluxDependent Eroder.
    If any of these parameters in SedFluxDep are changed from default: {b_sp, c_sp,
    k_w, fluid_density, sediment_density, k_Q, mannings_n}, then you cannot use
    the default values of valley widen parameters from SedDepEroder. You must pass
    values of {b_sde, Qs_thresh_prefactor, Qs_power_onA, Qs_power_onAthresh,
    Qs_prefactor} calculated in SedDepEroder to valley widen component. 
        
    b_sde = 0.5,    #b_sp from sedflux dep eroder
    Qs_thresh_prefactor = 3.668127525963118e-8,    #from sedflux dep eroder
    Qs_power_onAthresh = 0.33333333333333,    #from sedflux dep eroder
    Qs_prefactor = 3.5972042802486196e-7,    #from sedflux dep eroder
    Qs_power_onA = 0.6333333333333333,    #from sedflux dep eroder
    b_sde : float
        Power on drainage area to give discharge. This is from SedDepEroder
    I have a python script with a demo of what I described above named "valley_widen_tests_tcap.py"
    I will turn this into a notebook someday. 

    """


    def __init__(
        self,
        grid,
        Kl=None,
        Dchar=None,
        discharge_field="drainage_area",
        flow_accumulator=None,
        g=9.81,
        sed_density=2700,
        fluid_density=1000,
        shields_thresh=0.05,
        sec_per_year=31557600.0,
        b_sde = 0.5,    #b_sp from sedflux dep eroder
        Qs_thresh_prefactor = 3.668127525963118e-8,    #from sedflux dep eroder
        Qs_power_onAthresh = 0.33333333333333,    #from sedflux dep eroder
        Qs_prefactor = 3.5972042802486196e-7,    #from sedflux dep eroder
        Qs_power_onA = 0.6333333333333333,    #from sedflux dep eroder
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

        if Kl is None:
            raise ValueError(
                "Kl must be set as a float, node array, or field name. It was None."
            )

        # Create fields needed for this component if not already existing
        if "volume__lateral_erosion" in grid.at_node:
            self._vol_lat = grid.at_node["volume__lateral_erosion"]
        else:
            self._vol_lat = grid.add_zeros("volume__lateral_erosion", at="node")

        if "inlet_sediment__flux" in grid.at_node:
            self._qs_in_inlet = grid.at_node["inlet_sediment__flux"]
        # if "lateral_sediment__flux" in grid.at_node:
        #     self._qs_in = grid.at_node["lateral_sediment__flux"]
        # else:
        #     self._qs_in = grid.add_zeros("lateral_sediment__flux", at="node")

        if "lateral_erosion__depth_increment" in grid.at_node:
            self._dzlat = grid.at_node["lateral_erosion__depth_cum"]
        else:
            self._dzlat = grid.add_zeros("lateral_erosion__depth_cum", at="node")
#******************AL:, AL, you need to look at the above^ and see if its correct
#                   see also fixes in lateral erosion in teh matster landlab version
        if "status_lat_nodes" in grid.at_node:
            self._status_lat_nodes = grid.at_node["status_lat_nodes"]
        else:
            self._status_lat_nodes = grid.add_zeros("status_lat_nodes", at="node")
        if "block_size" in grid.at_node:
            self._block_size = grid.at_node["block_size"]
        else:
            self._block_size = grid.add_zeros("block_size", at="node")
        save_dzlat_ts = 1
        if save_dzlat_ts:
            if "dzlat_ts" in grid.at_node:
                self._dzlat_ts = grid.at_node["dzlat_ts"]
            else:
                self._dzlat_ts = grid.add_zeros("dzlat_ts", at="node")
        self._Kl = Kl  # can be overwritten with spatially variable

        # handling Kv for floats (inwhich case it populates an array N_nodes long) or
        # for arrays of Kv. Checks that length of Kv array is good.
        self._Kl = np.ones(self._grid.number_of_nodes, dtype=float) * Kl
        self._A = return_array_at_node(grid, discharge_field)
        ###^^^***** april 25, 2022 above from stream_power AND my new additions
        # to sed_flux_dependent_incision

    def run_one_step(self, dt=1.0):
        """Calculate lateral erosion for
        a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]
        qs_in/grid.at_node["lateral_sediment__flux"] : array
            qs_in will be in UNITS of m**3/time step. THis is the same
            units as needed for vertical incsion

        """
        grid = self._grid
        Kl = self._Kl

#        qs_in = self._qs_in
        #july 28, 2020: removign the above because I want qs_in reset every timestep
        vol_lat = self._grid.at_node["volume__lateral_erosion"]

        z = grid.at_node["topographic__elevation"]

        debug7 = 0
        if debug7:
            print("qs_in", qs_in)
        # clear qsin for next loop
        if "inlet_sediment__flux" in grid.at_node:
            # qs_in = np.copy(self._grid.at_node["inlet_sediment__flux"])
            qs_in = np.copy(self._qs_in_inlet)
        else:
            qs_in = grid.add_zeros("node", "lateral_sediment__flux", clobber=True)
        #^ version of line from 1/2020. 7/28/2020: changed noclobber=False to clobber=True

        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        status_lat_nodes = grid.add_zeros("status_lat_nodes", at="node", clobber=True)#, noclobber=False)
        dzlat_ts = np.zeros(grid.number_of_nodes, dtype=float)
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        node_A = self._A
        #4/25/2022 AL added thsi above. 
        # I believe this works now along with added stuff on line 358
        # node_A = grid.at_node["surface_water__discharge"]
        # ^ AL Dec 31: change using drainage area to using surface water discharge
        # that way if there is runoff added to flow accumulator, then lateral eroder will
        # pick it up. If no runoff is added, then surface water discharge is the same as 
        # drainage area. This is how I'm dealing with keeping K_sp for vertical adn
        # K_lat eroding at the same OOM with changign runoff rate in teh vertical 
        # component
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

        """
        #ALL***: below is only for finding the lateral node
        """
        for i in dwnst_nodes:
            # potential lateral erosion initially set to 0
            petlat = 0.0
            # Choose lateral node for node i. If node i flows downstream, continue.
            # if node i is the first cell at the top of the drainage network, don't go
            # into this loop because in this case, node i won't have a "donor" node
            if i in flowdirs:
                [lat_node, inv_rad_curv] = node_finder(grid, i, flowdirs, node_A)
                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[i] = lat_node
                if lat_node > 0 and z[lat_node] > z[i]:
                    # ^ if the elevation of the lateral node is higher than primary node, keep going
                    ### below v ARE YOU BLOCKS OR BEDROCK?
                    debug3=0
                    debug=0
                    if block_size[lat_node] > 0.0:
                        #^ if block size >0.0, you have collapsed, not bedrock
                        """
                        new version 23February2024: My vision is that this will be a lateral eroder that is more
                        simple than the amazing grainsize lateral eroder I have already. this version will 
                        1. undercut bedrock wall until collapse.
                        2. AFter collapse, teh erodivility of the wall will change to represent blocks instead of solid bedrock.
                        this can be done by changing value of K_lat.
                        3. Should then change elevation at lateral node every time step. 
                        4. When all blocks eroded away, switch back to bedrock. 
                       
                        """

                        #if lat node is already blocks: calc ******Elat WITH NEW K_lat******, track undercutting to remove this volume at the end of the loop.
                        petlat = -Kl[i] * node_A[i] * max_slopes[i] * inv_rad_curv
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * depth_at_node[i]
                        vol_lat[lat_node] += vol_lat_dt[lat_node] * dt
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
#                        voldiff = (z[i] + depth_at_node[i] - z[flowdirs[i]]) * grid.dx ** 2
                        voldiff = depth_at_node[i] * grid.dx ** 2
                        # below, send sediment downstream, units of volume
                        """
                        May11, remove qs_in
                        """
                        qs_in[flowdirs[i]] += (abs(petlat) * grid.dx * depth_at_node[i]) * dt
                        if np.any(np.isnan(qs_in))==True:
                            print("we got a nan in qs_in, line 347")
                            print("time = ", precip.elapsed_time)
                            toc=time.time()
                            print("elapsed time = ", toc-tic)
                            print(frog)
                        status_lat_nodes[lat_node] = 3
                        if debug3 and lat_node == 438:
                            print("blocks can't transport")
                        #*******WILL VALLEY WALL COLLAPSE again?
                        if vol_lat[lat_node] >= voldiff:
                            dzlat_ts[lat_node] = depth_at_node[i] * -1.0
                            # ^ Change elevation of lateral node by the length undercut
                            vol_lat[lat_node] = 0.0
                            # ^after the lateral node is eroded, reset its volume eroded to
                            # zero
                            ####HAVE ALL BLOCKS BEEN ERODED? compare lat and primary node within 5mm
                            if np.isclose(z[lat_node], z[i], atol=0.005):
                                block_size[lat_node] = 0.0

                    else:    # below is for fresh bedrock valley walls
                        petlat = -Kl[i] * node_A[i] * max_slopes[i] * inv_rad_curv
                        vol_lat_dt[lat_node] += abs(petlat) * grid.dx * depth_at_node[i]
                        vol_lat[lat_node] += vol_lat_dt[lat_node] * dt

                        
                        """
                        # trying somethign new for voldiff
                        vol diff is now going to be a percentage of the height of the
                        lateral node. This si arbitrary. 
                        So I'll go with when 20% of the lateral node volume has been eroded
                        Then it can collapse for the first time. 
                        Note, my explanation below is from the old code.
                        """
                        # vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as primary node
                        voldiff = (depth_at_node[i]) * grid.dx ** 2
                        # voldiff = (z[lat_node] - z[i]) * grid.dx **2 * 0.1

                        status_lat_nodes[lat_node] = 1
                        #^node status=1 means that now this br valley wall has experienced some erosion
                        if debug:
                            print("petlat_lateral", petlat*dt)
                        #*******WILL VALLEY WALL COLLAPSE for the first time?
                        if vol_lat[lat_node] >= voldiff:
                            #ALL***: ^now this line is just telling me: will this
                            # valley wall collapse?
                            #*****************************
                            #SOMETHING FUNKY GOING ON BELOW! CHECK!
                            dzlat_ts[lat_node] = depth_at_node[i] * -1.0
                            # ^ Change elevation of lateral node by the length undercut
                            vol_lat[lat_node] = 0.0
                            # ^after the lateral node is eroded, reset its volume eroded to
                            # zero
                            ####change block size status from bedrock to blocks
                            block_size[lat_node] = Dchar
                            status_lat_nodes[lat_node] = 2
                        # send sediment downstream. for bedrock erosion only
                        """
                        May11, remove qs_in
                        """
                        qs_in[flowdirs[i]] += (abs(petlat) * grid.dx * depth_at_node[i]) * dt
                        if np.any(np.isnan(qs_in))==True:
                                print("we got a nan in qs_in, line 397")
                                print(" ")
                                print("downstream node", flowdirs[i])
                                print("qs_in[flowdirs[i]]", qs_in[flowdirs[i]])
                                print("petlat", petlat)
                                print("depth_at_nodes", depth_at_node[i])
                                print("transcap", transcap_here_ts)
                                print("relsedflux", rel_sed_flux[i])
                                print("avail_trans_cap", avail_trans_cap)
                                print("pile_vol", pile_volume)
                                print("dzlat[latnode]", dzlat_ts[lat_node])
                                print("z[latnode]",z[lat_node])
                                print("z[i]",z[i])
                                print(" ")
                                print("pile volume", pile_volume)
                                print("trans cap here", chan_trans_cap[i])
                                print("transcaphere_ts", transcap_here_ts)
                                print("avail trans cap", avail_trans_cap)
                                print("dt", dt)
                                print("time = ", precip.elapsed_time)
                                toc=time.time()
                                print("elapsed time = ", toc-tic)

                                print(frog)
#                        print("qs_in[flowdirs[i]] AFTER", qs_in[flowdirs[i]])
#        qs[:] = qs_in
        debug2=0
        if debug2:
            print(" ")
#            print("qs_in", qs_in)
#            print("status latnodes", status_lat_nodes)
            print("dzlat_ts", dzlat_ts)
#            print(frog)
        
        grid.at_node["lateral_erosion__depth_cum"][:] += dzlat_ts
        #^ AL: this only keeps track of cumulative lateral erosion at each cell.

        if "dzlat_ts" in grid.at_node:
            grid.at_node["dzlat_ts"][:] = dzlat_ts
        #**AL: 11/18/21: added the above few lines to save lateral erosion per timestep
        # change height of landscape by just removing laterally eroded stuff.
        z[:] += dzlat_ts
#        print("z in lat", z)
        return grid
