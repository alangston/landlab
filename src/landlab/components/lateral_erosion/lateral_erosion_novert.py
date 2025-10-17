# -*- coding: utf-8 -*-
"""
Grid-based simulation of lateral erosion by channels in a drainage network.
ALangston


"""

import numpy as np
import matplotlib.pyplot as plt  # For plotting results; optional

from landlab import Component, RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from landlab.components.lateral_erosion.node_finder import node_finder
from landlab.utils.return_array import return_array_at_node

from landlab import imshow_grid  # For plotting results; optional

# Hard coded constants
cfl_cond = 0.3  # CFL timestep condition
wid_coeff = 0.4  # coefficient for calculating channel width
wid_exp = 0.35  # exponent for calculating channel width


class LateralErosionSedDep(Component):
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
        discharge_field="surface_water__discharge",
        solver="basic",
        flow_accumulator=None,
    ):
        super(LateralErosionSedDep, self).__init__(grid)

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

        if "lateral_sediment__influx" in grid.at_node:
             self._lat_sed_influx = grid.at_node["lateral_sediment__influx"]
        else:
             self._lat_sed_influx = grid.add_zeros("lateral_sediment__influx", at="node")

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
        # option use adaptive time stepping. Default is fixed dt supplied by user
        if solver == "basic":
            self.run_one_step = self.run_one_step_basic
        elif solver == "adaptive":
            self.run_one_step = self.run_one_step_adaptive
        self._Kl = Kl  # can be overwritten with spatially variable
        self._Dchar = Dchar


        # handling Kv for floats (inwhich case it populates an array N_nodes long) or
        # for arrays of Kv. Checks that length of Kv array is good.
        self._Kl = np.ones(self._grid.number_of_nodes, dtype=float) * Kl
        self._A = return_array_at_node(grid, discharge_field)
        self._slope = grid.at_node["topographic__steepest_slope"]

        ###^^^***** april 25, 2022 above from stream_power AND my new additions
        # to sed_flux_dependent_incision

    def run_one_step_basic(self, dt=1.0):
        """Calculate lateral erosion for
        a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]
        qs_in/grid.at_node["lateral_sediment__influx"] : array
            qs_in will be in UNITS of m**3/time step. THis is the same
            units as needed for vertical incsion

        """
        grid = self._grid
        Kl = self._Kl
        vol_lat = self._grid.at_node["volume__lateral_erosion"]
        """
        trying this to fix hole digging because of qsin = nan because depth 
        at node = nan because of Dan's sneakiness! in sed flux dependent.
		Octrober 4, 2022: Dan's sneakiness is on line 1155 where he calculates H. If slope is 0, H is 0, and then qsin is zero if H is zero.  <-- but I can't find the line where qs_in or sed flux is calculated wtih H in Dan's code.
		Ah, it doesn't matter. I just change channel depths that are nans into zeros
        """
        # depth_at_node = self._grid.at_node["channel__depth"]
            # water depth in meters, needed for lateral erosion calc
        # depth_at_node = wid_coeff * (self._A) ** wid_exp
        depth_at_node = self.calc_implied_depth(grain_diameter=0.2)

        # depth_nans = np.where(np.isnan(self.grid.at_node["channel__depth"])==True)
        # depth_at_node[depth_nans] = 0.0
        
        block_size = self._grid.at_node["block_size"]
        Dchar = self._Dchar
        """
        6october 2025: below, trying to decide how to handle inlet sediment and lateral erosion. 
        Maybe it doesn't matter if inlet sediment is zero here because there won't be any vertical erosion there.
        So maybe just reset inlet sediment after lateral erosion and let it be reset to zero here.
        """
        # clear qsin for next loop
        lat_sed_influx = self._lat_sed_influx[:] 
        lat_sed_influx[:] = 0.0

        debug7 = 0
        if debug7:
            print("lat_sed_influx", lat_sed_influx)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        # status_lat_nodes = grid.add_zeros("status_lat_nodes", at="node", clobber=True)#, noclobber=False)
        status_lat_nodes = self._status_lat_nodes
        dzlat_ts = np.zeros(grid.number_of_nodes, dtype=float)
        vol_lat_dt = np.zeros(grid.number_of_nodes)
        node_A = self._A
        slope = self._slope
        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        # slope = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]
        #^ALL 7/282020: this is from sed_flux_dep_incision.py
        z = grid.at_node["topographic__elevation"]
        sed_depth = grid.at_node["soil__depth"]
        z_br = grid.at_node["bedrock__elevation"]
        # make a list l, where node status is interior (signified by label 0) in s
        interior_s = s[np.where((grid.status_at_node[s] == 0))[0]]
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        slope[:] = slope.clip(0)
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
                    #%%
                    if block_size[lat_node] > 0.0:
                        pass
                        """
                        17October2025: this is why lateral erosion wasn't working earlier. See above. the lateral erosion
                        bit below would only happen if block_size at the node was set to zero. Below in the code, the 
                        block_size at node is set to 1 the first time the lateral node collapses. Comment that line out.
                        """
                    else:
                        # below is for fresh bedrock valley walls
                        petlat = -Kl[i] * node_A[i] * slope[i] * inv_rad_curv
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
                        # voldiff = (depth_at_node[i]) * grid.dx ** 2
                        voldiff = (z[lat_node] - z[i]) * grid.dx **2 * 0.1

                        status_lat_nodes[lat_node] = 1
                        #^node status=1 means that now this br valley wall has experienced some erosion
                        debug=0
                        if debug and i==130:
                            print("i", i)
                            print("lat_node", lat_node)
                            print("z[i]", z[i])
                            print("z[lat_node]", z[lat_node])
                            print("z[flowdirs[i]]",z[flowdirs[i]])
                            print("petlat_lateral", petlat*dt)
                            print("vol_diff", voldiff)
                            print("vol_lat[lat_node]", vol_lat[lat_node])
                            water_depth = self.calc_implied_depth(grain_diameter=0.2)
                            print("water_depth", water_depth[i])
                            print("depth at node", depth_at_node[i])
                            print("slope at node", slope[i])
                            print(" ")
                            print(frog)
                        #*******WILL VALLEY WALL COLLAPSE for the first time?
                        if vol_lat[lat_node] >= voldiff:
                            #ALL***: ^now this line is just telling me: will this
                            # valley wall collapse?
                            #*****************************

                            # dzlat_ts[lat_node] = depth_at_node[i] * -1.0
                            # dzlat_ts[lat_node] = z[flowdirs[i]] - z[lat_node]
                            dzlat_ts[lat_node] = z[i] - z[lat_node]

                            # ^ Change elevation of lateral node by the height of the undercut
                            vol_lat[lat_node] = 0.0
                            # ^after the lateral node is eroded, reset its volume eroded to
                            # zero
                            ####change block size status from bedrock to blocks
                            #block_size[lat_node] = Dchar
                            status_lat_nodes[lat_node] = 2

                            #print(" ")
                            #print("valley collapse at node: ", i)
                            #print("dzlat_ts", dzlat_ts[lat_node])
                            #print("discharge = ", max(node_A))
                            # water_depth = self.calc_implied_depth(grain_diameter=0.5)
                            # print("water_depth", water_depth[i])
                            # print("depth at node", depth_at_node[i])
                            # print("slope at node", slope[i])

                            # print(frog)
                        # send sediment downstream. for bedrock erosion only
                        """
                        May11, remove lat_sed_influx
                        """
                        lat_sed_influx[flowdirs[i]] += (abs(petlat) * grid.dx * depth_at_node[i]) * dt
                        if np.any(np.isnan(lat_sed_influx))==True:
                                print("we got a nan in lat_sed_influx, line 397")

        grid.at_node["lateral_erosion__depth_cum"][:] += dzlat_ts
        #^ AL: this only keeps track of cumulative lateral erosion at each cell.
        
        #***830Jan2025: below is where I change the grid sediment__influx to include lateral sed flux
        self.grid.at_node["lateral_sediment__influx"][:] = lat_sed_influx
        if "dzlat_ts" in grid.at_node:
            grid.at_node["dzlat_ts"][:] = dzlat_ts
        #**AL: 11/18/21: added the above few lines to save lateral erosion per timestep
        
        # change height of landscape by just removing laterally eroded stuff.
        #z_br[:] += dzlat_ts
        #z[:] = z_br[:] + sed_depth[:]
        #1October2025. removing sediment as well as bedrock.
        z[:] += dzlat_ts

        return grid


    """
    Below, this is from Vanessa's gravel bedrock eroder. Just giving it a try.'
    """
    
    def calc_implied_depth(self, grain_diameter=0.01):
        """Utility function that calculates and returns water depth implied by
        slope and grain diameter, using Wickert & Schildgen (2019) equation 8.
    
        The equation is::
    
            h = ((rho_s - rho / rho)) * (1 + epsilon) * tau_c * (D / S)
    
        where the factors on the right are sediment and water density, excess
        shear-stress factor, critical Shields stress, grain diameter, and slope
        gradient. Here the prefactor on ``D/S`` assumes sediment density of 2650 kg/m3,
        water density of 1000 kg/m3, shear-stress factor of 0.2, and critical
        Shields stress of 0.0495, giving a value of 0.09801.
    
        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> grid = RasterModelGrid((3, 3), xy_spacing=1000.0)
        >>> elev = grid.add_zeros("topographic__elevation", at="node")
        >>> elev[3:] = 10.0
        >>> sed = grid.add_zeros("soil__depth", at="node")
        >>> sed[3:] = 100.0
        >>> fa = FlowAccumulator(grid)
        >>> fa.run_one_step()
        >>> eroder = GravelBedrockEroder(grid)
        >>> water_depth = eroder.calc_implied_depth(grain_diameter=0.01)
        >>> int(water_depth[4] * 1000)
        98
        """
        depth_factor = 0.09801
        depth = np.zeros(self._grid.number_of_nodes)
        nonzero_slope = self._slope > 0.0
        depth[nonzero_slope] = (
            depth_factor * grain_diameter / self._slope[nonzero_slope]
        )
        return depth
