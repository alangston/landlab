"""Grid-based simulation of lateral erosion by channels in a drainage network.

ALangston
"""

import numpy as np

from landlab import Component
from landlab import RasterModelGrid
from landlab.components.flow_accum import FlowAccumulator

from .node_finder import node_finder

# Hard coded constants
cfl_cond = 0.3  # CFL timestep condition
wid_coeff = 0.4  # coefficient for calculating channel width
wid_exp = 0.35  # exponent for calculating channel width


class LateralEroder(Component):
    """Laterally erode neighbor node through fluvial erosion.

    Landlab component that finds a neighbor node to laterally erode and
    calculates lateral erosion.
    See the publication:

    Langston, A.L., Tucker, G.T.: Developing and exploring a theory for the
    lateral erosion of bedrock channels for use in landscape evolution models.
    Earth Surface Dynamics, 6, 1-27,
    `https://doi.org/10.5194/esurf-6-1-2018 <https://www.earth-surf-dynam.net/6/1/2018/>`_

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowAccumulator, LateralEroder
    >>> np.random.seed(2010)

    Define grid and initial topography

    * 5x4 grid with baselevel in the lower left corner
    * All other boundary nodes closed
    * Initial topography is plane tilted up to the upper right with noise

    >>> mg = RasterModelGrid((5, 4), xy_spacing=10.0)
    >>> mg.set_status_at_node_on_edges(
    ...     right=mg.BC_NODE_IS_CLOSED,
    ...     top=mg.BC_NODE_IS_CLOSED,
    ...     left=mg.BC_NODE_IS_CLOSED,
    ...     bottom=mg.BC_NODE_IS_CLOSED,
    ... )
    >>> mg.status_at_node[1] = mg.BC_NODE_IS_FIXED_VALUE
    >>> rand_noise = np.array(
    ...     [
    ...         [0.00436992, 0.03225985, 0.03107455, 0.00461312],
    ...         [0.03771756, 0.02491226, 0.09613959, 0.07792969],
    ...         [0.08707156, 0.03080568, 0.01242658, 0.08827382],
    ...         [0.04475065, 0.07391732, 0.08221057, 0.02909259],
    ...         [0.03499337, 0.09423741, 0.01883171, 0.09967794],
    ...     ]
    ... ).flatten()
    >>> mg.at_node["topographic__elevation"] = (
    ...     mg.node_y / 10.0 + mg.node_x / 10.0 + rand_noise
    ... )
    >>> U = 0.001
    >>> dt = 100

    Instantiate flow accumulation and lateral eroder and run each for one step

    >>> fa = FlowAccumulator(
    ...     mg,
    ...     surface="topographic__elevation",
    ...     flow_director="FlowDirectorD8",
    ...     runoff_rate=None,
    ...     depression_finder=None,
    ... )
    >>> latero = LateralEroder(mg, latero_mech="UC", Kv=0.001, Kl_ratio=1.5)

    Run one step of flow accumulation and lateral erosion to get the dzlat array
    needed for the next part of the test.

    >>> fa.run_one_step()
    >>> mg, dzlat = latero.run_one_step(dt)

    Evolve the landscape until the first occurence of lateral erosion. Save arrays
    volume of lateral erosion and topographic elevation before and after the first
    occurence of lateral erosion

    >>> while min(dzlat) == 0.0:
    ...     oldlatvol = mg.at_node["volume__lateral_erosion"].copy()
    ...     oldelev = mg.at_node["topographic__elevation"].copy()
    ...     fa.run_one_step()
    ...     mg, dzlat = latero.run_one_step(dt)
    ...     newlatvol = mg.at_node["volume__lateral_erosion"]
    ...     newelev = mg.at_node["topographic__elevation"]
    ...     mg.at_node["topographic__elevation"][mg.core_nodes] += U * dt
    ...

    Before lateral erosion occurs, *volume__lateral_erosion* has values at
    nodes 6 and 10.

    >>> np.around(oldlatvol, decimals=0)
    array([ 0.,  0., 0., 0.,
            0.,  0., 79., 0.,
            0.,  0., 24., 0.,
            0.,  0., 0., 0.,
            0.,  0., 0., 0.])


    After lateral erosion occurs at node 6, *volume__lateral_erosion* is reset to 0

    >>> np.around(newlatvol, decimals=0)
    array([ 0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,
            0.,  0., 24.,  0.,
            0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.])


    After lateral erosion at node 6, elevation at node 6 is reduced by -1.41
    (the elevation change stored in dzlat[6]). It is also provided as the
    at-node grid field *lateral_erosion__depth_increment*.

    >>> np.around(oldelev, decimals=2)
    array([0.  , 1.03, 2.03, 3.  ,
           1.04, 1.77, 2.45, 4.08,
           2.09, 2.65, 3.18, 5.09,
           3.04, 3.65, 4.07, 6.03,
           4.03, 5.09, 6.02, 7.1 ])

    >>> np.around(newelev, decimals=2)
    array([0.  , 1.03, 2.03, 3.  ,
           1.04, 1.77, 1.03, 4.08,
           2.09, 2.65, 3.18, 5.09,
           3.04, 3.65, 4.07, 6.03,
           4.03, 5.09, 6.02, 7.1 ])

    >>> np.around(dzlat, decimals=2)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  , -1.41,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0. ])

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    Langston, A., Tucker, G. (2018). Developing and exploring a theory for the
    lateral erosion of bedrock channels for use in landscape evolution models.
    Earth Surface Dynamics  6(1), 1--27.
    https://dx.doi.org/10.5194/esurf-6-1-2018

    **Additional References**

    None Listed

    """

    _name = "LateralEroder"

    _unit_agnostic = False

    _cite_as = """
    @article{langston2018developing,
      author = {Langston, A. L. and Tucker, G. E.},
      title = {{Developing and exploring a theory for the lateral erosion of
      bedrock channels for use in landscape evolution models}},
      doi = {10.5194/esurf-6-1-2018},
      pages = {1---27},
      number = {1},
      volume = {6},
      journal = {Earth Surface Dynamics},
      year = {2018}
    }
    """
    _info = {
        "drainage_area": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**2",
            "mapping": "node",
            "doc": "Upstream accumulated surface area contributing to the node's discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "lateral_erosion__depth_increment": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Change in elevation at each node from lateral erosion during time step",
        },
        "sediment__influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/y",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment entering each node)",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "volume__lateral_erosion": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3",
            "mapping": "node",
            "doc": "Array tracking volume eroded at each node from lateral erosion",
        },
    }

    def __init__(
        self,
        grid,
        latero_mech="UC",
        alph=0.8,
        Kv=0.001,
        Kl_ratio=1.0,
        discharge_field="surface_water__discharge"

    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A Landlab square cell raster grid object
        latero_mech : string, optional (defaults to UC)
            Lateral erosion algorithm, choices are "UC" for undercutting-slump
            model and "TB" for total block erosion
        alph : float, optional (defaults to 0.8)
            Parameter describing potential for deposition, dimensionless
        Kv : float, node array, or field name
            Bedrock erodibility in vertical direction, 1/years
        Kl_ratio : float, optional (defaults to 1.0)
            Ratio of lateral to vertical bedrock erodibility, dimensionless
        solver : string
            Solver options:
                (1) 'basic' (default): explicit forward-time extrapolation.
                    Simple but will become unstable if time step is too large or
                    if bedrock erodibility is vry high.
                (2) 'adaptive': subdivides global time step as needed to
                    prevent slopes from reversing.
        
        """
        super().__init__(grid)

        assert isinstance(
            grid, RasterModelGrid
        ), "LateralEroder requires a sqare raster grid."

        if "flow__receiver_node" in grid.at_node and grid.at_node[
            "flow__receiver_node"
        ].size != grid.size("node"):
            raise NotImplementedError(
                "A route-to-multiple flow director has been "
                "run on this grid. The LateralEroder is not currently "
                "compatible with route-to-multiple methods. Use a route-to-"
                "one flow director."
            )

        if latero_mech not in ("UC", "TB"):
            raise ValueError(
                "value for latero_mech not understood ({val} not one of {valid})".format(
                    val=latero_mech, valid=", ".join(("UC", "TB"))
                )
            )

        if Kv is None:
            raise ValueError(
                "Kv must be set as a float, node array, or field name. It was None."
            )

        # Create fields needed for this component if not already existing
        if "volume__lateral_erosion" not in grid.at_node:
            grid.add_zeros("volume__lateral_erosion", at="node")
        self._vol_lat = grid.at_node["volume__lateral_erosion"]

        if "sediment__influx" not in grid.at_node:
            grid.add_zeros("sediment__influx", at="node")
        self._qs_in = grid.at_node["sediment__influx"]

        if "lateral_erosion__depth_increment" not in grid.at_node:
            grid.add_zeros("lateral_erosion__depth_increment", at="node")
        self._dzlat = grid.at_node["lateral_erosion__depth_increment"]

        # for backward compatibility (remove in version 3.0.0+)
        grid.at_node["sediment__flux"] = grid.at_node["sediment__influx"]

        # you can specify the type of lateral erosion model you want to use.
        # But if you don't the default is the undercutting-slump model
        if latero_mech == "TB":
            self._TB = True
            self._UC = False
        else:
            self._UC = True
            self._TB = False
        self._alph = alph
        self._Kv = Kv  # can be overwritten with spatially variable
        self._Klr = float(Kl_ratio)  # default ratio of Kv/Kl is 1. Can be overwritten

        self._dzlatdt = grid.add_zeros(
            "dzlatdt", at="node", clobber=True
        )  # elevation change rate (M/Y)

        # handling Kv for floats (inwhich case it populates an array N_nodes long) or
        # for arrays of Kv. Checks that length of Kv array is good.
        self._Kv = np.ones(self._grid.number_of_nodes, dtype=float) * Kv

    def run_one_step(self, dt=1.0):
        """Calculate lateral erosion for a time period 'dt'.

        Parameters
        ----------
        dt : float
            Model timestep [T]
        """
        Klr = self._Klr
        grid = self._grid
        UC = self._UC
        TB = self._TB
        Kv = self._Kv
        qs_in = self._qs_in
        dzlatdt = self._dzlatdt
        alph = self._alph
        vol_lat = self._grid.at_node["volume__lateral_erosion"]
        kw = 10.0
        F = 0.02
        q = grid.at_node["surface_water__discharge"]
        da = q / grid.dx**2
        
        # May 2, runoff calculated below (in m/s) is important for calculating
        # discharge and water depth correctly. renamed runoffms to prevent
        # confusion with other uses of runoff
        runoffms = (Klr * F / kw) ** 2
        # Kl is calculated from ratio of lateral to vertical K parameters
        Kl = Kv * Klr
        z = grid.at_node["topographic__elevation"]
        # clear qsin for next loop
        qs_in = grid.add_zeros("sediment__influx", at="node", clobber=True)
        qs = grid.add_zeros("qs", at="node", clobber=True)
        lat_nodes = np.zeros(grid.number_of_nodes, dtype=int)
        #dzver = np.zeros(grid.number_of_nodes)
        vol_lat_dt = np.zeros(grid.number_of_nodes)

        # dz_lat needs to be reset. Otherwise, once a lateral node
        # erode's once, it will continue eroding at every subsequent
        # time setp. If you want to track all lateral erosion, create
        # another attribute, or add self.dzlat to itself after each time step.
        self._dzlat.fill(0.0)

        # flow__upstream_node_order is node array contianing downstream to
        # upstream order list of node ids
        s = grid.at_node["flow__upstream_node_order"]
        max_slopes = grid.at_node["topographic__steepest_slope"]
        flowdirs = grid.at_node["flow__receiver_node"]

        # make a list l, where node status is interior (signified by label 0) in s
        interior_s = s[np.where(grid.status_at_node[s] == 0)[0]]
        dwnst_nodes = interior_s.copy()
        # reverse list so we go from upstream to down stream
        dwnst_nodes = dwnst_nodes[::-1]
        max_slopes[:] = max_slopes.clip(0)
        for i in dwnst_nodes:
            # calc deposition and erosion
            #dep = alph * qs_in[i] / da[i]
            #ero = -Kv[i] * da[i] ** (0.5) * max_slopes[i]
            #dzver[i] = dep + ero
            # potential lateral erosion initially set to 0
            petlat = 0.0
            # water depth in meters, needed for lateral erosion calc
            wd = wid_coeff * (da[i] * runoffms) ** wid_exp

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
                if lat_node > 0 and z[lat_node] > z[i]:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative
                    petlat = -Kl[i] * da[i] * max_slopes[i] * inv_rad_curv
                    # the calculated potential lateral erosion is mutiplied by
                    # the length of the node and the bank height, then added
                    # to an array, vol_lat_dt, for volume eroded laterally
                    # *per timestep* at each node. This vol_lat_dt is reset to zero for
                    # each timestep loop. vol_lat_dt is added to itself in case
                    # more than one primary nodes are laterally eroding this lat_node
                    # volume of lateral erosion per timestep
                    vol_lat_dt[lat_node] += abs(petlat) * grid.dx * wd

            # send sediment downstream. sediment eroded from vertical incision
            # and lateral erosion is sent downstream
            #            print("debug before 406")
            qs_in[flowdirs[i]] += (
                qs_in[i] - (petlat * grid.dx * wd)
            )  # qsin to next node
        #qs[:] = qs_in - (dzver * grid.dx**2)
        #dzdt[:] = dzver * dt ##############
        vol_lat[:] += vol_lat_dt * dt
        # this loop determines if enough lateral erosion has happened to change
        # the height of the neighbor node.
        for i in dwnst_nodes:
            lat_node = lat_nodes[i]
            wd = wid_coeff * (da[i] * runoffms) ** wid_exp
            # greater than zero now bc inactive neighbors are value -1
            if lat_node > 0 and z[lat_node] > z[i]:
                # vol_diff is the volume that must be eroded from lat_node so that its
                # elevation is the same as node downstream of primary node
                # UC model: this would represent undercutting (the water height at
                # node i), slumping, and instant removal.
                if UC == 1:
                    voldiff = (z[i] + wd - z[flowdirs[i]]) * grid.dx**2
                # TB model: entire lat node must be eroded before lateral erosion
                # occurs
                if TB == 1:
                    voldiff = (z[lat_node] - z[flowdirs[i]]) * grid.dx**2
                # if the total volume eroded from lat_node is greater than the volume
                # needed to be removed to make node equal elevation,
                # then instantaneously remove this height from lat node. already has
                # timestep in it
                if vol_lat[lat_node] >= voldiff:
                    self._dzlat[lat_node] = z[flowdirs[i]] - z[lat_node]  # -0.001
                    # after the lateral node is eroded, reset its volume eroded to
                    # zero
                    vol_lat[lat_node] = 0.0
        # change height of landscape
        dz = self._dzlat ##############
        z[:] += dz
        return grid, self._dzlat
