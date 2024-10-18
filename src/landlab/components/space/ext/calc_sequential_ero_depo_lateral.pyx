import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double x) nogil

from libc.math cimport log,exp
DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

DTYPE_INT = int
ctypedef np.int_t DTYPE_INT_t

DTYPE_LONG = long
ctypedef np.long_t DTYPE_LONG_t

DTYPE_UINT8 = np.uint8
ctypedef np.uint8_t DTYPE_UINT8_t

from libc.stdio cimport printf

from libc.stdio cimport printf

def _sequential_ero_depo_lateral(np.ndarray[DTYPE_INT_t, ndim=1] stack_flip_ud_sel,
                    np.ndarray[DTYPE_INT_t, ndim=1] flow_receivers,
                    np.ndarray[DTYPE_INT_t, ndim=2] links_at_node,
                    np.ndarray[DTYPE_INT_t, ndim=2] active_adjacent_nodes_at_node,
                    np.ndarray[DTYPE_INT_t, ndim=2] diagonal_adjacent_nodes_at_node,
                    np.ndarray[DTYPE_INT_t, ndim=1] lat_nodes,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat_dt,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] vol_lat,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] x_of_node,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] y_of_node,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] cell_area,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] q,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] qs,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] qs_in,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] Es,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] Er,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] Q_to_the_m,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] slope,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] da,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] wd,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] Kl_bed,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] Kl_sed,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] H,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] br,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] z,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] sed_erosion_term,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] bed_erosion_term,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] K_sed,
                    DTYPE_INT_t lateral_erosion,
                    DTYPE_FLOAT_t v,
                    DTYPE_FLOAT_t phi,
                    DTYPE_FLOAT_t F_f,
                    DTYPE_FLOAT_t H_star,
                    DTYPE_FLOAT_t dt,
                    DTYPE_FLOAT_t thickness_lim,
                    DTYPE_FLOAT_t dx,
                    DTYPE_FLOAT_t runoffms):

    """Calculate and qs and qs_in."""
    # define internal variables
    cdef unsigned int node_id
    cdef unsigned int i
    cdef int lat_node
    cdef double H_Before, vol_SSY_riv
    cdef double petlat, petlat_br, petlat_sed, inv_rad_curv, el_lat_node, bedrock_lat_node, br_percent
    vol_SSY_riv =0.0


    for node_id in stack_flip_ud_sel:


        # -------------------------- Lateral Erosion --------------------------
        # Choose lateral node for node i. If node i flows downstream, continue.
        # if node i is the first cell at the top of the drainage network, don't go
        # into this loop because in this case, node i won't have a "donor" node
        if lateral_erosion:
            if node_id in flow_receivers:
                # print('\n')
                # print('Lateral erosion occurs, node id is: ' + str(node_id))

                # node_finder picks the lateral node to erode based on angle
                # between segments between three nodes
                [lat_node, inv_rad_curv] = _node_finder(
                    node_id,
                    dx,
                    flow_receivers,
                    q,
                    links_at_node,
                    active_adjacent_nodes_at_node,
                    diagonal_adjacent_nodes_at_node,
                    x_of_node, y_of_node
                )

                # print('lat_node: ' + str(lat_node))
                # print('inv_rad_curv: ' + str(inv_rad_curv))
                # print('radcurv_angle: ')
                # print(radcurv_angle)


                # node_finder returns the lateral node ID and the radius of curvature
                lat_nodes[node_id] = lat_node
                # if the lateral node is not 0 or -1 continue. lateral node may be
                # 0 or -1 if a boundary node was chosen as a lateral node. then
                # radius of curavature is also 0 so there is no lateral erosion
                if lat_node > 0 and z[lat_node] > z[node_id]:
                    # if the elevation of the lateral node is higher than primary node,
                    # calculate a new potential lateral erosion (L/T), which is negative

                    #%%
                    el_lat_node = max(z[lat_node] - z[node_id],0)
                    bedrock_lat_node = max(0,el_lat_node - H[lat_node])
                    br_percent = bedrock_lat_node/el_lat_node
                    petlat_br = -Kl_bed[node_id] * da[node_id] * slope[node_id] * inv_rad_curv *br_percent
                    petlat_sed = -Kl_sed[node_id] * da[node_id] * slope[node_id] * inv_rad_curv *(1-br_percent)
                    petlat = petlat_br + petlat_sed

                    # print('Percentage petlat_br: ' + str(br_percent))
                    # print('petlat_br: ' + str(petlat_br))
                    # print('petlat_sed: ' + str(petlat_sed))
                    #%%

                    # the calculated potential lateral erosion is mutiplied by
                    # the length of the node and the bank height, then added
                    # to an array, vol_lat_dt, for volume eroded laterally
                    # *per timestep* at each node. This vol_lat_dt is reset to zero for
                    # each timestep loop. vol_lat_dt is added to itself in case
                    # more than one primary nodes are laterally eroding this lat_node
                    # volume of lateral erosion per timestep
                    vol_lat_dt[lat_node] += abs(petlat) * dx * wd[node_id]
                    # send sediment downstream. sediment eroded from vertical incision
                    # and lateral erosion is sent downstream
                    #            print("debug before 406")
                    #Correct for different densities and check variable layers!!
                    qs_in[flow_receivers[node_id]] += - (petlat_br * dx * wd[node_id])
                    qs_in[flow_receivers[node_id]] += - (petlat_sed * dx * wd[node_id]) * (1-phi)
                    vol_lat[lat_node]+= vol_lat_dt[lat_node] * dt

        # ----- ORIGINAL SPACE LARGE SCALE ERODER -------
        qs_out = (qs_in[node_id] +
                  Es[node_id]*cell_area[node_id] +
                  (1.0-F_f)*Er[node_id]* cell_area[node_id]) / \
                        (1.0+(v*cell_area[node_id]/q[node_id]))
        depo_rate = v*qs_out/q[node_id]
        H_loc       =   H[node_id]
        H_Before    =   H[node_id]
        slope_loc   =   slope[node_id]
        sed_erosion_loc = sed_erosion_term[node_id]
        bed_erosion_loc = bed_erosion_term[node_id]

        # Correct for thick soils where soil thickness can grow to inf
        if (H_loc > thickness_lim or slope_loc <= 0 or   sed_erosion_loc==0):
            H_loc += (depo_rate / (1 - phi) - sed_erosion_loc/ (1 - phi)) * dt
        else:
            # Blowup
            if (depo_rate == (K_sed[node_id] * Q_to_the_m[node_id] * slope_loc)) :
                H_loc = H_loc * log(
                    ((sed_erosion_loc/ (1 - phi)) / H_star)
                    * dt
                    + exp(H_loc / H_star)
                )
            # No blowup
            else:
                H_loc = H_star* np.log(
                    (1 / ((depo_rate / (1 - phi)) / (sed_erosion_loc/ (1 - phi))- 1))
                    * (
                        exp((depo_rate / (1 - phi)- (sed_erosion_loc/ (1 - phi)))* (dt / H_star))
                        * (((depo_rate/ (1 - phi)/ (sed_erosion_loc/ (1 - phi))) - 1)* exp(H_loc/ H_star)+1)
                        - 1
                        )
                )
            # In case soil depth evolves to infinity, fall back to no entrainment
            if H_loc == np.inf:
                H_loc =H[node_id]+ (depo_rate / (1 - phi) - sed_erosion_loc/ (1 - phi)) * dt


        H_loc = max(0,H_loc)
        ero_bed = bed_erosion_loc* (exp(-H_loc / H_star))
        qs_out_adj =  qs_in[node_id] - ((H_loc - H_Before)*(1-phi)*cell_area[node_id]/dt) +(1.0-F_f)*ero_bed* cell_area[node_id]# should always be bigger than 0

        qs[node_id] = qs_out_adj
        qs_in[node_id] = 0
        qs_in[flow_receivers[node_id]] += qs[node_id]

        H[node_id] = H_loc
        br[node_id]  += -dt * ero_bed
        vol_SSY_riv += F_f*ero_bed* cell_area[node_id]



    return vol_SSY_riv


def _angle_finder(
        dn,
        cn,
        rn,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] x_of_node,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] y_of_node):

    # define internal variables
    cdef tuple vertex,vec_1,vec_2
    """Find the interior angle between two vectors on a grid.

    Parameters
    ----------
    dn : int
        Node at the end of the first vector.
    cn : int
        Node at the vertex between vectors.
    rn : int
        Node at the end of the second vector.
    x_of_node: array of doubles
    y_of_node: array of doubles

    Returns
    -------
    float or array of float
        Angle between vectors (in radians).

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.lateral_erosion.node_finder import _angle_finder

    >>> grid = RasterModelGrid((3, 4))
    >>> np.rad2deg(_angle_finder(8, 5, 0,grid.x_of_node,grid.y_of_node))
    90.0
    """
    vertex = np.take(x_of_node, cn), np.take(y_of_node, cn)
    vec_1 = np.take(x_of_node, dn) - vertex[0], np.take(y_of_node, dn) - vertex[1]
    vec_2 = np.take(x_of_node, rn) - vertex[0], np.take(y_of_node, rn) - vertex[1]

    return np.arccos(
        (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1])
        / np.sqrt((vec_1[0] ** 2 + vec_1[1] ** 2) * (vec_2[0] ** 2 + vec_2[1] ** 2))
    )


def _forty_five_node(
        DTYPE_INT_t donor,
        DTYPE_INT_t node_id,
        DTYPE_INT_t receiver,
        np.ndarray[DTYPE_INT_t, ndim=1] neighbors,
        np.ndarray[DTYPE_INT_t, ndim=1] diag_neigh):

    cdef signed int lat_node
    cdef double radcurv_angle

    radcurv_angle = 0.67
    lat_node = 0

    # In Landlab 2019: diagonal list goes [NE, NW, SW, SE]. Node list are ordered as [E,N,W,S]
    # if water flows SE-N OR if flow NE-S or E-NW or E-SW, erode west node
    if (
        donor == diag_neigh[0]
        and receiver == neighbors[3]
        or donor == diag_neigh[3]
        and receiver == neighbors[1]
        or donor == neighbors[0]
        and receiver == diag_neigh[2]
        or donor == neighbors[0]
        and receiver == diag_neigh[1]
    ):

        lat_node = neighbors[2]
    # if flow is from SW-N or NW-S or W-NE or W-SE, erode east node
    elif (
        donor == diag_neigh[1]
        and receiver == neighbors[3]
        or donor == diag_neigh[2]
        and receiver == neighbors[1]
        or donor == neighbors[2]
        and receiver == diag_neigh[3]
        or donor == neighbors[2]
        and receiver == diag_neigh[0]
    ):
        lat_node = neighbors[0]
    # if flow is from SE-W or SW-E or S-NE or S-NW, erode north node
    elif (
        donor == diag_neigh[3]
        and receiver == neighbors[2]
        or donor == diag_neigh[2]
        and receiver == neighbors[0]
        or donor == neighbors[3]
        and receiver == diag_neigh[0]
        or donor == neighbors[3]
        and receiver == diag_neigh[1]
    ):
        lat_node = neighbors[1]
    # if flow is from NE-W OR NW-E or N-SE or N-SW, erode south node
    elif (
        donor == diag_neigh[0]
        and receiver == neighbors[2]
        or donor == diag_neigh[1]
        and receiver == neighbors[0]
        or donor == neighbors[1]
        and receiver == diag_neigh[3]
        or donor == neighbors[1]
        and receiver == diag_neigh[2]
    ):
        lat_node = neighbors[3]
    return lat_node, radcurv_angle


def _ninety_node(
        DTYPE_INT_t donor,
        DTYPE_INT_t node_id,
        DTYPE_INT_t receiver,
        np.ndarray[DTYPE_INT_t, ndim=1] neighbors,
        np.ndarray[DTYPE_INT_t, ndim=1] diag_neigh):

    cdef signed int lat_node
    cdef double radcurv_angle


    # if flow is 90 degrees
    if donor in diag_neigh and receiver in diag_neigh:
        radcurv_angle = 1.37
        # if flow is NE-SE or NW-SW, erode south node
        if (
            donor == diag_neigh[0]
            and receiver == diag_neigh[3]
            or donor == diag_neigh[1]
            and receiver == diag_neigh[2]
        ):
            lat_node = neighbors[3]
        # if flow is SW-NW or SE-NE, erode north node
        elif (
            donor == diag_neigh[2]
            and receiver == diag_neigh[1]
            or donor == diag_neigh[3]
            and receiver == diag_neigh[0]
        ):
            lat_node = neighbors[1]
        # if flow is SW-SE or NW-NE, erode east node
        elif (
            donor == diag_neigh[2]
            and receiver == diag_neigh[3]
            or donor == diag_neigh[1]
            and receiver == diag_neigh[0]
        ):
            lat_node = neighbors[0]
        # if flow is SE-SW or NE-NW, erode west node
        elif (
            donor == diag_neigh[3]
            and receiver == diag_neigh[2]
            or donor == diag_neigh[0]
            and receiver == diag_neigh[1]
        ):
            lat_node = neighbors[2]
    elif donor not in diag_neigh and receiver not in diag_neigh:
        radcurv_angle = 1.37
        # if flow is from east, erode west node
        if donor == neighbors[0]:
            lat_node = neighbors[2]
        # if flow is from north, erode south node
        elif donor == neighbors[1]:
            lat_node = neighbors[3]
        # if flow is from west, erode east node
        elif donor == neighbors[2]:
            lat_node = neighbors[0]
        # if flow is from south, erode north node
        elif donor == neighbors[3]:
            lat_node = neighbors[1]
    return lat_node, radcurv_angle


def _straight_node(
        DTYPE_INT_t donor,
        DTYPE_INT_t node_id,
        DTYPE_INT_t receiver,
        np.ndarray[DTYPE_INT_t, ndim=1] neighbors,
        np.ndarray[DTYPE_INT_t, ndim=1] diag_neigh):

    cdef signed int lat_node
    cdef double radcurv_angle
    # ***FLOW LINK IS STRAIGHT, NORTH TO SOUTH***#
    if donor == neighbors[1] or donor == neighbors[3]:
        # print "flow is stright, N-S from ", donor, " to ", flowdirs[i]
        radcurv_angle = 0.23
        # neighbors are ordered E,N,W, S
        # if the west cell is boundary (neighbors=-1), erode from east node
        if neighbors[2] == -1:
            lat_node = neighbors[0]
        elif neighbors[0] == -1:
            lat_node = neighbors[2]
        else:
            # if could go either way, choose randomly. 0 goes East, 1 goes west
            ran_num = np.random.randint(0, 2)
            if ran_num == 0:
                lat_node = neighbors[0]
            if ran_num == 1:
                lat_node = neighbors[2]
    # ***FLOW LINK IS STRAIGHT, EAST-WEST**#
    elif donor == neighbors[0] or donor == neighbors[2]:
        radcurv_angle = 0.23
        #  Node list are ordered as [E,N,W,S]
        # if the north cell is boundary (neighbors=-1), erode from south node
        if neighbors[1] == -1:
            lat_node = neighbors[3]
        elif neighbors[3] == -1:
            lat_node = neighbors[1]
        else:
            # if could go either way, choose randomly. 0 goes south, 1 goes north
            ran_num = np.random.randint(0, 2)
            if ran_num == 0:
                lat_node = neighbors[1]
            if ran_num == 1:
                lat_node = neighbors[3]
    # if flow is straight across diagonal, choose node to erode at random
    elif donor in diag_neigh and receiver in diag_neigh:
        radcurv_angle = 0.23
        if receiver == diag_neigh[0]:
            poss_diag_nodes = neighbors[0 : 1 + 1]
        elif receiver == diag_neigh[1]:
            poss_diag_nodes = neighbors[1 : 2 + 1]
        elif receiver == diag_neigh[2]:
            poss_diag_nodes = neighbors[2 : 3 + 1]
        elif receiver == diag_neigh[3]:
            poss_diag_nodes = [neighbors[3], neighbors[0]]
        ran_num = np.random.randint(0, 2)
        if ran_num == 0:
            lat_node = poss_diag_nodes[0]
        if ran_num == 1:
            lat_node = poss_diag_nodes[1]
    return lat_node, radcurv_angle


def _node_finder(
        DTYPE_INT_t node_id,
        DTYPE_FLOAT_t dx,
        np.ndarray[DTYPE_INT_t, ndim=1] flow_receivers,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] q,
        np.ndarray[DTYPE_INT_t, ndim=2] links_at_node,
        np.ndarray[DTYPE_INT_t, ndim=2] active_adjacent_nodes_at_node,
        np.ndarray[DTYPE_INT_t, ndim=2] diagonal_adjacent_nodes_at_node,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] x_of_node,
        np.ndarray[DTYPE_FLOAT_t, ndim=1] y_of_node,
        ):
    """Find lateral neighbor node of the primary node for straight, 45 degree,
    and 90 degree channel segments.

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object
    i : int
        node ID of primary node
    flowdirs : array
        Flow direction array
    drain_area : array
        drainage area array

    Returns
    -------
    lat_node : int
        node ID of lateral node
    radcurv_angle : float
        inverse radius of curvature of channel at lateral node
    """
    cdef unsigned int receiver
    cdef unsigned int donor, maxinfln
    cdef signed int lat_node
    cdef double inv_rad_curv, angle_diff
    cdef tuple inflow
    cdef link_list,neighbors,diag_neigh
    cdef np.ndarray[DTYPE_LONG_t, ndim=1] maxinfl,


    # receiver node of flow is flowdirs[i]
    receiver = flow_receivers[node_id]

    # find indicies of where flowdirs=i to find donor nodes.
    # will donor nodes always equal the index of flowdir list?
    inflow = np.where(flow_receivers == node_id)

    # if there are more than 1 donors, find the one with largest drainage area
    if len(inflow[0]) > 1:
        if isinstance(inflow[0][np.where(q[inflow] == max(q[inflow]))], np.ndarray):

            maxinfl = inflow[0][np.where(q[inflow] == max(q[inflow]))]

            # if donor nodes have same drainage area, choose one randomly
            donor = maxinfl[np.random.randint(0, len(maxinfl))]
                # donor = [maxinfln]
        else:
            donor = inflow[0][np.where(q[inflow] == max(q[inflow]))]
        # if inflow is empty, no donor
    elif len(inflow[0]) == 0:
        donor = node_id
    # else donor is the only inflow
    else:
        donor = inflow[0]


    # now we have chosen donor cell, next figure out if inflow/outflow lines are
    # straight, 45, or 90 degree angle. and figure out which node to erode
    link_list = links_at_node[node_id]
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = active_adjacent_nodes_at_node[node_id]
    # this gives list of all diagonal neighbors for specified node
    # the order of this list is: [NE,NW,SW,SE]
    diag_neigh = diagonal_adjacent_nodes_at_node[node_id]

    angle_diff = np.rad2deg(
        _angle_finder(
            donor,
            node_id,
            receiver,
            x_of_node,
            y_of_node)
        )

    # print('donor is: ' + str(donor))
    # print('node_id is: ' + str(node_id))
    # print('receiver is: ' + str(receiver))
    # print('angle_diff is: ' + str(angle_diff))

    if (donor == flow_receivers[node_id]) or (donor == node_id):
        # this is a sink. no lateral ero
        radcurv_angle = 0.0
        lat_node = 0
    elif np.isclose(angle_diff, 0.0) or np.isclose(angle_diff, 180.0):
        [lat_node, radcurv_angle] = _straight_node(
            donor, node_id, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 45.0) or np.isclose(angle_diff, 135.0):
        [lat_node, radcurv_angle] = _forty_five_node(
            donor, node_id, receiver, neighbors, diag_neigh
        )
    elif np.isclose(angle_diff, 90.0):

        [lat_node, radcurv_angle] = _ninety_node(
            donor, node_id, receiver, neighbors, diag_neigh
        )
    else:
        lat_node = 0
        radcurv_angle = 0.0


    # INVERSE radius of curvature.
    radcurv_angle = radcurv_angle / dx

    # print('lat_node: ' + str(lat_node))
    # print('radcurv_angle: ' + str(radcurv_angle))

    return int(lat_node), radcurv_angle
