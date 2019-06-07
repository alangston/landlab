"""
finds lateral neighbor node for straight, 45 degree, and 90 degree channel segments
"""

import numpy as np
import math

# %%


def angle_finder(grid, dn, cn, rn):
    xcoord = grid.node_axis_coordinates(axis=0)
    ycoord = grid.node_axis_coordinates(axis=1)
    sl1 = (ycoord[cn] - ycoord[dn]) / (xcoord[cn] - xcoord[dn])
    sl2 = (ycoord[rn] - ycoord[cn]) / (xcoord[rn] - xcoord[cn])
    angle1 = math.degrees(math.atan(sl1))
    angle2 = math.degrees(math.atan(sl2))
    angle_diff = angle2 - angle1
    angle_diff = abs(angle2 - angle1)
    return angle_diff
# %%


def FortyfiveNode(donor, i, receiver, neighbors, diag_neigh):
    radcurv_angle = 0.67
    # april 9. getting weird flow patterns in a 4x4 grid. flow from 10->9->6.
    # resultign in no lateral node assigned. for a hack for now, start with
    # lateral node=0,
    lat_node = 0
    # LL 2017: diagonal list goes [NE, NW, SW, SE]. Node list are ordered as [E,N,W,S]
    # if water flows SE-N OR if flow NE-S or E-NW or E-SW, erode west node
    if (donor == diag_neigh[0] and receiver == neighbors[3] or
        donor == diag_neigh[3] and receiver == neighbors[1]
        or donor == neighbors[0] and receiver == diag_neigh[2] or
            donor == neighbors[0] and receiver == diag_neigh[1]):
        lat_node = neighbors[2]
    # if flow is from SW-N or NW-S or W-NE or W-SE, erode east node
    elif (donor == diag_neigh[1] and receiver == neighbors[3] or
          donor == diag_neigh[2] and receiver == neighbors[1] or
          donor == neighbors[2] and receiver == diag_neigh[3] or
          donor == neighbors[2] and receiver == diag_neigh[0]):
        lat_node = neighbors[0]
    # if flow is from SE-W or SW-E or S-NE or S-NW, erode north node
    elif (donor == diag_neigh[3] and receiver == neighbors[2] or
          donor == diag_neigh[2] and receiver == neighbors[0] or
          donor == neighbors[3] and receiver == diag_neigh[0] or
          donor == neighbors[3] and receiver == diag_neigh[1]):
        lat_node = neighbors[1]
    # if flow is from NE-W OR NW-E or N-SE or N-SW, erode south node
    elif (donor == diag_neigh[0] and receiver == neighbors[2] or
          donor == diag_neigh[1] and receiver == neighbors[0] or
          donor == neighbors[1] and receiver == diag_neigh[3] or
          donor == neighbors[1] and receiver == diag_neigh[2]):
        lat_node = neighbors[3]
    return lat_node, radcurv_angle
# %% Ninety Node


def NinetyNode(donor, i, receiver, link_list, neighbors, diag_neigh):
    # if flow is 90 degrees
    if(donor in diag_neigh and receiver in diag_neigh):
        radcurv_angle = 1.37
        # if flow is NE-SE or NW-SW, erode south node
        if (donor == diag_neigh[0] and receiver == diag_neigh[3] or
                donor == diag_neigh[1] and receiver == diag_neigh[2]):
            lat_node = neighbors[3]
        # if flow is SW-NW or SE-NE, erode north node
        elif (donor == diag_neigh[2] and receiver == diag_neigh[1] or
              donor == diag_neigh[3] and receiver == diag_neigh[0]):
            lat_node = neighbors[1]
        # if flow is SW-SE or NW-NE, erode east node
        elif (donor == diag_neigh[2] and receiver == diag_neigh[3] or
              donor == diag_neigh[1] and receiver == diag_neigh[0]):
            lat_node = neighbors[0]
         # if flow is SE-SW or NE-NW, erode west node
        elif (donor == diag_neigh[3] and receiver == diag_neigh[2] or
              donor == diag_neigh[0] and receiver == diag_neigh[1]):
            lat_node = neighbors[2]
         # print "lat_node", lat_node
    elif(donor not in diag_neigh and receiver not in diag_neigh):
        radcurv_angle = 1.37
        # if flow is from east, erode west node
        if (donor == neighbors[0]):
            lat_node = neighbors[2]
        # if flow is from north, erode south node
        elif (donor == neighbors[1]):
            lat_node = neighbors[3]
        # if flow is from west, erode east node
        elif (donor == neighbors[2]):
            lat_node = neighbors[0]
        # if flow is from south, erode north node
        elif (donor == neighbors[3]):
            lat_node = neighbors[1]
    return lat_node, radcurv_angle

# %% Straight Node finder


def StraightNode(donor, i, receiver, neighbors, diag_neigh):
    #####FLOW LINK IS STRAIGHT, NORTH TO SOUTH######
    if ((donor == neighbors[1] or donor == neighbors[3])):
        #	print "flow is stright, N-S from ", donor, " to ", flowdirs[i]
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
    #####FLOW LINK IS STRAIGHT, EAST-WEST#####
    elif (donor == neighbors[0] or donor == neighbors[2]):
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
    elif(donor in diag_neigh and receiver in diag_neigh):
        radcurv_angle = 0.23
        if receiver == diag_neigh[0]:
            poss_diag_nodes = neighbors[0:1 + 1]
        elif receiver == diag_neigh[1]:
            poss_diag_nodes = neighbors[1:2 + 1]
        elif receiver == diag_neigh[2]:
            poss_diag_nodes = neighbors[2:3 + 1]
        elif receiver == diag_neigh[3]:
            poss_diag_nodes = [neighbors[3], neighbors[0]]
        ran_num = np.random.randint(0, 2)
        if ran_num == 0:
            lat_node = poss_diag_nodes[0]
        if ran_num == 1:
            lat_node = poss_diag_nodes[1]
    return lat_node, radcurv_angle
# %% Node finder


def Node_Finder2(grid, i, flowdirs, drain_area):
    debug = 0
    print_debug = 0
    # receiver node of flow is flowdirs[i]
    receiver = flowdirs[i]

    # find indicies of where flowdirs=i to find donor nodes.
    # will donor nodes always equal the index of flowdir list?
    inflow = np.where(flowdirs == i)

    # if there are more than 1 donors, find the one with largest drainage area

    if len(inflow[0]) > 1:
        drin = drain_area[inflow]
        drmax = max(drin)
        maxinfl = inflow[0][np.where(drin == drmax)]
        if(debug):
            print("old inflow", inflow[0])
        # if donor nodes have same drainage area, choose one randomly
        if len(maxinfl) > 1:
            ran_num = np.random.randint(0, len(maxinfl))
            maxinfln = maxinfl[ran_num]
            donor = [maxinfln]
            if(debug):
                print("random donor", donor)
        else:
            donor = maxinfl
            if(debug):
                print("donor with larger drainage area", donor)
        # if inflow is empty, no donor
    elif len(inflow[0]) == 0:
        if(debug):
            print("no donor")
        donor = i
    # else donor is the only inflow
    else:
        donor = inflow[0]

    if(print_debug):
        print("donor", donor)
        print("i", i)
        print("receiver", receiver)
    # now we have chosen donor cell, next figure out if inflow/outflow lines are
    # straight, 45, or 90 degree angle. and figure out which node to erode
    link_list = grid.links_at_node[i]
    # this gives list of active neighbors for specified node
    # the order of this list is: [E,N,W,S]
    neighbors = grid.active_neighbors_at_node[i]
    # this gives list of all diagonal neighbors for specified node
    # the order of this list is: [NE,NW,SW,SE]
    diag_neigh = grid.diagonal_adjacent_nodes_at_node[i]
    linkdirs = grid.active_link_dirs_at_node[i]
    angle_diff = angle_finder(grid, donor, i, receiver)

    if(debug):
        print("link_list", link_list)
        print("neighbors", neighbors)
        print("diagneighbors", diag_neigh)
        print("angle_diff", angle_diff)
        print(" ")

    if donor == flowdirs[i]:
        # this is a sink. no lateral ero
        if(debug):
            print("this is a sink")
        radcurv_angle = 0.
        lat_node = 0
    if donor == i:
        # this is a sink. no lateral ero
        if(debug):
            print("this is a sink")
        radcurv_angle = 0.
        lat_node = 0
    if angle_diff == 0.0:
        [lat_node, radcurv_angle] = StraightNode(
            donor, i, receiver, neighbors, diag_neigh)
    if (angle_diff == 45.0 or angle_diff == 135.0):
        [lat_node, radcurv_angle] = FortyfiveNode(
            donor, i, receiver, neighbors, diag_neigh)
    if angle_diff == 90.0:
        [lat_node, radcurv_angle] = NinetyNode(
            donor, i, receiver, link_list, neighbors, diag_neigh)
    if(debug):
        print("lat_node", lat_node)
        print("end of node finder")
    if lat_node > 2e9:
        # print "old latnode", lat_node
        lat_node = 0
        radcurv_angle = 0.0

    dx = grid.dx
    # May24, 2017: this is actually INVERSE radius of curvature. It works out
    # in the main lateral ero
    radcurv_angle = radcurv_angle / dx
    return int(lat_node), radcurv_angle
