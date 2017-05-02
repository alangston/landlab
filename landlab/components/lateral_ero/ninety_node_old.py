"""
lateral erosion component

needs:
fraction of bed exposed: fe=(1-qsqt)
percent lateral erosion of total erosion : Plat=qsqt
lateral erosion: Elat=Plat*E  (E is total erosion, dzdt)
vertical erosion: Evert=(1-Elat)*E



"""

import numpy as np




def NinetyNode(donor, i, flowdirs, link_list, neighbors, diag_neigh):
        debug=0
        print_debug=0
	
	
	#if flow is 90 degrees
	if(donor in diag_neigh and flowdirs[i] in diag_neigh):
		if (debug):
			print "flow is 90 degrees on diagonals from ", donor, " to ", flowdirs[i]
		radcurv_angle=1.37
		#if flow is NE-SE or NW-SW, erode south node
		if (donor==diag_neigh[3] and flowdirs[i]==diag_neigh[0] or 
		donor==diag_neigh[2] and flowdirs[i]==diag_neigh[1]):
			lat_node=neighbors[1]
		#if flow is SW-NW or SE-NE, erode north node
		elif (donor==diag_neigh[1] and flowdirs[i]==diag_neigh[2] or 
		donor==diag_neigh[0] and flowdirs[i]==diag_neigh[3]):
			lat_node=neighbors[3]
		#if flow is SW-SE or NW-NE, erode east node
		elif (donor==diag_neigh[1] and flowdirs[i]==diag_neigh[0] or 
		donor==diag_neigh[2] and flowdirs[i]==diag_neigh[3]):
			lat_node=neighbors[0]
		#if flow is SE-SW or NE-NW, erode west node
		elif (donor==diag_neigh[0] and flowdirs[i]==diag_neigh[1] or 
		donor==diag_neigh[3] and flowdirs[i]==diag_neigh[2]):
			lat_node=neighbors[2]
		#print "lat_node", lat_node
	elif(donor not in diag_neigh and flowdirs[i] not in diag_neigh):
		if (debug):
			print "flow is 90 degrees (not on diagonal) from ", donor, " to ", flowdirs[i]
		radcurv_angle=1.37
		#if flow is N-E or N-W, erode south node
		#this also works if flow is 90 degrees and originates from north
		if (donor==neighbors[0]):
			lat_node=neighbors[2]
		#if flow is from south, erode north node
		elif (donor==neighbors[2]):
			lat_node=neighbors[0]
		#if flow is west, erode east node
		elif (donor==neighbors[1]):
			lat_node=neighbors[3]
		#if flow is east, erode west node
		elif (donor==neighbors[3]):
			lat_node=neighbors[1]
	

		
	
	#lat_node=0
	#radcurv_angle=0.0		
	return lat_node, radcurv_angle
