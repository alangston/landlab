"""




"""

import numpy as np




def StraightNode(donor, i, flowdirs, conn_link2, link_list, neighbors, diag_neigh):
        debug=0
        print_debug=0

	#####FLOW LINK IS STRAIGHT, NORTH TO SOUTH######
	if ((conn_link2==link_list[0] or conn_link2==link_list[2])):
		
		#if (debug):
		#	print "flow is stright, N-S from ", donor, " to ", flowdirs[i]
		radcurv_angle=0.23
		if(print_debug):
			print "erode node to east or west"
		# links are ordered as [N, W, S, E]. Neighbor list is ordered as [E, S, W, N]
		#july7, link list and neighbor lists have been reordered from old way, 
		#which was used when I first developed the random neighbor choosing algorithm.
		# Old way:  [S,W,N,E]. Neighbor list is ordered as [E,N,W,S]
		#the old order still works if chosing random neighbor to erode if flow N-S
		#because west and east haven't changed positions
		#if the west cell is boundary (link=-1), erode from east node
		if link_list[1]==-1:
			lat_node=neighbors[0]
			if(print_debug):  
				print "eroding east node, id = ", lat_node
		elif link_list[3]==-1:
			lat_node=neighbors[2]
			if(print_debug):
				print "eroding west node, id = ", lat_node
		
		else:
		#if could go either way, choose randomly. 0 goes East, 1 goes west
			ran_num=np.random.randint(0,2)
			if ran_num==0:
				lat_node=neighbors[0]
				if(print_debug):
					print "eroding east node (random)"
			if ran_num==1:
				lat_node=neighbors[2]
				if(print_debug):
					print "eroding west node (random)"				
		
	#####FLOW LINK IS STRAIGHT, EAST-WEST#####	
	elif (conn_link2==link_list[1] or conn_link2==link_list[3]):
		if (debug):
			print "flow is stright, E-W"
		radcurv_angle=0.23
		if(print_debug):
			print "erode node to north or south"
		# links are ordered as [S,W,N,E]. Node list are ordered as [E,N,W,S]
		#if the north cell is boundary (link=-1), erode from south node
		if link_list[0]==-1:
			lat_node=neighbors[1]
			if(print_debug):  
				print "eroding south node, id = ", lat_node
		elif link_list[2]==-1:
			lat_node=neighbors[3]
			if(print_debug):
				print "eroding north node, id = ", lat_node
		else:
		#if could go either way, choose randomly. 0 goes south, 1 goes north
			ran_num=np.random.randint(0,2)
			if ran_num==0:
				lat_node=neighbors[3]
				if(print_debug):
					print "eroding north node (random), id = ", lat_node
			if ran_num==1:
				lat_node=neighbors[1]
				if(print_debug):
					print "eroding south node (random), id = ", lat_node
	#if flow is straight across diagonal, choose node to erode at random
	elif(donor in diag_neigh and flowdirs[i] in diag_neigh):
		radcurv_angle=0.23
		if (debug):
			print "flow is straight across diagonal"
		if flowdirs[i]==diag_neigh[0]:
			if(print_debug):
				print "erode east or north"
			poss_diag_nodes=neighbors[0:1+1]
			if(print_debug):
				print "poss_diag_nodes", poss_diag_nodes
		elif flowdirs[i]==diag_neigh[1]:
			if(print_debug):
				print "erode north or west"
			poss_diag_nodes=neighbors[1:2+1]
			if(print_debug):
				print "poss_diag_nodes", poss_diag_nodes
		elif flowdirs[i]==diag_neigh[2]:
			if(print_debug):
				print "erode west or south"
			poss_diag_nodes=neighbors[2:3+1]
			if(print_debug):
				print "poss_diag_nodes", poss_diag_nodes
		elif flowdirs[i]==diag_neigh[3]:
			if(print_debug):
				print "erode south or east"
			poss_diag_nodes=[neighbors[3], neighbors[0]]
			if(print_debug):
				print "poss_diag_nodes", poss_diag_nodes
		ran_num=np.random.randint(0,2)
		if ran_num==0:
			lat_node=poss_diag_nodes[0]
			if(print_debug):
				print "eroding first poss diag node (random), id = ", lat_node
		if ran_num==1:
			lat_node=poss_diag_nodes[1]
			if(print_debug):
				print "eroding second poss diag node (random), id = ", lat_node
													
	
	return lat_node, radcurv_angle
