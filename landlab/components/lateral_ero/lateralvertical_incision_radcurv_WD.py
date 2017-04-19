#! /usr/env/python
# -*- coding: utf-8 -*-
"""
Component for detachment-limited fluvial incision using a simple power-law model.

E=K (tau^P - tau_c^P)

E=incision rate (M Y^(-1) )
K=bedrock erodibility (M^(1-3m) Y^(m-1) ) #read in from input file
tau=shear stress, rho*g*h*S
rho=density of water
g=gravity
h=height of water
S=slope of landscape (negative of the gradient in topography, dimensionless, 
and only applies on positive slopes)  
tau_c=critical shear stress, read in from input file


NOTE, in units above M=meters.  This component assumes that variables are given
in units of meters and years, including rainfall!

NOTE, Only incision happens in this class.  NO DEPOSITION.  
NO TRACKING OF SEDIMENT FLUX.

This assumes that a grid has already been instantiated.

To run this, first instantiate the class member, then run one storm
incisor = PowerLawIncision('input_file_name',grid)
z = incisior.run_one_storm(grid, z, rainrate=optional, storm_duration=optional)

Note that the component knows the default rainfall rate (m/yr) and storm duration (yr)
so these values need not be passed in.  Elevationare eroded and sent back.


"""

#import landlab
from landlab import ModelParameterDictionary
#from landlab.components.flow_routing.flow_routing_D8 import RouteFlowD8
from landlab.components.flow_routing.route_flow_dn import FlowRouter
from landlab.components.rad_curv.node_finder2 import Node_Finder2
from landlab.components.rad_curv.radius_curv_dz import radius_curv_dz
#from landlab.components.flow_accum.flow_accumulation2 import AccumFlow
from landlab.utils import structured_grid
import numpy as np
from random import uniform
import matplotlib.pyplot as plt

class LateralVerticalIncisionRD(object):
#class LateralIncision(object):

    def __init__(self, input_stream, grid, current_time=0.):

        self.grid = grid
        #create and initial grid if one doesn't already exist
        #if self.grid==None:
        #    self.grid = create_and_initialize_grid(input_stream)

        self.current_time = current_time
        self.initialize(grid, input_stream)

    def initialize(self, grid, input_stream):

        # Create a ModelParameterDictionary for the inputs
        if type(input_stream)==ModelParameterDictionary:
            inputs = input_stream
        else:
            inputs = ModelParameterDictionary(input_stream)

        # Read input/configuration parameters
        self.alph = inputs.get('ALPH', ptype=float)
        self.Kv = inputs.get('KV_COEFFICIENT', ptype=float)
        #self.Klr = inputs.get('KL_RATIO', ptype=float)
        self.rain_duration_yr = inputs.get('RAIN_DURATION_YEARS', ptype=float)
        self.inlet_node = inputs.get('INLET_NODE', ptype=float)
        self.inlet_area = inputs.get('INLET_AREA', ptype=float)
        self.qsinlet = inputs.get('QSINLET', ptype=float)
        self.frac = 0.3 #for time step calculations

        # Set up state variables

        #initialize qsin for each interior node, all zero initially.
        self.qsin = grid.zeros(centering='node')    # qsin (M^3/Y)
        self.qt = grid.zeros(centering='node')    # transport capacity
        self.dzdt = grid.zeros(centering='node')    # elevation change rate (M/Y)
        self.qsqt = grid.zeros(centering='node')    # potential elevation change rate (M/Y)
    def save_multipagepdf(f_handle):    
    	savefig(f_handle, format='pdf')
    	close()


    def run_one_storm(self, grid, z, vol_lat, rainrate=None, storm_dur=None, qsinlet=None, inlet_area=None):

        if rainrate==None:
            rainrate = self.rainfall_myr
        if storm_dur==None:
            storm_dur = self.rain_duration_yr   
        inlet_node=self.inlet_node
        #inlet_area=self.inlet_area
        if qsinlet==None:
            qsinlet=self.qsinlet
        if inlet_area==None:
            inlet_area=self.inlet_area


        Kv=self.Kv
        #Klr=self.Klr
        frac = self.frac
        qsin=self.qsin
        qt=self.qt
        dzdt=self.dzdt
        qsqt=self.qsqt
        alph=self.alph
        #**********ADDED FOR WATER DEPTH CHANGE***************
        #vs is a constant settling velocity. runoff is run off rate in m/second
        vs=1e-6
        runoff=vs/alph
        kw=10.
        F=0.02
        #September 11. Making Klratio dependent on alpha and rain rate calculated from alpha
        Klr=runoff**0.5*kw/F
        #ratio of lateral to vertical K parameters
        Kl=Kv*Klr

        dx=grid.dx
        nr=grid.number_of_node_rows
        nc=grid.number_of_node_columns
        interior_nodes = grid.get_core_nodes()
        boundary_nodes=structured_grid.boundary_nodes((nr,nc))
        #clear qsin for next loop
        qsin = grid.zeros(centering='node')
        qsqt = grid.zeros(centering='node')
        #eronode=np.zeros(grid.number_of_nodes)
        lat_nodes=np.zeros(grid.number_of_nodes)
        dzlat=np.zeros(grid.number_of_nodes)
        dzver=np.zeros(grid.number_of_nodes)
        vol_lat_dt=np.zeros(grid.number_of_nodes)
        #z_bank=np.zeros(grid.number_of_nodes)
	#vol_diff=np.zeros(grid.number_of_nodes)
        #instantiate variable of type RouteFlowDNClass
        flow_router = FlowRouter(grid)

        #4.6.14: also passing it areas of nodes for inlet nodes
        node_area=dx**2 * np.ones(grid.number_of_nodes)
        node_area[inlet_node]=inlet_area
        #node_area=node_area[interior_nodes]
        numcel=grid.number_of_cells
        numnode=grid.number_of_nodes
        
        
        
        flowdirs, drain_area, q, max_slopes, s, receiver_link = flow_router.route_flow(elevs=z, node_cell_area=node_area, runoff_rate=runoff)
        #line below added for analytical solution REMOVE FOR REAL RUNS!
        #drain_area = np.ones(len(drain_area))*dx**2
        #order interior nodes
        #find interior nodes in downstream ordered vector s

        #make a list l, where node status is interior in s
        l=s[np.where((grid.node_status[s] == 0))[0]]
        #this misses an interior nodes that is set as constant value, 1
        #but the below grabs nodes that are set as open boundaries. no good for me here
        #l2=s[np.where((grid.node_status[s] == 1))[0]]
        #combine lists
        #dwnst_nodes=np.insert(l,0,l2)
        dwnst_nodes=l
        #reverse list so we go from upstream to down stream
        #print "dwnst_nodes before reversal", dwnst_nodes
        dwnst_nodes=dwnst_nodes[::-1]
        #local time
        time=0
        dt = storm_dur
        dtmax=storm_dur
              

        while time < storm_dur:
            #Calculate incision rate, should be in m/yr, should be negative
            #First make sure that there are no negative (uphill slopes)
            #Set those to zero, because incision rate should be zero there.
            max_slopes.clip(0)
            #here calculate dzdt for each node, with initial time step
            #print "dwnst_nodes", dwnst_nodes
            for i in dwnst_nodes:
                if i==inlet_node:
                    qsin[i]=qsinlet
                    #print "qsin[inlet]", qsin[i]
                    #print "inlet area", drain_area[i]
                #calc deposition and erosion
                #dzver is vertical erosion/deposition only
                dep = alph*qsin[i]/drain_area[i]
                ero = -Kv * drain_area[i]**(0.5)*max_slopes[i]
                dzver[i] =  dep + ero
                #qsqt[i] = dep/-ero

                #calculate transport capacity
                qt[i]=Kv*drain_area[i]**(3./2.)*max_slopes[i]/alph
							
                #lateral erosion component
                #potential lateral erosion initially set to 0
                petlat=0.                
                #bank height initially set to 0
                z_bank=0.0
                #**********ADDED FOR WATER DEPTH CHANGE***************
                #water depth
                wd=0.4*q[i]**0.35
                #print "i", i
                #print "wd[i]", wd
                
                #if node i flows downstream, continue. That is, if node i is the 
                #first cell at the top of the drainage network, don't go into this
                # loop because in this case, node i won't have a "donor" node found
                # in NodeFinder and needed to calculate the angle difference
                if i in flowdirs:
                #if flowdirs[i] == i:
                    #Node_finder picks the lateral node to erode based on angle
                    # between segments between three nodes
                    [lat_node, inv_rad_curv]=Node_Finder2(grid, i, flowdirs, drain_area)
                    #node_finder returns the lateral node ID and the radius of curvature
                    lat_nodes[i]=lat_node
                    #if the lateral node is not 0 continue. lateral node may be 
                    # 0 if a boundary node was chosen as a lateral node. then 
                    # radius of curavature is also 0 so there is no lateral erosion
                    if lat_node!=0.0:
                        #if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                        if z[lat_node] > z[i]:                           
                                                        
                            petlat=-Kl*drain_area[i]*max_slopes[i]*inv_rad_curv
                            
                            #bank height. 
                            z_bank=z[lat_node]-z[i]
                            
                            #the calculated potential lateral erosion is mutiplied by the length of the node 
                            #and the bank height, then added to an array, vol_lat_dt, for volume eroded 
                            #laterally  *per year* at each node. This vol_lat_dt is reset to zero for 
                            #each timestep loop. vol_lat_dt is added to itself more than one primary nodes are
                            # laterally eroding this lat_node                       
                            vol_lat_dt[lat_node]+=abs(petlat)*dx*wd                           
                            
                		
                # the following is always done, even if lat_node is 0 or lat_node 
                # lower than primary node. however, petlat is 0 in these cases
                    
                #send sediment downstream. sediment eroded from vertical incision 
                # and lateral erosion is sent downstream           	            	
                qsin[flowdirs[i]]+=qsin[i]-(dzver[i]*dx**2)-(petlat*dx*wd)   #qsin to next node
                #qsqt[i]=qsin[i]/qt[i]
                     

                if (0):
                    #if drain_area[i] > 500:
                    print 'node id', i
                    print "flowdirs[i]", flowdirs[i]
                    print "drain_area[i]", drain_area[i]
                    print "slope[i]", max_slopes[i]
                    print "qsin", qsin[i]
                    print "qsin downstream", qsin[flowdirs[i]]
                    print 'qt', qt[i]
                    print "qsqt", qsqt[i]
                    print "dzdt", dzdt[i]
                    print " "
                    
            if(0):
            	print "dzlat", dzlat
                print "dzdt", dzdt
                
            dzdt=dzver            
            #Do a time-step check
            #If the downstream node is eroding at a slower rate than the 
            #upstream node, there is a possibility of flow direction reversal,
            #or at least a flattening of the landscape. 
            #Limit dt so that this flattening or reversal doesn't happen.
            #How close you allow these two points to get to eachother is
            #determined by the variable frac.
            
            #dtn is an arbitrarily large number to begin with, but will be adapted as we step through
            #the nodes
            dtn=dt*50 #starting minimum timestep for this round
            for i in dwnst_nodes:
                if(0):
                    print "node", i
                #are points converging? ie, downstream eroding slower than upstream
                dzdtdif = dzdt[flowdirs[i]]-dzdt[i]
                #if points converging, find time to zero slope
                if dzdtdif > 0. and max_slopes[i] > 1e-7:
                    dtflat = (z[i]-z[flowdirs[i]])/dzdtdif	#time to flat between points
                    #if time to flat is smaller than dt, take the lower value
                    if dtflat < dtn:
                        dtn = dtflat
                    #if dzdtdif*dtflat will make upstream lower than downstream, find time to flat
                    if dzdtdif*dtn > (z[i]-z[flowdirs[i]]):
                        dtn=(z[i]-z[flowdirs[i]])/dzdtdif
                        

            #print "out of ts loop"
            dtn*=frac			
            #new minimum timestep for this round of nodes
            dt=min(dtn, dt)
            #should now have a stable timestep.
            #print "stable time step=", dt
            
            #******Needed a stable timestep size to calculate volume eroded from lateral node for 
            # each stable time step********************
            
            #vol_lat is the total volume eroded from the lateral nodes through
            # the entire model run. So vol_lat is itself plus vol_lat_dt (for current loop)
            # times stable timestep size
            if(0):
                print "vol_lat before", vol_lat
                print "dt", dt
            vol_lat += vol_lat_dt*dt
            if (0): 
                print "vol_lat_dt", vol_lat_dt
                print "vol_lat after", vol_lat
                
            
            
                        
            for i in dwnst_nodes:
                lat_node=lat_nodes[i]
                if lat_node!=0.0:
                        if z[lat_node] > z[i]:                        
                            #vol_diff is the volume that must be eroded from lat_node so that its
                            # elevation is the same as primary node
                            #August 23: Changing from above so that lateral node will be lower
                            # than the downstream node of the primary node. 
                            voldiff=(z[lat_node]-z[flowdirs[i]])*dx**2
                            #if the total volume eroded from lat_node is greater than the volume 
                            # needed to be removed to make node equal elevation, 
                            # then instantaneously remove this height from lat node. already has timestep in it    
                            if vol_lat[lat_node]>=voldiff:
                                dzlat[lat_node]=z[flowdirs[i]]-z[lat_node]-0.001
                                if(0):
                                    print "chunk of lateral erosion occured"                        
                                    print "node", lat_node
                                    print "dzlat[lat_node]", dzlat[lat_node]
                                    print "vol_lat[latnode]", vol_lat[lat_node]
                                    print "vol_lat_dt", vol_lat_dt[lat_node]
                                    print "dt", dt
                                    print "z[lat]", z[lat_node]
                                    print "z[i]", z[i]
                                    print "z[flowdirs[i]]", z[flowdirs[i]]
                                    print "volume diff", voldiff
                                    print delta
                                #after the lateral node is eroded, reset its volume eroded to zero
                                vol_lat[lat_node]=0.0
            
            #multiply dzver(changed to dzdt above) by timestep size and combine with lateral erosion
            #dzlat, which is already a length for the chosen time step
            dz=dzdt*dt+dzlat
            
            
            #change height of landscape
            z[interior_nodes]=dz[interior_nodes]+z[interior_nodes]
            #update elapsed time
            time=dt+time           
                       
            #check to see that you are within 0.01% of the storm duration, if so done, if not continue

            if time > 0.9999*storm_dur:
                time = storm_dur
                #recalculate flow directions for plotting
                flowdirs, drain_area, q, max_slopes, s, receiver_link = flow_router.route_flow(elevs=z, node_cell_area=node_area, runoff_rate=runoff)
                #recalculate downstream order
                dsind = np.where((s >= min(interior_nodes)) & (s <= max(interior_nodes)))
                l=s[np.where((grid.node_status[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]
            else:
                dt = storm_dur - time
                #recalculate flow directions
                flowdirs, drain_area, q, max_slopes, s, receiver_link = flow_router.route_flow(elevs=z, node_cell_area=node_area, runoff_rate=runoff)
                #recalculate downstream order
                dsind = np.where((s >= min(interior_nodes)) & (s <= max(interior_nodes)))
                l=s[np.where((grid.node_status[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]
                #clear qsin for next loop
                qsin = grid.zeros(centering='node')
                qt = grid.zeros(centering='node')
                lat_nodes=np.zeros(grid.number_of_nodes)
                dzlat=np.zeros(grid.number_of_nodes)
                vol_lat_dt=np.zeros(grid.number_of_nodes)
                dzver=np.zeros(grid.number_of_nodes)
	        
        
        return z, qt, qsin, dzdt, dzlat, flowdirs, drain_area, dwnst_nodes, max_slopes, dt       