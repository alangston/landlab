#! /usr/env/python
# -*- coding: utf-8 -*-
"""
April 4, 2019 Starting to rehab lateral erosion
ALangston


"""


#below was C/P from diffuser and this caused the import to work for the first time!!!!
from landlab import (
    FIXED_GRADIENT_BOUNDARY,
    INACTIVE_LINK,
    Component,
    FieldError,
    RasterModelGrid,
)
from pylab import *
from landlab import ModelParameterDictionary
from landlab.components.flow_director import FlowDirectorD8
from landlab.components.flow_accum import FlowAccumulator
#from landlab.components import DepressionFinderAndRouter
#from landlab.components import(FlowDirectorD8, 
#                               FlowDirectorDINF, 
#                               FlowDirectorMFD, 
#                               FlowDirectorSteepest)
#from landlab.components import FlowAccumulator
from landlab.components.lateral_erosion.node_finder2 import Node_Finder2
from landlab.utils import structured_grid
import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import uniform
import matplotlib.pyplot as plt

class LateralEroder(Component):
    """
    Laterally erode node through fluvial erosion.

    Landlab component that finds the node to laterally erode and calculates lateral erosion.

    Construction:
        LateralEroder(grid, ***** Unknown so far.....)
    """
#***from how to make a component: every component must start with init
# first parameter is grid and **kwds is last parameter (allows to pass dictionary)
# in between, need individual parameters.
    def __init__(self, grid, vol_lat, input_stream, laterotype=None, alph=0.8, Kv=0.005, Klr=1.0, dt=10.):
        #**4/4/2019: come back to this: why did I put the underscore in from of grid? because diffusion said so.
        self._grid = grid
        self.laterotype = laterotype
        self.dt = dt
        self.initialize(grid, input_stream)
    # Create fields needed for this component if not already existing
#    if 'vol_lat' in grid.at_node:
#        self.vol_lat = grid.at_node['vol_lat']
#    else:
#        self.vol_lat = grid.add_zeros('node', 'vol_lat')
    
        self.laterotype=laterotype
        #you can specify the type of lateral erosion model you want to use. But if you don't
        # the default is the undercutting-slump model
        if laterotype is None:
            laterotype='UC'
        if laterotype=='UC':
            UC=1
            TB=0
        if laterotype=='TB':
            TB=1
            UC=0
    def initialize(self, grid, input_stream):

        # Create a ModelParameterDictionary for the inputs
        if type(input_stream)==ModelParameterDictionary:
            inputs = input_stream
        else:
            inputs = ModelParameterDictionary(input_stream)

        # Read input/configuration parameters
        self.alph = inputs.get('ALPH', ptype=float)
        self.Kv = inputs.get('KV_COEFFICIENT', ptype=float)
        self.Klr = inputs.get('KL_RATIO', ptype=float)
        self.rainrate = inputs.get('rain_rate', ptype=float)
        self.inlet_node = inputs.get('INLET_NODE', ptype=float)
        self.inlet_area = inputs.get('INLET_AREA', ptype=float)
        self.qsinlet = inputs.get('QSINLET', ptype=float)
        self.frac = 0.3 #for time step calculations

        # Set up state variables

        #initialize qsin for each interior node, all zero initially.
        self.qsin = grid.zeros(centering='node')    # qsin (M^3/Y)
#        self.qt = grid.zeros(centering='node')    # transport capacity
        self.dzdt = grid.zeros(centering='node')    # elevation change rate (M/Y)

    def run_one_step(self, grid,  vol_lat,  dt, qsinlet=None, inlet_area=None, Klr=None):

#        if rainrate==None:
#            rainrate = self.rainrate

        inlet_node=self.inlet_node
        #inlet_area=self.inlet_area
        if qsinlet==None:
            qsinlet=self.qsinlet
        if inlet_area==None:
            inlet_area=self.inlet_area
        if Klr==None:    #Added10/9 to allow changing rainrate (indirectly this way.)
            Klr=self.Klr

        laterotype=self.laterotype
                # the default is the undercutting-slump model
        if laterotype is None:
            laterotype='UC'
        if laterotype=='UC':
            UC=1
            TB=0
        if laterotype=='TB':
            TB=1
            UC=0
#        print("UC", UC)
#        print("TB", TB)
        Kv=self.Kv
        #Klr=self.Klr
        frac = self.frac
        qsin=self.qsin
        dzdt=self.dzdt
        alph=self.alph
        dt=self.dt
        
#        #you can specify the type of lateral erosion model you want to use. But if you don't
#        # the default is the undercutting-slump model
#        if laterotype is None:
#            laterotype='UC'
#        if laterotype=='UC':
#            UC=1
#            TB=0
#        if laterotype=='TB':
#            TB=1
#            UC=0
        #**********ADDED FOR WATER DEPTH CHANGE***************
        #now KL/KV ratio is a parameter set from infile again.
        #still need to calculate runoff for Q and water depth calculation
        kw=10.
        F=0.02
        #May 2, runoff calculated below (in m/s) is important for calculating
        #discharge and water depth correctly. renamed runoffms to prevent
        #confusion with other uses of runoff
        runoffms=(Klr*F/kw)**2

        #Kl is calculated from ratio of lateral to vertical K parameters
        Kl=Kv*Klr
        
        z=grid.at_node['topographic__elevation']
#        print("z",z)
        dx=grid.dx
        nr=grid.number_of_node_rows
        nc=grid.number_of_node_columns
        interior_nodes = grid.core_nodes
        boundary_nodes=structured_grid.boundary_nodes((nr,nc))
        #clear qsin for next loop
        qsin = grid.zeros(centering='node')
        lat_nodes=np.zeros(grid.number_of_nodes, dtype=int)
        dzlat=np.zeros(grid.number_of_nodes)
        dzver=np.zeros(grid.number_of_nodes)
        vol_lat_dt=np.zeros(grid.number_of_nodes)

        # 4/24/2017 add inlet to change drainage area with spatially variable runoff rate
        #runoff is an array with values of the area of each node (dx**2)
        #****4/5/2019: getting rid of inlet node for now.
#        runoffinlet=np.ones(grid.number_of_nodes)*dx**2
        #Change the runoff at the inlet node to node area + inlet node
#        runoffinlet[inlet_node]=+inlet_area
#        _=grid.add_field('node', 'water__unit_flux_in', runoffinlet,
#                     noclobber=False)




        #flow__upstream_node_order is node array contianing downstream to upstream order list of node ids
        s=grid.at_node['flow__upstream_node_order']
        da=grid.at_node['drainage_area']    #renamed this drainage area set by flow router
        max_slopes=grid.at_node['topographic__steepest_slope']
        flowdirs=grid.at_node['flow__receiver_node']
        #see below for clue of how to get inlet set again. probably have to look at old codes
#        drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
        if(0):
            print("LL da", da.reshape(nr,nc))
#            print("LL q", q.reshape(nr,nc))
        if (0):
            print('nodeIDs', grid.core_nodes)
            print ('flowupstream order', s)
            print("status at node[s]", grid.status_at_node[s])
            print ('runoffms', runoffms)
            print('flowdirs', flowdirs)

#            print(delta)
#        max_slopes2=grid.calc_grad_at_active_link(z)
#        max_slopes=grid.calc_slope_at_node()
#        print 'lengthflowdirs', len(flowdirs)
        #temporary hack for drainage area
#        drain_area[inlet_node]=inlet_area

        #order interior nodes
        #find interior nodes in downstream ordered vector s

        #make a list l, where node status is interior (signified by label 0) in s
        l=s[np.where((grid.status_at_node[s] == 0))[0]]

        dwnst_nodes=l
        #reverse list so we go from upstream to down stream
        #4/20/2017: this works
#        print "dwnst_nodes before reversal", dwnst_nodes
        dwnst_nodes=dwnst_nodes[::-1]
#        print("dwnst_nodes after reversal", dwnst_nodes)
#        print(delta)
        #local time
        time=0
        globdt=dt



#        print("time", time)
#        print("dt", dt)
        while time < globdt:
            #Calculate incision rate, should be in m/yr, should be negative
            #First make sure that there are no negative (uphill slopes)
            #Set those to zero, because incision rate should be zero there.
#            print 'slopes', max_slopes
            max_slopes=max_slopes.clip(0)
#            print 'slopes', max_slopes
#            print 'slope2', max_slopes2
#            print len(max_slopes)
#            print len(max_slopes2)
#            print delta
            #here calculate dzdt for each node, with initial time step
            #print "dwnst_nodes", dwnst_nodes
            for i in dwnst_nodes:
                if i==inlet_node:
                    qsin[i]=qsinlet
                    #print "qsin[inlet]", qsin[i]
                    #print "inlet area", drain_area[i]

                #calc deposition and erosion
                #dzver is vertical erosion/deposition only
#                print ('i ', i)
#                print ('slope', max_slopes[i])
#                print ('area', da[i])
#                print('kv', Kv)

                dep = alph*qsin[i]/da[i]
                ero = -Kv * da[i]**(0.5)*max_slopes[i]
#                print( 'dep', dep)
#                print('ero', ero)
                dzver[i] =  dep + ero


                #calculate transport capacity
#                qt[i]=Kv*da[i]**(3./2.)*max_slopes[i]/alph

                #lateral erosion component
                #potential lateral erosion initially set to 0
                petlat=0.
                #water depth in meters, needed for lateral erosion calc
                wd=0.4*(da[i]*runoffms)**0.35
                if(0):
                    print('i', i)
                    print('da[i]', da[i])
                    print('da*runoffms', da[i]*runoffms)
                    print('wd', wd)


#                if Klr!= 0.0:
#                    print('warning, in latero loop')
#                    print ' '
#                    print 'petlat before', petlat
#                    print 'i', i
                #if node i flows downstream, continue. That is, if node i is the
                #first cell at the top of the drainage network, don't go into this
                # loop because in this case, node i won't have a "donor" node found
#                fa = FlowAccumulator(grid, 'topographic__elevation')
#                headwaters=fa.headwater_nodes()
#                print("headwaters", headwaters)
                # in NodeFinder and needed to calculate the angle difference
                if i in flowdirs:

                    if(0):
                        print("i", i)
                        print("flowdirs", flowdirs)
                    #Node_finder picks the lateral node to erode based on angle
                    # between segments between three nodes
                    [lat_node, inv_rad_curv]=Node_Finder2(grid, i, flowdirs, da)
#                        print("lateroline 263")
#                    print ("lat_node", lat_node)
                        
                    #node_finder returns the lateral node ID and the radius of curvature
                    lat_nodes[i]=lat_node

                    #if the lateral node is not 0 continue. lateral node may be
                    # 0 if a boundary node was chosen as a lateral node. then
                    # radius of curavature is also 0 so there is no lateral erosion
                    if lat_node!=0:
                        #if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                        if z[lat_node] > z[i]:
                            petlat=-Kl*da[i]*max_slopes[i]*inv_rad_curv

                            #bank height.
                            #z_bank=z[lat_node]-z[i]

                            #the calculated potential lateral erosion is mutiplied by the length of the node
                            #and the bank height, then added to an array, vol_lat_dt, for volume eroded
                            #laterally  *per year* at each node. This vol_lat_dt is reset to zero for
                            #each timestep loop. vol_lat_dt is added to itself more than one primary nodes are
                            # laterally eroding this lat_node
                            vol_lat_dt[lat_node]+=abs(petlat)*dx*wd
                    if (0):
                        print("i ", i)
                        print('lat_node', lat_node)
#                       print 'petlat after', petlat
#                       print ' '
                # the following is always done, even if lat_node is 0 or lat_node
                # lower than primary node. however, petlat is 0 in these cases

                #send sediment downstream. sediment eroded from vertical incision
                # and lateral erosion is sent downstream
                qsin[flowdirs[i]]+=qsin[i]-(dzver[i]*dx**2)-(petlat*dx*wd)   #qsin to next node

                if (0):
                    #if drain_area[i] > 500: i==inlet_node:
                    print('node id', i)
#                    print "flowdirs[i]", flowdirs[i]
#                    print "drain_area[i]", drain_area[i]
#                    print "slope[i]", max_slopes[i]
#                    print "qsin", qsin[i]
#                    print "qsin downstream", qsin[flowdirs[i]]
#                    print "dzver", dzver[i]
#                    print 'petlat after', petlat
#                    print 'wd', wd
#                    print 'q[i]', q[i]
#                    print " "
#            print(delta)
            if(0):
                print("dzlat", dzlat)
#                print "dzdt", dzdt

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
                    print("node", i)
                #are points converging? ie, downstream eroding slower than upstream
                dzdtdif = dzdt[flowdirs[i]]-dzdt[i]
                #if points converging, find time to zero slope
                if dzdtdif > 0. and max_slopes[i] > 1e-7:
                    dtflat = (z[i]-z[flowdirs[i]])/dzdtdif	#time to flat between points
                    #if time to flat is smaller than dt, take the lower value
                    #april9, 2019: *** HACK WITH ABS(DTN) COME BACK to this. was getting negative dtn values
                    if abs(dtflat) < dtn:
                        dtn = abs(dtflat)
#                        print("dtflat", dtflat)
                    #if dzdtdif*dtflat will make upstream lower than downstream, find time to flat
                    if dzdtdif*dtn > abs((z[i]-z[flowdirs[i]])):
                        dtn=abs((z[i]-z[flowdirs[i]])/dzdtdif)
#                        print("timetoflat", dtn)


            #print "out of ts loop"
#            print("dtn",dtn)
#            print("dt",dt)
            dtn*=frac
            #new minimum timestep for this round of nodes
            dt=min(dtn, dt)
            #should now have a stable timestep.
#            print("stable time step=", dt)
#            print delta

            #******Needed a stable timestep size to calculate volume eroded from lateral node for
            # each stable time step********************

            #vol_lat is the total volume eroded from the lateral nodes through
            # the entire model run. So vol_lat is itself plus vol_lat_dt (for current loop)
            # times stable timestep size
            if(0):
                print("vol_lat before", vol_lat)
#                print "dt", dt
            vol_lat += vol_lat_dt*dt
            if (0):
                print("vol_lat_dt", vol_lat_dt)
#                print "vol_lat after", vol_lat


            #this loop determines if enough lateral erosion has happened to change the height of the neighbor node.
#            if Klr != 0.0:
#                print('warning in lat ero, line 330')
#                print("lat_nodes", lat_nodes)
            for i in dwnst_nodes:
                lat_node=lat_nodes[i]
                wd=0.4*(da[i]*runoffms)**0.35
                if lat_node>0:    #greater than zero now bc inactive neighbors are value -1
#                        print("latero, line 372")
#                    print("[lat_node]", lat_node)
#                        print("z[lat_node]", z[lat_node])
                    if z[lat_node] > z[i]:
                        #vol_diff is the volume that must be eroded from lat_node so that its
                        # elevation is the same as node downstream of primary node
                        # UC model: this would represent undercutting (the water height at node i), slumping, and instant removal.
                        if UC==1:
#                                print("UC model")
                            voldiff=(z[i]+wd-z[flowdirs[i]])*dx**2
                        # TB model: entire lat node must be eroded before lateral erosion occurs
                        if TB==1:
#                                print("TB model")
                            voldiff=(z[lat_node]-z[flowdirs[i]])*dx**2
                        #if the total volume eroded from lat_node is greater than the volume
                        # needed to be removed to make node equal elevation,
                        # then instantaneously remove this height from lat node. already has timestep in it
                        if vol_lat[lat_node]>=voldiff:
                            dzlat[lat_node]=z[flowdirs[i]]-z[lat_node]-0.001
                            if(1):
                                print("chunk of lateral erosion occured")
#                                    print "node", lat_node
#                                    print "dzlat[lat_node]", dzlat[lat_node]
#                                    print "vol_lat[latnode]", vol_lat[lat_node]
#                                    print "vol_lat_dt", vol_lat_dt[lat_node]
#                                    print "z[lat]", z[lat_node]
#                                    print "z[i]", z[i]
#                                    print "z[flowdirs[i]]", z[flowdirs[i]]
#                                    print "volume diff", voldiff
#                                    print 'wd', wd
#                                    print "drain_area[i]", drain_area[i]
#                                    print 'wd', wd
#                                    print 'q[i]', q[i]
#                                    print 'qms', qms
#                                    print delta
                            #after the lateral node is eroded, reset its volume eroded to zero
                            vol_lat[lat_node]=0.0

            #multiply dzver(changed to dzdt above) by timestep size and combine with lateral erosion
            #dzlat, which is already a length for the chosen time step
            dz=dzdt*dt+dzlat
            if(0):
                print('dzlat[inlet_node]', dzlat[inlet_node])
#                print 'dzdt[inlet_node]', dzdt[inlet_node]
#                print 'dz[inlet_node]', dz[inlet_node]

            #change height of landscape
            z=dz+z
#            z[interior_nodes]=dz[interior_nodes]+z[interior_nodes]
            grid['node'][ 'topographic__elevation'] =  z
#            print("z", reshape(z,(nr,nc)))
#            print("dz", reshape(dz,(nr,nc)))
#            print("grid", reshape(grid['node'][ 'topographic__elevation'],(nr,nc)))
#            print(delta)
            #update elapsed time
            time=dt+time
#            print 'dz', dz
#            print ('time', time)

            #check to see that you are within 0.01% of the storm duration, if so done, if not continue

            if time > 0.9999*globdt:
                time = globdt
#                #recalculate flow directions for plotting
#                fa = FlowAccumulator(grid, 
#                                     surface='topographic__elevation',
#                                     flow_director='FlowDirectorD8',
#                                     runoff_rate=None,
#                                     depression_finder=None, routing='D8')
#                (da, q) = fa.accumulate_flow()
#                s=grid.at_node['flow__upstream_node_order']
#                drain_area_fr=grid.at_node['drainage_area']
#                max_slopes=grid.at_node['topographic__steepest_slope']
#                q=grid.at_node['surface_water__discharge']
#                flowdirs=grid.at_node['flow__receiver_node']
##                drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
#                #recalculate downstream order
#                dsind = np.where((s >= min(interior_nodes)) & (s <= max(interior_nodes)))
#                l=s[np.where((grid.status_at_node[s] == 0))[0]]
#                dwnst_nodes=l
#                dwnst_nodes=dwnst_nodes[::-1]
##                max_slopes=grid.calc_grad_at_active_link(z)
#                #temporary hack for drainage area
##                drain_area[inlet_node]=inlet_area
            else:
                dt = globdt - time
#                print("small time steps. dt=",dt )
#                print(delt)
                #recalculate flow directions
                fa = FlowAccumulator(grid, 
                                     surface='topographic__elevation',
                                     flow_director='FlowDirectorD8',
                                     runoff_rate=None,
                                     depression_finder=None, routing='D8')
                (da, q) = fa.accumulate_flow()
                s=grid.at_node['flow__upstream_node_order']
                drain_area_fr=grid.at_node['drainage_area']
                max_slopes=grid.at_node['topographic__steepest_slope']
                q=grid.at_node['surface_water__discharge']
                flowdirs=grid.at_node['flow__receiver_node']
#                drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
                #recalculate downstream order
                dsind = np.where((s >= min(interior_nodes)) & (s <= max(interior_nodes)))
                l=s[np.where((grid.status_at_node[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]
                #what did I do here with this slopes, grad at active links?
#                max_slopes=grid.calc_grad_at_active_link(z)
                #temporary hack for drainage area
#                drain_area[inlet_node]=inlet_area
                #clear qsin for next loop
                qsin = grid.zeros(centering='node')
                qt = grid.zeros(centering='node')
                lat_nodes=np.zeros(grid.number_of_nodes, dtype=int)
                dzlat=np.zeros(grid.number_of_nodes)
                vol_lat_dt=np.zeros(grid.number_of_nodes)
                dzver=np.zeros(grid.number_of_nodes)

        return grid, dzlat, qsin, dzdt
#        return z, qt, qsin, dzdt, dzlat, flowdirs, da, dwnst_nodes, max_slopes, dt