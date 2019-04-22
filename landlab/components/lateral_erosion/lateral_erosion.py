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
from six import string_types
from landlab import ModelParameterDictionary
from landlab.components.flow_director import FlowDirectorD8
from landlab.components.flow_accum import FlowAccumulator
from landlab.components.flow_routing import DepressionFinderAndRouter
#from landlab import Component
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
    Laterally erode neighbor node through fluvial erosion.

    Landlab component that finds a neighbor node to laterally erode and calculates lateral erosion.

    Construction:
        LateralEroder(grid, latero_mech="UC", alph=0.8, Kv=None, Kl_ratio=1.0, inlet_node=None, inlet_area=None, qsinlet=None)
        
    Parameteters
    ------------
    grid : ModelGrid
        A Landlab grid object
    latero_mech : string, optional (defaults to UC)
        Lateral erosion algorithm, choices are "UC" for undercutting-slump model and "TB" for total block erosion
    alph : float, optional (defaults to 0.8)
        Parameter describing potential for deposition, dimensionless
    Kv : float, node array, or field name
        Bedrock erodibility in vertical direction, 1/years
    Kl_ratio : float, optional (defaults to 1.0)
        Ratio of lateral to vertical bedrock erodibility, dimensionless
    inlet_node : integer, optional
        Node location of inlet (source of water and sediment)
    inlet_area : float, optional
        Drainage area at inlet node, must be specified if inlet node is on, m^2
    
    """
#***from how to make a component: every component must start with init
# first parameter is grid and **kwds is last parameter (allows to pass dictionary)
# in between, need individual parameters.
    def __init__(self, grid, latero_mech="UC", alph=0.8, Kv=None, Kl_ratio=1.0, inlet_node=None, inlet_area=None, qsinlet=0.): #input_stream,
        #**4/4/2019: come back to this: why did I put the underscore in from of grid? because diffusion said so.
        self._grid = grid
#        self.initialize(grid, input_stream)
    # Create fields needed for this component if not already existing
        if 'volume__lateral_erosion' in grid.at_node:
            self.vol_lat = grid.at_node['volume__lateral_erosion']
        else:
            self.vol_lat = grid.add_zeros('node', 'volume__lateral_erosion')
        if 'qsin' in grid.at_node:
            self.qsin = grid.at_node['qsin']
        else:
            self.qsin = grid.add_zeros('node', 'qsin')

        #you can specify the type of lateral erosion model you want to use. But if you don't
        # the default is the undercutting-slump model
        assert latero_mech in ("UC", "TB")
        if latero_mech == "TB":
#            assert isinstance(self.grid, RasterModelGrid)
            self._TB = True
            self._UC = False
        else:
            self._UC = True
            self._TB = False

        self.alph=alph
        self.Kv=Kv    #can be overwritten with spatially variable 
        self.inlet_on=False    #will be overwritten below if inlet area is provided
        self.Klr=float(Kl_ratio)    #default ratio of Kv/Kl is 1. Can be overwritten
        self.frac = 0.3 #for time step calculations
        
##optional inputs
        if inlet_node is not None:
            self.inlet_node = inlet_node
            self.inlet_on=True
                #below, adding flag calling for Kv to be specified. as of April 15, this only works for
            #arrays and floats, NOT field names.
            if self.inlet_on and inlet_area is None:
                raise ValueError(
                    "Inlet area must be provided if inlet node is active. "
                    + "No inlet area was found."
                )
            #note on April 22, 2019: I need to do below in init function because have to route water 
        # correctly from the beginning if there is an inlet.
            self.inlet_area = inlet_area
            # 4/24/2017 add inlet to change drainage area with spatially variable runoff rate
            #runoff is an array with values of the area of each node (dx**2)
            runoffinlet=np.zeros(grid.number_of_nodes)*grid.dx**2
            #Change the runoff at the inlet node to node area + inlet node
            print("inletnode", inlet_node)
            runoffinlet[inlet_node]=+inlet_area
            _=grid.add_field('node', 'water__unit_flux_in', runoffinlet,
                             noclobber=False)
            print("waterflux", reshape(grid['node'][ 'water__unit_flux_in'],(grid.number_of_node_rows,grid.number_of_node_columns)))
            self.qsinlet = qsinlet
            print("qsinlet", qsinlet)

#        if inlet_area is not None:
#            self.inlet_area = inlet_area
#            # 4/24/2017 add inlet to change drainage area with spatially variable runoff rate
#            #runoff is an array with values of the area of each node (dx**2)
#            runoffinlet=np.zeros(grid.number_of_nodes)*grid.dx**2
#            #Change the runoff at the inlet node to node area + inlet node
#            print("inletnode", inlet_node)
#            runoffinlet[inlet_node]=+inlet_area
#            _=grid.add_field('node', 'water__unit_flux_in', runoffinlet,
#                             noclobber=False)
#            print("waterflux", reshape(grid['node'][ 'water__unit_flux_in'],(grid.number_of_node_rows,grid.number_of_node_columns)))

        #initialize qsin for each interior node, all zero initially.
#        self.qsin = grid.zeros(centering='node')    # qsin (M^3/Y)
        self.dzdt = grid.zeros(centering='node')    # elevation change rate (M/Y)

#below, adding flag calling for Kv to be specified. as of April 15, this only works for
        #arrays and floats, NOT field names.
        if self.Kv is None:
            raise ValueError(
                "K_sp must be set as a float, node array, or "
                + "field name. It was None."
            )
# handling Kv for floats (inwhich case it populates an array N_nodes long) or 
# for arrays of Kv. Checks that length of Kv array is good.
        if isinstance(Kv, (float, int)):
            self.Kv = np.ones(self.grid.number_of_nodes)*float(Kv)
        else:
            self.Kv = np.asarray(Kv, dtype=float)
            if len(self.Kv) != self.grid.number_of_nodes:
                raise TypeError("Supplied value of Kv is not n_nodes long")

    def run_one_step(self, grid, dt=None, Klr=None, inlet_area_ts=None, qsinlet_ts=None):

        if Klr==None:    #Added10/9 to allow changing rainrate (indirectly this way.)
            Klr=self.Klr

        #have to define it this way from teh self.uc defined in initialize.
        #anything defined in initialize, you have to redefine here.
        UC=self._UC
        TB=self._TB
        inlet_on=self.inlet_on    #this is a true/false flag
        Kv=self.Kv
        frac = self.frac
        qsin=self.qsin
        dzdt=self.dzdt
        alph=self.alph

        self.dt=dt
        vol_lat=self.grid.at_node['volume__lateral_erosion']

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
        #clear qsin for next loop
        qsin = grid.zeros(centering='node')
        lat_nodes=np.zeros(grid.number_of_nodes, dtype=int)
        dzlat=np.zeros(grid.number_of_nodes)
        dzver=np.zeros(grid.number_of_nodes)
        vol_lat_dt=np.zeros(grid.number_of_nodes)

        if inlet_on:
            #define inlet_node
            inlet_node=self.inlet_node
            print("inlet_node", inlet_node)
            if qsinlet_ts==None:
                qsinlet=self.qsinlet
                qsin[inlet_node]=qsinlet
                print("qsinlet normal", qsinlet)
                print("qsinlet", reshape(qsin,(grid.number_of_node_rows,grid.number_of_node_columns)))
                print(elta)
            if qsinlet_ts is not None:
                qsinlet=qsinlet_ts
                qsin[inlet_node]=qsinlet
#                print("qsinlet ts")
                
                
            #******START HERE NEXT TIME***#
            #why do I use both water unit flux in and surface water discharge?
            if inlet_area_ts==None:    
#                q=grid.at_node['surface_water__discharge']
#                da=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
                print("inletarea normal")
            if inlet_area_ts is not None:
                inlet_area=inlet_area_ts
                runoffinlet=np.ones(grid.number_of_nodes)*grid.dx**2
                #Change the runoff at the inlet node to node area + inlet node
                runoffinlet[inlet_node]=+inlet_area
                _=grid.add_field('node', 'water__unit_flux_in', runoffinlet,
                             noclobber=False)
                print("inletarea ts")
            q=grid.at_node['surface_water__discharge']
            da=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
            print("da", da.reshape(nr,nc))
            #also change qsinlet
#            qsin[inlet_node]=qsinlet
#            print("inlet on")
        else:
            da=grid.at_node['drainage_area']    #renamed this drainage area set by flow router
#        print("da", da.reshape(nr,nc))
#        print("qsin", qsin.reshape(nr,nc))
        #flow__upstream_node_order is node array contianing downstream to upstream order list of node ids
        s=grid.at_node['flow__upstream_node_order']

        max_slopes=grid.at_node['topographic__steepest_slope']
        flowdirs=grid.at_node['flow__receiver_node']
        if(0):
            print("LL da", da.reshape(nr,nc))
#            print("LL q", q.reshape(nr,nc))
        if (0):
            print('nodeIDs', grid.core_nodes)
            print ('flowupstream order', s)
            print("status at node[s]", grid.status_at_node[s])
            print ('runoffms', runoffms)
            print('flowdirs', flowdirs)

        #order interior nodes
        #find interior nodes in downstream ordered vector s

        #make a list l, where node status is interior (signified by label 0) in s
        l=s[np.where((grid.status_at_node[s] == 0))[0]]
        dwnst_nodes=l
        #reverse list so we go from upstream to down stream
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
            max_slopes=max_slopes.clip(0)
            #here calculate dzdt for each node, with initial time step
#            print("dwnst_nodes", dwnst_nodes)
            #vectorized dz works. I'm thinking on how to vectorize the node finder, but not working today
#            dzvec[dwnst_nodes] = -Kv[dwnst_nodes] * da[dwnst_nodes]**(0.5)*max_slopes[dwnst_nodes]
#            print("vector erosion")
#            print(dzvec)
            for i in dwnst_nodes:
                #calc deposition and erosion
                #dzver is vertical erosion/deposition only
#                print ('i ', i)
#                print ('slope', max_slopes[i])
#                print ('area', da[i])
#                print('kv', Kv)

                dep = alph*qsin[i]/da[i]
#                dep=0.
                ero = -Kv[i] * da[i]**(0.5)*max_slopes[i]
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

                #*****april 10. I used to have this flag so that I could run the code iwthout vertical incision
                # I took it away so that we are always running vertical incision.
#                if Klr!= 0.0:

                #if node i flows downstream, continue. That is, if node i is the
                #first cell at the top of the drainage network, don't go into this
                # loop because in this case, node i won't have a "donor" node found
                # ********CHECK ON THIS. I don't think that this explanation of above
                # gives me what i think it does. CHECKED. IT'S GOOD.
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
                    #node_finder returns the lateral node ID and the radius of curvature
                    lat_nodes[i]=lat_node

                    #if the lateral node is not 0 or -1 continue. lateral node may be
                    # 0 or -1 if a boundary node was chosen as a lateral node. then
                    # radius of curavature is also 0 so there is no lateral erosion
                    #**** 4/19: I think now boundary nodes are designated iwth a -1 flag!!!!!
                    if lat_node>0:
                        #if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                        if z[lat_node] > z[i]:
                            petlat=-Kl[i]*da[i]*max_slopes[i]*inv_rad_curv

                            #the calculated potential lateral erosion is mutiplied by the length of the node
                            #and the bank height, then added to an array, vol_lat_dt, for volume eroded
                            #laterally  *per timestep* at each node. This vol_lat_dt is reset to zero for
                            #each timestep loop. vol_lat_dt is added to itself more than one primary nodes are
                            # laterally eroding this lat_node
                            vol_lat_dt[lat_node]+=abs(petlat)*dx*wd    #volume of lateral erosion per timestep
                    if (0):
                        print("i ", i)
                        print('lat_node', lat_node)
                # the following is always done, even if lat_node is 0 or lat_node
                # lower than primary node. however, petlat is 0 in these cases

                #send sediment downstream. sediment eroded from vertical incision
                # and lateral erosion is sent downstream
                #***** CHECK ON THIS! what about time step? what units are qsin?
                # april 10. this has to be here so that deposition can happen. dep takes qsin
                qsin[flowdirs[i]]+=qsin[i]-(dzver[i]*dx**2)-(petlat*dx*wd)   #qsin to next node
#            print("qsinafter", qsin.reshape(nr,nc))
            dzdt=dzver
#            print("original erosion")
#            print(dzver)
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
            tsdb=0
            if tsdb:
                print("start ts loop")
            for i in dwnst_nodes:
                if(tsdb):
                    print("node", i)
                    print("flowdirs[i]", flowdirs[i])
                    print("dzdt[i]", dzdt[i])
                    print("dzdt[flowdirs[i]]", dzdt[flowdirs[i]])
                #are points converging? ie, downstream eroding slower than upstream
                dzdtdif = dzdt[flowdirs[i]]-dzdt[i]
                if(tsdb):
                    print("dzdtdif", dzdtdif)
                #if points converging, find time to zero slope
                if dzdtdif > 1.e-5 and max_slopes[i] > 1e-5:
                    dtflat = (z[i]-z[flowdirs[i]])/dzdtdif	#time to flat between points
                    if(tsdb):
                        print("z[i]", z[i])
                        print("z[flowdirs[i]", z[flowdirs[i]])
                        print("max_slopes", max_slopes[i])
                        print("dtflat", dtflat)
                    #if time to flat is smaller than dt, take the lower value
                    #april9, 2019: *** HACK WITH ABS(DTN) COME BACK to this. was getting negative dtn values
                    #april 15, 2019: actually had negative dtflat values because of tiny slopes. hacked to 
                    # have abs(dtflat)
                    if dtflat < dtn:
                        dtn = dtflat
#                        assert dtn>0, "dtn <0 at dtflat"
                        if(tsdb):
                            print("dtflat<dtn", dtn)
                    #if dzdtdif*dtflat will make upstream lower than downstream, find time to flat
                    if dzdtdif*dtflat > (z[i]-z[flowdirs[i]]):
                        if(tsdb):
                            print("dzdtdif*dtn", dzdtdif*dtn)
                            print("dzdtdif*dtflat", dzdtdif*dtflat)
                            print("(z[i]-z[flowdirs[i]])", (z[i]-z[flowdirs[i]]))
                        dtn=(z[i]-z[flowdirs[i]])/dzdtdif
#                        assert dtn>0, "dtn <0 at dtflat"
                        if(tsdb):
                            print("t2flat", dtn)
            if(tsdb):
                print ("out of ts loop")
                print("dtn",dtn)
                print("dtn*frac", dtn*frac)
                print("dt",dt)
            dtn*=frac
            #new minimum timestep for this round of nodes
            dt=min(abs(dtn), dt)
            assert dt>0., "timesteps less than 0."
            #should now have a stable timestep.
#            print("stable time step=", dt)
#            print delta

            #******Needed a stable timestep size to calculate volume eroded from lateral node for
            # each stable time step********************

            #vol_lat is the total volume eroded from the lateral nodes through
            # the entire model run. So vol_lat is itself plus vol_lat_dt (for current loop)
            # times stable timestep size
            if(0):
                print("vol_lat before", vol_lat.reshape(nr,nc))
#                print "dt", dt
            vol_lat += vol_lat_dt*dt
            if (0):
                print("vol_lat_dt", vol_lat_dt.reshape(nr,nc))
                print("vol_lat after", vol_lat.reshape(nr,nc))

            debug=0
            #this loop determines if enough lateral erosion has happened to change the height of the neighbor node.
            wdnode=np.zeros(grid.number_of_nodes)
            wdnode[dwnst_nodes]=0.4*(da[dwnst_nodes]*runoffms)**0.35
            if(debug):
                print("lat_nodes", lat_nodes.reshape(nr,nc))
                print("nodesids", grid.nodes.reshape(nr,nc))
                print("elevs", z.reshape(nr,nc))
#            print(delt)

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
                            if(0):
                                print("chunk of lateral erosion occured", lat_node)
                            #after the lateral node is eroded, reset its volume eroded to zero
                            vol_lat[lat_node]=0.0
                            if(0):
                                print("vol_latafter", vol_lat.reshape(nr,nc))
            #multiply dzver(changed to dzdt above) by timestep size and combine with lateral erosion
            #dzlat, which is already a length for the chosen time step
            dz=dzdt*dt+dzlat

            #change height of landscape
            z=dz+z
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

            else:
                dt = globdt - time
                print("small time steps. dt=",dt )
                
                #clear qsin for next loop
                qsin = grid.zeros(centering='node')
                #recalculate flow directions
                fa = FlowAccumulator(grid, 
                                     surface='topographic__elevation',
                                     flow_director='FlowDirectorD8',
                                     runoff_rate=None,
                                     depression_finder="DepressionFinderAndRouter", router="D8")
                (da, q) = fa.accumulate_flow()
#                print("da", da.reshape(nr,nc))
#                print("q2", q.reshape(nr,nc))
                if inlet_on:
#                    q=grid.at_node['surface_water__discharge']
                    da=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
                    qsin[inlet_node]=qsinlet
#                    print("inlet on")
#                    print("da2", da.reshape(nr,nc))
#                    print("q2", q.reshape(nr,nc))
#                    print(delt)
                else:
                    da=grid.at_node['drainage_area']    #renamed this drainage area set by flow router
#                    print("da", da.reshape(nr,nc))
                da=grid.at_node['drainage_area']    #renamed this drainage area set by flow router
                s=grid.at_node['flow__upstream_node_order']
                max_slopes=grid.at_node['topographic__steepest_slope']
                q=grid.at_node['surface_water__discharge']
                flowdirs=grid.at_node['flow__receiver_node']
#                drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
                #recalculate downstream order
                l=s[np.where((grid.status_at_node[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]
                #what did I do here with this slopes, grad at active links?
#                max_slopes=grid.calc_grad_at_active_link(z)
                #temporary hack for drainage area
#                drain_area[inlet_node]=inlet_area
#               clear other things too.....
                lat_nodes=np.zeros(grid.number_of_nodes, dtype=int)
                dzlat=np.zeros(grid.number_of_nodes)
                vol_lat_dt=np.zeros(grid.number_of_nodes)
                dzver=np.zeros(grid.number_of_nodes)

        return grid, dzlat, qsin, dzdt
#        return z, qt, qsin, dzdt, dzlat, flowdirs, da, dwnst_nodes, max_slopes, dt