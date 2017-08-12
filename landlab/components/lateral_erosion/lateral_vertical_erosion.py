#! /usr/env/python
# -*- coding: utf-8 -*-
"""
Component for lateral and vertical incision. 

Abby Langston, October 2014-May 2017


"""

#import landlab
from landlab import ModelParameterDictionary
from landlab import ModelParameterDictionary, Component
from landlab.core.model_parameter_dictionary import MissingKeyError, \
    ParameterValueError
from landlab.utils.decorators import use_file_name_or_kwds
from landlab.field.scalar_data_fields import FieldError
from landlab import RasterModelGrid



from landlab.components.flow_routing import FlowRouter
#from landlab.components.lateral_erosion.node_finder2 import Node_Finder2
#from landlab.components.lateral_erosion.angle_finder import angle_finder
from landlab.components.lateral_erosion.straight_node_finder import StraightNode
#from landlab.components.lateral_erosion.fortyfive_node import FortyfiveNode
from landlab.components.lateral_erosion.ninety_node import NinetyNode
from landlab.utils import structured_grid
import numpy as np
np.set_printoptions(threshold=np.nan)
from random import uniform
import math


class LateralVerticalErosion(Component):
    '''


    The primary method of this class is :func:`run_one_step`.
    Runs the flow router from within this component. 

    Construction::

        LateralVerticalERosion(grid, lat_ero_type='UC', Kv_coef=1e-4, KlKv_ratio=1.0, alph=0.8, inlet_node=0,
                 inlet_area=0.0, qsinlet=0.0, **kwds)

    Parameters
    ----------
    grid : ModelGrid
        A grid.
    lat_ero_type: lateral erosion model formulation, either 'UC' for undercutting-
        slump or 'TB' for total block erosion. A lateral erosion type must be 
        specified. If no lateral erosion is desired, set KlKv_ratio = 0.0
    Kv_coef : float or array
        K in the stream power equation (units vary with other parameters).
    KlKv_ratio : float, optional
        ratio of lateral erodibility (Kl) to vertical erodibility (Kv_coef), 
        Tested ranges from 0.0 to 1.5. KlKv_ration value of 0.0 means no lateral
        erosion occurs, only vertical erosion.
    alph: float.
        v_s d*/R in Langston &Tucker 2017. Sediment mobility number where
        alph<1 is more mobile sediment, alph>1 less mobile sediment
        Performance will be VERY degraded if n < 1.
    inlet_node : float, optional.
        index of node that is designated as an inlet
    inlet_area : float, optional.
        drainage area (m^2) of inlet node. represents upstream catchment.
    qsinlet : float, array, optional.
        volumetric sediment flux at inlet node (m^3/yr). can be time varying
    '''
    #COPIED FROM DAN. COME BACK TO THIS!!!!*****************************
    _name = 'LateralVerticalErosion'

    _input_var_names = (
        'topographic__elevation',

    )

    _output_var_names = (
        'topographic__elevation',
    )

    _var_units = {
        'topographic__elevation': 'm',
        'drainage_area': 'm**2',
        'flow__link_to_receiver_node': '-',
        'flow__upstream_node_order': '-',
        'flow__receiver_node': '-',
    }

    _var_mapping = {
        'topographic__elevation': 'node',
        'drainage_area': 'node',
        'flow__link_to_receiver_node': 'node',
        'flow__upstream_node_order': 'node',
        'flow__receiver_node': 'node',
    }

    _var_doc = {
        'topographic__elevation': 'Land surface topographic elevation',
        'drainage_area':
            "Upstream accumulated surface area contributing to the node's "
            "discharge",
        'flow__link_to_receiver_node':
            'ID of link downstream of each node, which carries the discharge',
        'flow__upstream_node_order':
            'Node array containing downstream-to-upstream ordered list of '
            'node IDs',
        'flow__receiver_node':
            'Node array of receivers (node that receives flow from current '
            'node)',
    }
        
    # may 31, 2017: perhaps have to add vol_lat to this initialize section.
    #BELOW IS THE INITIALIZE SECTION   dt_years=10.0, model_form=''
    @use_file_name_or_kwds
    def __init__(self, grid, lat_ero_type='UC',  Kv_coef=1e-4, KlKv_ratio=1.0, alph=0.8, inlet_node=None,
                 inlet_area=None, qsinlet=None, **kwds):
        self._grid = grid


        # Read input/configuration parameters
#        self.dt=dt_years
        self.lat_ero_type=lat_ero_type
        assert self.lat_ero_type in ('UC', 'TB')
        self.alph = alph
        
        if type(Kv_coef) is np.ndarray:
            self.Kv=grid.add_zeros('node', 'bedrock__erodibility')
            self.Kv += Kv_coef
#            rg.at_node['topographic__elevation'][rg.nodes]=Kv_coef[rg.nodes]
            print 'Kv[100:107]', self.Kv[100:107]
            print 'Kv[2500:2510]', self.Kv[2500:2507]
#            self._kd = self.grid.at_node[linear_diffusivity]
#            print delta
        else:
            self.Kv=grid.add_zeros('node', 'bedrock__erodibility')
            self.Kv += Kv_coef
            print 'Kv[100:107]', self.Kv[100:107]
            print 'Kv[2500:2510]', self.Kv[2500:2507]
#            print delta
        self.Klr = KlKv_ratio

        if inlet_node is None:
            self.inlet_node=0
        else:
            self.inlet_node = inlet_node
        if inlet_area is None:
            self.inlet_area=0.0
        else:
            self.inlet_area = inlet_area
        if qsinlet is None:
            self.qsinlet=0.0
        else:
            self.qsinlet = qsinlet
        self.frac = 0.3 #for time step calculations

        # Set up state variables
        #kw and F are for calculating water depth
        self.kw=10.
        self.F=0.02
        #Kl is calculated from ratio of lateral to vertical K parameters
        self.Kl=self.Kv*KlKv_ratio
        print 'Kl[100:107]', self.Kl[100:107]
        print 'Kl[2500:2510]', self.Kl[2500:2507]
#        print delta
        self.vol_lat= grid.add_zeros('node', 'lateral_erosion__cumulative')
        #initialize qsin for each interior node, all zero initially.
        self.qsin = grid.zeros(centering='node')    # qsin (M^3/Y)
        self.dzdt = grid.zeros(centering='node')    # elevation change rate (M/Y)
        # initialize new fields in grid to store values on. never mind. See
        #tutorial that says can/should set input values in driver. never mind again
        # now I'm setting grid fields at the end
#        self.dzlat=grid.add_zeros['node','lateral_erosion__timestep']

        print 'inletnode', inlet_node
        print 'inletarea', inlet_area
        print 'alpha', self.alph
        print 'klr', self.Klr
#        print delta
# below: from Dan streampower line 278-281
#        else:
#            self.use_W = True
#            try:
#                self._W = self.grid.at_node[use_W]
                
                
                
                



    def erodelateral(self, dt, **kwds ):


        inlet_node=self.inlet_node
        qsinlet=self.qsinlet    #****HOW DOES VARIABLE QSIN AFFECT THESE?
        inlet_area=self.inlet_area

        Kv=self.Kv
        Klr=self.Klr
        Kl=self.Kl
        frac = self.frac
#        qsin=self.qsin
#        print 'max(qsin)', max(qsin)
        dzdt=self.dzdt
        alph=self.alph
        F=self.F
        kw=self.kw
        
        z=self.grid.at_node['topographic__elevation']
        grid=self.grid
        vol_lat=grid.at_node['lateral_erosion__cumulative']

        #May 2, runoff calculated below (in m/s) is important for calculating
        #discharge and water depth correctly. renamed runoffms to prevent
        #confusion with other uses of runoff
        runoffms=(Klr*F/kw)**2
        dx=grid.dx


        #clear qsin for next loop
        qsin = grid.zeros(centering='node')
        lat_nodes=np.zeros(grid.number_of_nodes)
        dzlat=np.zeros(grid.number_of_nodes)
        dzver=np.zeros(grid.number_of_nodes)
        vol_lat_dt=np.zeros(grid.number_of_nodes)
        #z_bank=np.zeros(grid.number_of_nodes)
    #vol_diff=np.zeros(grid.number_of_nodes)
        #instantiate variable of type RouteFlowDNClass
        flow_router = FlowRouter(grid)

        # 4/24/2017 add inlet to change drainage area with spatially variable runoff rate
        #runoff is an array with values of the area of each node (dx**2)
        runoffinlet=np.ones(grid.number_of_nodes)*dx**2
        #Change the runoff at the inlet node to node area + inlet node
        runoffinlet[inlet_node]=+inlet_area
        _=grid.add_field('node', 'water__unit_flux_in', runoffinlet,
                     noclobber=False)

        flow_router.route_flow(method='D8')
        #flow__upstream_node_order is node array contianing downstream to upstream order list of node ids
        s=grid.at_node['flow__upstream_node_order']
        drain_area_fr=grid.at_node['drainage_area']    #renamed this drainage area set by flow router
        max_slopes=grid.at_node['topographic__steepest_slope']
        q=grid.at_node['surface_water__discharge']
        flowdirs=grid.at_node['flow__receiver_node']
        drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff. 

        #make a list l, where node status is interior (signified by label 0) in s
        l=s[np.where((grid.status_at_node[s] == 0))[0]]
        dwnst_nodes=l
        dwnst_nodes=dwnst_nodes[::-1]

        #local time and maximum time step
        local_time=0
        dtmax=dt
        
        # Main loop where lateral and vertical erosion are calculated
        while local_time < dt:
            #Calculate incision rate, should be in m/yr, should be negative
            #First make sure that there are no negative (uphill slopes)
            #Set those to zero, because incision rate should be zero there.
#            print 'slopes', max_slopes
            max_slopes=max_slopes.clip(0)

            #here calculate dzdt for each node, with initial time step
            for i in dwnst_nodes:
                if i==inlet_node:
                    qsin[i]=qsinlet

                #calc deposition and erosion. dzver is vertical erosion/deposition only
                dep = alph*qsin[i]/drain_area[i]
                ero = -Kv[i] * drain_area[i]**(0.5)*max_slopes[i]
                dzver[i] =  dep + ero

                #lateral erosion component
                #potential lateral erosion initially set to 0
                petlat=0.                

                #may1, 2017. Need Q in m^3/s NOT M^3/yr!!!
                #water depth
#                qch=q[:]
#                qms=qch[i]/31536000.
#                wd=0.4*qms**0.35
                wd=0.4*(drain_area[i]*runoffms)**0.35
                if(0):
                    print 'i', i
                    print 'drain area[i]', drain_area[i]
                    print 'dr_area*runoffms', drain_area[i]*runoffms
                    print 'wd', wd
                
                # If Kl/Kv ration is not zero, go into this loop to calculate lateral erosion
                if Klr!= 0.0:
#                    print 'warning, in latero loop'

#                   # if node i flows downstream, continue. That is, if node i is the 
                    # first cell at the top of the drainage network, don't go into this
                    # loop because in this case, node i won't have a "donor" node found
                    # in NodeFinder and needed to calculate the angle difference
                    if i in flowdirs:
                    #Node_finder picks the lateral node to erode based on angle
                    # between segments between three nodes
                        [lat_node, inv_rad_curv]=self.Node_Finder(grid, i, flowdirs, drain_area)
                    #node_finder returns the lateral node ID and the radius of curvature
                        lat_nodes[i]=lat_node
                        
                    #if the lateral node is not 0 continue. lateral node may be 
                    # 0 if a boundary node was chosen as a lateral node. then 
                    # radius of curavature is also 0 so there is no lateral erosion
                        if lat_node!=0.0:
                        #if the elevation of the lateral node is higher than primary node,
                        # calculate a new potential lateral erosion (L/T), which is negative
                            if z[lat_node] > z[i]:                           
                                petlat=-Kl[lat_node]*drain_area[i]*max_slopes[i]*inv_rad_curv
                            
                            #bank height. 
                            #z_bank=z[lat_node]-z[i]
                            
                            #the calculated potential lateral erosion is mutiplied by the length of the node 
                            #and the bank height, then added to an array, vol_lat_dt, for volume eroded 
                            #laterally  *per year* at each node. This vol_lat_dt is reset to zero for 
                            #each timestep loop. vol_lat_dt is added to itself more than one primary nodes are
                            # laterally eroding this lat_node                       
                                vol_lat_dt[lat_node]+=abs(petlat)*dx*wd                           
                        if (0):
                            print "i ", i
                            print 'lat_node', lat_node
                            print 'petlat after', petlat
                            print ' '
                # the following is always done, even if lat_node is 0 or lat_node 
                # lower than primary node. however, petlat is 0 in these cases
                    
                #send sediment downstream. sediment eroded from vertical incision 
                # and lateral erosion is sent downstream                               
                qsin[flowdirs[i]]+=qsin[i]-(dzver[i]*dx**2)-(petlat*dx*wd)   #qsin to next node
           
                #qsqt[i]=qsin[i]/qt[i]
                     

                if (0):
                    #if drain_area[i] > 500: i==inlet_node:
                    print 'node id', i
                    print "flowdirs[i]", flowdirs[i]
                    print "drain_area[i]", drain_area[i]
                    print "slope[i]", max_slopes[i]
                    print "qsin", qsin[i]
                    print "qsin downstream", qsin[flowdirs[i]]
                    print "dzver", dzver[i]
                    print 'petlat after', petlat
                    print 'wd', wd
                    print 'q[i]', q[i]
#                    print 'qms', qms
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
                    dtflat = (z[i]-z[flowdirs[i]])/dzdtdif    #time to flat between points
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
#            print "stable time step=", dt
#            print delta
            
            #******Needed a stable timestep size to calculate volume eroded from lateral node for 
            # each stable time step********************
            
            #vol_lat is the total volume eroded from the lateral nodes through
            # the entire model run. So vol_lat is itself plus vol_lat_dt (for current loop)
            # times stable timestep size

            vol_lat += vol_lat_dt*dt

            #this loop determines if enough lateral erosion has happened to change the height of the neighbor node.
            if Klr != 0.0:
#                print 'warning in lat ero, line 330'
                for i in dwnst_nodes:
                    lat_node=lat_nodes[i]
                    wd=0.4*(drain_area[i]*runoffms)**0.35
                    if lat_node!=0.0:
                        if z[lat_node] > z[i]:
                            #voldiff is the volume that must be eroded from lat_node so that its 
                            # elevation is the same as primary node
                            # UC is undercutting-slump case the upper limit isn't the top of the node, but the water height at node i
                            # this would represent undercutting, slumping, and instant removal.
                            if self.lat_ero_type=='UC': 
                                voldiff=(z[i]+wd-z[flowdirs[i]])*dx**2
                            # TB is total block erosion case
                            elif self.lat_ero_type=='TB':
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
                                    print "z[lat]", z[lat_node]
                                    print "z[i]", z[i]
                                    print "z[flowdirs[i]]", z[flowdirs[i]]
                                    print "volume diff", voldiff
                                    print 'wd', wd
                                    print "drain_area[i]", drain_area[i]
                                    print 'wd', wd
                                    print 'q[i]', q[i]
#                                    print 'qms', qms
#                                    print delta
                                #after the lateral node is eroded, reset its volume eroded to zero
                                vol_lat[lat_node]=0.0
            
            #multiply dzver(changed to dzdt above) by timestep size and combine with lateral erosion
            #dzlat, which is already a length for the chosen time step
            dz=dzdt*dt+dzlat        
            if(0):
                print 'dzlat[inlet_node]', dzlat[inlet_node]
                print 'dzdt[inlet_node]', dzdt[inlet_node]
                print 'dz[inlet_node]', dz[inlet_node]
            
            #change height of landscape
            z[:]=dz+z[:]
#           # reset our field values with updated values of elevation and 
            # cumulative lateral erosion
            
#            self.grid.at_node['surface_water__depth'] = self.h
#            self.grid.at_link['surface_water__discharge'] = self.q

##***May 31, 2017: just name all of the fields down here. Best idea i have for now.
            grid['node'][ 'topographic__elevation'] =  z[:]
            grid.at_node['lateral_erosion__cumulative']=vol_lat
            grid.at_node['lateral_erosion__timestep']=dzlat
            grid.at_node['channel_sediment__volumetric_flux']=qsin
            grid.at_node['land_surface__elevation_change']=dzdt
            #update elapsed time
            local_time=dt+local_time

            #check to see that you are within 0.01% of the storm duration, if so done, if not continue

            if local_time > 0.9999*dtmax:
                time = dtmax
                #recalculate flow directions for plotting
#                flowdirs, drain_area, q, max_slopes, s, receiver_link = flow_router.route_flow(elevs=z, node_cell_area=node_area, runoff_rate=runoff)
                flow_router.route_flow(method='D8')
                s=grid.at_node['flow__upstream_node_order']
                drain_area_fr=grid.at_node['drainage_area']
                max_slopes=grid.at_node['topographic__steepest_slope']
                q=grid.at_node['surface_water__discharge']
                flowdirs=grid.at_node['flow__receiver_node']
                drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.                
                #recalculate downstream order
                l=s[np.where((grid.status_at_node[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]

            else:
                dt = dtmax - local_time
                #recalculate flow directions
#                flowdirs, drain_area, q, max_slopes, s, receiver_link = flow_router.route_flow(elevs=z, node_cell_area=node_area, runoff_rate=runoff)
                flow_router.route_flow(method='D8')
                s=grid.at_node['flow__upstream_node_order']
                drain_area_fr=grid.at_node['drainage_area']
                max_slopes=grid.at_node['topographic__steepest_slope']
                q=grid.at_node['surface_water__discharge']
                flowdirs=grid.at_node['flow__receiver_node']                
                drain_area=q/dx**2    #this is the drainage area that I need for code below with an inlet set by spatially varible runoff.
                #recalculate downstream order
                l=s[np.where((grid.status_at_node[s] == 0))[0]]
                dwnst_nodes=l
                dwnst_nodes=dwnst_nodes[::-1]

                #clear qsin for next loop
                qsin = grid.zeros(centering='node')
                lat_nodes=np.zeros(grid.number_of_nodes)
                dzlat=np.zeros(grid.number_of_nodes)
                vol_lat_dt=np.zeros(grid.number_of_nodes)
                dzver=np.zeros(grid.number_of_nodes)

        return self.grid
    
    def run_one_step(self, dt=1.0, **kwds ):
        self.erodelateral(dt, **kwds)
        
        
        
        
    # Define function Node_Finder that finds the lateral node to erode for each
    # primary node that has vertical incision (and an upstream and downstream stream link).
    def Node_Finder(self, grid, i, flowdirs, drain_area):
        # function to find the angle of two intersecting stream links
        def angle_finder(grid, dn, cn, rn):
    #        xcoord=grid.node_axis_coordinates(axis=0)
    #        ycoord=grid.node_axis_coordinates(axis=1)    #june 3
    #                x_coord=grid.node_x[3510]
    #        y_coord=grid.node_y[3510]
        
            sl1=(grid.node_y[cn]-grid.node_y[dn])/(grid.node_x[cn]-grid.node_x[dn])
            sl2=(grid.node_y[rn]-grid.node_y[cn])/(grid.node_x[rn]-grid.node_x[cn])
        
            angle1=math.degrees(math.atan(sl1))
            angle2=math.degrees(math.atan(sl2))
        
            angle_diff=angle2-angle1
            angle_diff=abs(angle2-angle1)
        
            return angle_diff
        
        def StraightNode(donor, i, receiver, neighbors, diag_neigh):
            debug=0
            print_debug=0
        
        #####FLOW LINK IS STRAIGHT, NORTH TO SOUTH######
            if ((donor==neighbors[1] or donor==neighbors[3])):
            
                #if (debug):
            #	print "flow is stright, N-S from ", donor, " to ", flowdirs[i]
                radcurv_angle=0.23
                if(print_debug):
                    print "erode node to east or west"
               #neighbors are ordered E,N,W, S
                #if the west cell is boundary (neighbors=-1), erode from east node
                if neighbors[2]==-1:
                    lat_node=neighbors[0]
                    if(print_debug):
                        print "eroding east node, id = ", lat_node
                elif neighbors[0]==-1:
                    lat_node=neighbors[2]
                    if(print_debug):
                        print "eroding west node, id = ", lat_node
        
                else:
                    #if could go either way, choose randomly. 0 goes East, 1 goes west
                    ran_num=np.random.randint(0,2)
                    if ran_num==0:
                        lat_node=neighbors[0]
                        if(print_debug):
                            print "eroding east node (random)", lat_node
                    if ran_num==1:
                        lat_node=neighbors[2]
                        if(print_debug):
                            print "eroding west node (random)", lat_node
        
            #####FLOW LINK IS STRAIGHT, EAST-WEST#####	
            elif (donor==neighbors[0] or donor==neighbors[2]):
                if (debug):
                    print "flow is stright, E-W"
                radcurv_angle=0.23
                if(print_debug):
                    print "erode node to north or south"
            #  Node list are ordered as [E,N,W,S]
            #if the north cell is boundary (neighbors=-1), erode from south node
                if neighbors[1]==-1:
                    lat_node=neighbors[3]
                    if(print_debug):  
                        print "eroding south node, id = ", lat_node
                elif neighbors[3]==-1:
                    lat_node=neighbors[1]
                    if(print_debug):
                        print "eroding north node, id = ", lat_node
                else:
            #if could go either way, choose randomly. 0 goes south, 1 goes north
                    ran_num=np.random.randint(0,2)
                    if ran_num==0:
                        lat_node=neighbors[1]
                        if(print_debug):
                            print "eroding north node (random), id = ", lat_node
                    if ran_num==1:
                        lat_node=neighbors[3]
                        if(print_debug):
                            print "eroding south node (random), id = ", lat_node
        
            #if flow is straight across diagonal, choose node to erode at random
            elif(donor in diag_neigh and receiver in diag_neigh):
                radcurv_angle=0.23
                if (debug):
                    print "flow is straight across diagonal"
                if receiver==diag_neigh[0]:
                    if(print_debug):
                        print "erode east or north"
                    poss_diag_nodes=neighbors[0:1+1]
                    if(print_debug):
                        print "poss_diag_nodes", poss_diag_nodes
                elif receiver==diag_neigh[1]:
                    if(print_debug):
                        print "erode north or west"
                    poss_diag_nodes=neighbors[1:2+1]
                    if(print_debug):
                        print "poss_diag_nodes", poss_diag_nodes
                elif receiver==diag_neigh[2]:
                    if(print_debug):
                        print "erode west or south"
                    poss_diag_nodes=neighbors[2:3+1]
                    if(print_debug):
                        print "poss_diag_nodes", poss_diag_nodes
                elif receiver==diag_neigh[3]:
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

        # Finds lateral node for a 45 degree bend. 
        def FortyfiveNode(donor, i, receiver, link_list, neighbors, diag_neigh):
            debug=0
            print_debug=0
            if (debug):
                print "flow from ", donor, " to ", receiver, " is 45 degrees"
            radcurv_angle=0.67
            if(print_debug):
                print "node is crossing diagonal"
            #OLD WAY: diagonal list goes [SE, SW, NW, NE]. Node list are ordered as [E,S,W,N]
            #LL 2017: diagonal list goes [NE, NW, SW, SE]. Node list are ordered as [E,N,W,S]				
            #if water flows SE-N OR if flow NE-S or E-NW or E-SW, erode west node
            if (donor==diag_neigh[0] and receiver==neighbors[3] or 
            donor==diag_neigh[3] and receiver==neighbors[1]
            or donor==neighbors[0] and receiver==diag_neigh[2] or
            donor==neighbors[0] and receiver==diag_neigh[1]):
                if(print_debug):
                    print "flow SE-N or NE-S or E-NW or E-SW, erode west node"
                lat_node=neighbors[2]
                if(print_debug):
                    print "lat_node", lat_node
            #if flow is from SW-N or NW-S or W-NE or W-SE, erode east node
            elif (donor==diag_neigh[1] and receiver==neighbors[3] or 
            donor==diag_neigh[2] and receiver==neighbors[1] or
            donor==neighbors[2] and receiver==diag_neigh[3] or
            donor==neighbors[2] and receiver==diag_neigh[0]):
                if(print_debug):
                    print "flow from SW-N or NW-S or W-NE or W-SE, erode east node"
                lat_node=neighbors[0]
                if(print_debug):
                    print "lat_node", lat_node
            #if flow is from SE-W or SW-E or S-NE or S-NW, erode north node
            elif (donor==diag_neigh[3] and receiver==neighbors[2] or 
            donor==diag_neigh[2] and receiver==neighbors[0] or 
            donor==neighbors[3] and receiver==diag_neigh[0] or
            donor==neighbors[3] and receiver==diag_neigh[1]):
                if(print_debug):
                    print "flow from SE-W or SW-E or S-NE or S-NW, erode north node"
                    
                lat_node=neighbors[1]
                if(print_debug):
                    print "lat_node", lat_node
            #if flow is from NE-W OR NW-E or N-SE or N-SW, erode south node
            elif (donor==diag_neigh[0] and receiver==neighbors[2] or 
            donor==diag_neigh[1] and receiver==neighbors[0] or
            donor==neighbors[1] and receiver==diag_neigh[3] or
            donor==neighbors[1] and receiver==diag_neigh[2]):
                if(print_debug):
                    print "flow from NE-W or NW-E or N-SE or N-SW, erode south node"
                lat_node=neighbors[3]
                if(print_debug):
                    print "lat_node", lat_node
        
            return lat_node, radcurv_angle
        
        # Finds lateral node for a 90 degree bend.
        def NinetyNode(donor, i, receiver, link_list, neighbors, diag_neigh):
            debug=0
            print_debug=0
            if (debug):
                print 'donor', donor
                print 'i', i
                print 'receiver', receiver
        
            #if flow is 90 degrees
            if(donor in diag_neigh and receiver in diag_neigh):
                if (debug):
                    print "flow is 90 degrees on diagonals from ", donor, " to ", receiver
                radcurv_angle=1.37
                #if flow is NE-SE or NW-SW, erode south node
                if (donor==diag_neigh[0] and receiver==diag_neigh[3] or 
                    donor==diag_neigh[1] and receiver==diag_neigh[2]):
                    lat_node=neighbors[3]
                #if flow is SW-NW or SE-NE, erode north node
                elif (donor==diag_neigh[2] and receiver==diag_neigh[1] or 
                      donor==diag_neigh[3] and receiver==diag_neigh[0]):
                    lat_node=neighbors[1]
                #if flow is SW-SE or NW-NE, erode east node
                elif (donor==diag_neigh[2] and receiver==diag_neigh[3] or 
                      donor==diag_neigh[1] and receiver==diag_neigh[0]):
                    lat_node=neighbors[0]
                 #if flow is SE-SW or NE-NW, erode west node
                elif (donor==diag_neigh[3] and receiver==diag_neigh[2] or 
                       donor==diag_neigh[0] and receiver==diag_neigh[1]):
                     lat_node=neighbors[2]
                 #print "lat_node", lat_node
            elif(donor not in diag_neigh and receiver not in diag_neigh):
                if (debug):
                    print "flow is 90 degrees (not on diagonal) from ", donor, " to ", receiver
                radcurv_angle=1.37
                
                #if flow is from east, erode west node
                if (donor==neighbors[0]):
                    lat_node=neighbors[2]
                #if flow is from north, erode south node
                elif (donor==neighbors[1]):
                    lat_node=neighbors[3]
                #if flow is from west, erode east node
                elif (donor==neighbors[2]):
                    lat_node=neighbors[0]
                #if flow is from south, erode north node
                elif (donor==neighbors[3]):
                    lat_node=neighbors[1]
            if(debug):
                print 'lat_node', lat_node
            return lat_node, radcurv_angle

        debug=0
        print_debug=0
    
        #receiver node of flow is flowdirs[i]
        receiver=flowdirs[i]
    
        #find indicies of where flowdirs=i to find donor nodes.
        #will donor nodes always equal the index of flowdir list?
        inflow=np.where(flowdirs==i)
        #if there are more than 1 donors, find the one with largest drainage area
        if len(inflow[0])>1:
            drin=drain_area[inflow]
            drmax=max(drin)
            maxinfl=inflow[0][np.where(drin==drmax)]
            if(debug):
                print "old inflow", inflow[0]
                print "max(drin)", max(drin)
                print "maxinfl", maxinfl
            #inind=np.where(drin==drmax)
            #if donor nodes have same drainage area, choose one randomly
            if len(maxinfl)>1:
                ran_num=np.random.randint(0,len(maxinfl))
                maxinfln=maxinfl[ran_num]
                donor=[maxinfln]
                if(debug):
                    print "random donor", donor
    
            else:
                donor=maxinfl
                if(debug):
                    print "donor with larger drainage area", donor
            #else donor is the only inflow
        else:
            donor=inflow[0]
    
        if(print_debug):
    
            print "donor", donor
            print "i", i
            print "receiver", receiver
        #now we have chosen donor cell, next figure out if inflow/outflow lines are
        #straight, 45, or 90 degree angle. and figure out which node to erode
    
        link_list=grid.links_at_node[i]
        neighbors=grid.active_neighbors_at_node[i]    #this gives list of active neighbors for specified node
        diag_neigh=grid._diagonal_neighbors_at_node[i]
#        linkdirs=grid.active_link_dirs_at_node[i]

#        print 'xcord', x_coord
#        print 'ycord', y_coord
#        print delta
        angle_diff=angle_finder(grid, donor, i, receiver)
#        angle_diff=self.angle_finder(grid, donor, i, receiver)

        if(debug):
            print "link_list", link_list
            print "neighbors", neighbors
            print "diagneighbors", diag_neigh
            print "angle_diff", angle_diff
            print " "
    
        if donor == flowdirs[i]:
            #this is a sink. no lateral ero
            if(debug):
                print "this is a sink"
            radcurv_angle=0.
            lat_node=0.
        if angle_diff==0.0:
            [lat_node, radcurv_angle]=StraightNode(donor, i, receiver, neighbors, diag_neigh)
        if (angle_diff==45.0 or angle_diff==135.0):
            [lat_node, radcurv_angle]=FortyfiveNode(donor, i, receiver, link_list, neighbors, diag_neigh)
        if angle_diff==90.0:
            [lat_node, radcurv_angle]=NinetyNode(donor, i, receiver, link_list, neighbors, diag_neigh)
    
        if lat_node > 2e9:
            #print "old latnode", lat_node
            lat_node=0.
            radcurv_angle=0.0
            #print "new lat", lat_node
            #print delta
        dx=grid.dx
        radcurv_angle=radcurv_angle/dx    #May24, 2017: this is actually INVERSE radius of curvature. It works out in the main lateral ero
        return lat_node, radcurv_angle