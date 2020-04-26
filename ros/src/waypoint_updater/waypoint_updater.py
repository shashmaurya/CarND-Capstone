#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of lookahead points, reduced form 200 to improve performance
MAX_DECEL = 0.5     # Deceleration limit for braking calculation

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        # Declare member variables
        self.base_lane = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_waypoints = None
        
        
        # Define subscribers for current pose and waypoints 
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Subscriber for /traffic_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Publisher for final waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        #rospy.spin()
        self.loop()
        
        
        
    def loop(self):
        ## Runs a loop to publish the waypoints while rospy is running
        
        #rate = rospy.Rate(50)
        rate = rospy.Rate(22)       # Rate redcued for performance reasons, just over 20
        
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # pose and base lane exist call function to publish waypoints
                self.publish_waypoints()
            rate.sleep()
        
        
        
    def get_closest_waypoint_idx(self):
        ## Gets the closest waypoint based on current pose
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Ensure the point is ahead
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Use hyperplane equation to determine the nearest point
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        
        # If dot product is positive next point is the closest one
        if val>0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        
        return closest_idx
    
    
   
    def publish_waypoints(self):
        ## Calls the function to generate the lane
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        
        
        
    def generate_lane(self):
        ## Generates the lane
        
        lane = Lane()   # Declare a new lane object
        
        # Start with base points between nearest and farthest point, based on lookahead
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        # If stop line wapoint is not initialized, or farther tha lookahed distance
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
            
        else:
            # Update the waypoints to incorporate deceleration 
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
    
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        ## Creates a new set of waypoints for slowdown
        
        temp = []       # Create a new list of waypoints
        for i, wp in enumerate(waypoints):
            
            p = Waypoint()
            p.pose = wp.pose
            
            # Stop with car center a little behind the stop line
            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2* MAX_DECEL * dist)
            
            # Stop the car when velocity nears zero
            if vel<1.0:
                vel = 0.0
            
            # Use speed limit value if the calculated velocity is higher
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp
    
    

    def pose_cb(self, msg):
        ## Callback function for '/current_pose'
        self.pose = msg     
        #pass

    def waypoints_cb(self, waypoints):
        ## Callback function for '/base_waypoints'        
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        
        #pass

    def traffic_cb(self, msg):
        ## Callback function for '/traffic_waypoint'
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        ## Not Implemented
        pass

    def get_waypoint_velocity(self, waypoint):
        ## Returns velocity at the arg waypoint
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        ## Set the arg velocity at a specific waypoint in arg lane
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        ## Returns sum of linear distance between two waypoints in a lane object
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        
        # Get the sum of distances of the itermediary waypoints between the two wps
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
