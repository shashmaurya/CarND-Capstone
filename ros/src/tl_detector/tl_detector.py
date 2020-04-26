#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np
import time

STATE_COUNT_THRESHOLD = 3      # Number of consecutive states that should be same for TL update 

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        
        # Decalre the member variables 
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None

        # Define the subscribers for Current pose and base waypoints
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        
        # Subscibers for traffic lights, and camera image
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        # Get config for traffic lights
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        # Define publisher for waypoint
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        # Intialize member functions inherited, and variables
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        ## Callback function for 'current_pose'
        self.pose = msg

    def waypoints_cb(self, waypoints):
        ## Callback function for 'base_waypoints'
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        ## Callback function for 'traffic_lights'
        self.lights = msg.lights

    def image_cb(self, msg):
        ## Callback function for 'image_color'
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        # Call function to process the image received and get the light waypoint and state
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        
        # Based on the previous state and its count, either increment or 
        # publish the pertinent waypoint
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:     # Threshold of 3 currently
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1



    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        
        ## Convert to 2D waypoints if not already done, using KDTree
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)        
        
        # Section to get the closest waypoint index
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]        
        
        # Point should be ahead
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Use hyperplane equation to determine if the closest index is ahead
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        
        # Update the closest wp index if dop product is positive
        if val>0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)       
        

        return closest_idx

        

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Return false if there is no image
        #if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False
        
        
        ## Save the images from camera for model training
        #light_ind = light.state
        #cur_img = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8") 
        #cur_time = int(time.time())
        #file_path = r"/home/workspace/CarND-Capstone/ros/src/tl_detector/training_data/" + str(cur_time) + "_" + str(light_ind) + ".png"
        #if self.has_image:
            #cv2.imwrite(file_path, cur_img)            #Uncomment to save training data

            
        ## Section to determine light state
        # Convert the message image to cv2 to bgr8 
        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Call classifer function to determine state
        #return self.light_classifier.get_classification(cv_image)
                
        return light.state
        
    

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        closest_light = None
        line_wp_idx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
             
            # Find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            
            for i, light in enumerate(self.lights):
                # Stop line index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                # Get nearest stop line waypoint
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
         
        # If closest light exists, call function to get its state
        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
        # If stop line index is not known, and/or the light state is not known, return corresponding indicators
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
