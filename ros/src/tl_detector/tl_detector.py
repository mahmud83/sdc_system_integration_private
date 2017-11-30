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
import time
import math

STATE_COUNT_THRESHOLD = 3
# If the next variable is True, the imported TLClassifier object won't be used.
# Instead, the ground truth traffic light information from the `/vehicle/traffic_lights` topic will be used,
# i.e. the car will have perfect knowledge of each upcoming light and its state.
# This is useful for debugging other features.
PERFECT_TL_DETECTION = True

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.ego_pose = None
        self.closest_waypoint = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions'] # List of positions as `(x, y)` coordinates that correspond to the line to stop in front of for a given intersection.
        self.stop_line_waypoints = None # List containing the index of the closest waypoint for each traffic light stop line on the map.

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Unless we know the waypoints and where we are located, there is no point in starting.
        while (self.waypoints is None) or (self.ego_pose is None):
            time.sleep(0.02)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # Get the closest waypoints to all traffic light stop lines.
        self.get_stop_line_waypoints()

        if PERFECT_TL_DETECTION:
            sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.light_stop_waypoint = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.ego_pose = msg.pose

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.sub2.unregister() # The waypoints are being published only once, so we will no longer need to subscribe to this node afterwards.

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        '''
        Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Arguments:
            msg (Image): image from car-mounted camera
        '''
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.light_stop_waypoint = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.light_stop_waypoint))
        self.state_count += 1

    def get_l2_distance(self, point1, point2):
        '''
        Computes the l2 distance between the two given points. The points can be either 2D or 3D, but both
        points must have the same dimensionality.
        '''
        if len(point1) != len(point2):
            raise ValueError("Both points must have the same dimensionality.")

        if len(point1) == 3:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
        elif len(point1) == 2:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        else:
            raise ValueError("The given points must be either 2D or 3D.")

    def get_closest_waypoint(self, point, begin=0, end=None):
        '''
        Returns the index of the waypoint with the smallest l2 distance to the given point,
        where the point can be either 2D or 3D.

        Arguments:
            point (tuple): Either a 2-tuple representing a point in 2D Cartesian coordinates,
                or a 3-tuple representing a point in 3D Cartesian coordinates. The point which
                the closest waypoint is to be found.
            begin (int, optional): The waypoint index at which to begin the search. In case information
                is available about the last closest waypoint to `point`, this argument can be used
                to narrow down the search for higher efficiency rather than naively searching over
                all existing waypoints. Defaults to 0.
            end (int, optional): The waypoint index at which to end the search. See `begin`.
                Defaults to `None`, in which case it will be set to the last waypoint.

        Returns:
            The index of the waypoint with the smallest l2 distance to the given point.
        '''
        if not len(point) in {2, 3}:
            raise ValueError("The given point must be either 2D or 3D.")

        closest_waypoint = None
        closest_waypoint_dist = float('inf')

        if end is None:
            end = len(self.waypoints.waypoints)

        if end < begin: # Wrap around after the last waypoint.
            for i in range(begin, len(self.waypoints.waypoints)):
                wp_x = self.waypoints.waypoints[i].pose.pose.position.x
                wp_y = self.waypoints.waypoints[i].pose.pose.position.y
                wp_z = self.waypoints.waypoints[i].pose.pose.position.z
                if len(point) == 3:
                    dist = self.get_l2_distance((wp_x, wp_y, wp_z), point)
                else:
                    dist = self.get_l2_distance((wp_x, wp_y), point)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
            for i in range(0, end):
                wp_x = self.waypoints.waypoints[i].pose.pose.position.x
                wp_y = self.waypoints.waypoints[i].pose.pose.position.y
                wp_z = self.waypoints.waypoints[i].pose.pose.position.z
                if len(point) == 3:
                    dist = self.get_l2_distance((wp_x, wp_y, wp_z), point)
                else:
                    dist = self.get_l2_distance((wp_x, wp_y), point)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
        else:
            for i in range(begin, end):
                wp_x = self.waypoints.waypoints[i].pose.pose.position.x
                wp_y = self.waypoints.waypoints[i].pose.pose.position.y
                wp_z = self.waypoints.waypoints[i].pose.pose.position.z
                if len(point) == 3:
                    dist = self.get_l2_distance((wp_x, wp_y, wp_z), point)
                else:
                    dist = self.get_l2_distance((wp_x, wp_y), point)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist

        return closest_waypoint

    def get_stop_line_waypoints(self):
        '''
        Gets the closest waypoint for each traffic light stop line on the map.
        '''
        self.stop_line_waypoints = []

        for stop_line in self.stop_line_positions:
            self.stop_line_waypoints.append(self.get_closest_waypoint(stop_line))

        self.stop_line_waypoints.sort()

    def get_next_stop_line(self):
        '''
        Finds the next traffic light stop line ahead of the ego car's current position.

        Return:
            The index of the waypoint that is closest to the next upcoming stop line
            or `None` if there are no stop lines in `stop_line_waypoints`.
        '''
        if len(self.stop_line_waypoints) == 0:
            return None
        elif len(self.stop_line_waypoints) == 1:
            return self.stop_line_waypoints[0]
        else:
            # The following procedure works because the stop_line_waypoints list is sorted.
            for i in range(len(self.stop_line_waypoints)):
                if self.closest_waypoint < self.stop_line_waypoints[i]:
                    return self.stop_line_waypoints[i]
            # If we haven't returned at this point, then we're at the end of the
            # track loop (i.e. we've already passed the last stop line on the track),
            # which means the first stop line is the next stop line.
            return self.stop_line_waypoints[0]

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Get the waypoint that is closest to the ego car.
        car_x = self.ego_pose.position.x
        car_y = self.ego_pose.position.y
        car_z = self.ego_pose.position.z
        car_pos = (car_x, car_y, car_z)
        if not (self.closest_waypoint is None):
            self.closest_waypoint = self.get_closest_waypoint(point=car_pos, begin=self.closest_waypoint, end=((self.next_waypoint + 100) % len(self.waypoints.waypoints)))
        else:
            self.closest_waypoint = self.get_closest_waypoint(point=car_pos)
        # Check there is a stop line within a specified horizon of the ego car's currently closest waypoint.
        next_stop_line = self.get_next_stop_line()

        if next_stop_line is None: # If there are no traffic lights.
            return -1, TrafficLight.UNKNOWN
        elif PERFECT_TL_DETECTION:
            state = self.lights[self.stop_line_waypoints.index(next_stop_line)].state
            if state == TrafficLight.RED:
                return next_stop_line, state
            else:
                return -1, state
        else: # Check what color the next upcoming traffic light is.
            state = self.get_light_state()
            if state == TrafficLight.RED:
                return next_stop_line, state
            else:
                return -1, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
