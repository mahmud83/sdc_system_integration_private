#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from tf.transformations import euler_from_quaternion

import math
import time

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

LOOKAHEAD_WPS = 200 # The number of waypoints that will be published in each cycle.
BRAKE_PATH = 100 # How many waypoints ahead of a red light to start braking.

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.rate = rospy.Rate(50)

        self.ego_pose = None

        self.waypoints = None # This is the list of waypoints (as a `Lane` object) received from the `/base_waypoints` topic. We don't change those.
        self.final_waypoints = None # This is the list of waypoints to publish that we update every cycle.
        self.next_waypoint = None
        self.previous_next_waypoint = None
        self.next_stop_line = -1
        self.braking_initiated = False

        while (self.waypoints is None) or (self.ego_pose is None):
            time.sleep(0.05)

        # Publish the final waypoints.
        while not rospy.is_shutdown():
            # 1: Create the basic list of the next `LOOKAHEAD_WPS`-many waypoints.
            self.next_waypoint = self.get_next_waypoint()

            red_light_in_range = self.next_stop_line > 0
            if self.next_stop_line >= self.next_waypoint:
                red_light_in_range = red_light_in_range and (self.next_stop_line - self.next_waypoint) <= LOOKAHEAD_WPS
            else:
                red_light_distance = len(self.waypoints.waypoints) - self.next_waypoint + self.next_stop_line
                red_light_in_range = red_light_in_range and red_light_distance <= LOOKAHEAD_WPS

            if red_light_in_range and (not self.braking_initiated): # If we need to stop at a red light and realize that for the first time, i.e. if we don't have a braking trajectory yet.
                # Extract the relevant waypoints.
                if self.next_stop_line >= self.next_waypoint:
                    self.final_waypoints = self.waypoints.waypoints[self.next_waypoint:self.next_stop_line]
                else:
                    self.final_waypoints = self.waypoints.waypoints[self.next_waypoint:]
                    self.final_waypoints += self.waypoints.waypoints[0:self.next_stop_line]
                rospy.loginfo('next_stop_line: %s, next_waypoint: %s, len(self.final_waypoints): %s', self.next_stop_line, self.next_waypoint, len(self.final_waypoints))
                # Adjust the target velocity of the waypoints leading up to the stop line.
                current_target_vel = self.get_waypoint_velocity(self.waypoints.waypoints[self.next_waypoint])
                halt_buffer = 2 # For how many waypoints before the actual stop line the target speed should be set to zero.
                # Decrement velocity linearly. Not ideal, but let's stick with that for now.
                brake_path = min(BRAKE_PATH, len(self.final_waypoints))
                if brake_path < halt_buffer: halt_buffer = 0
                vel_decrement = current_target_vel / float(brake_path - halt_buffer)
                start_index = len(self.final_waypoints) - brake_path
                for i in range(start_index, start_index + brake_path - halt_buffer):
                    velocity = current_target_vel - (i + 1 - start_index) * vel_decrement
                    self.set_waypoint_velocity(self.final_waypoints, i, velocity)
                for i in range(len(self.final_waypoints) - halt_buffer, len(self.final_waypoints)):
                    velocity = 0.0
                    self.set_waypoint_velocity(self.final_waypoints, i, velocity)
                # Publish the waypoints.
                self.braking_initiated = True
                final_waypoints = Lane()
                final_waypoints.waypoints = self.final_waypoints
            elif red_light_in_range and self.braking_initiated: # If we've already started publishing the braking trajectory.
                # We already have a braking trajectory, just publish the part of it that still lies ahead of the car.
                num_waypoints_passed = self.next_waypoint - self.previous_next_waypoint
                self.final_waypoints = self.final_waypoints[num_waypoints_passed:]
                rospy.loginfo('next_stop_line: %s, next_waypoint: %s, len(self.final_waypoints): %s', self.next_stop_line, self.next_waypoint, len(self.final_waypoints))
                final_waypoints = Lane()
                final_waypoints.waypoints = self.final_waypoints
            else: # If self.next_stop_line == -1, there either is no traffic light or if there is one, then it is not red (anymore).
                self.braking_initiated = False
                final_waypoints = Lane()
                if self.next_waypoint + LOOKAHEAD_WPS <= len(self.waypoints.waypoints):
                    final_waypoints.waypoints = self.waypoints.waypoints[self.next_waypoint:self.next_waypoint+LOOKAHEAD_WPS]
                else: # Wrap around at the end of the track.
                    final_waypoints.waypoints += self.waypoints.waypoints[self.next_waypoint:len(self.waypoints.waypoints)]
                    num_waypoints_left_to_add = LOOKAHEAD_WPS - (len(self.waypoints.waypoints) - self.next_waypoint)
                    final_waypoints.waypoints += self.waypoints.waypoints[0:num_waypoints_left_to_add]
                rospy.loginfo('next_stop_line: %s, next_waypoint: %s, len(self.final_waypoints): %s', self.next_stop_line, self.next_waypoint, len(final_waypoints.waypoints))

            self.previous_next_waypoint = self.next_waypoint
            self.final_waypoints_pub.publish(final_waypoints)
            self.rate.sleep()

    def pose_cb(self, msg):
        self.ego_pose = msg.pose

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.next_stop_line = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def get_distance(self, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(self.waypoints.waypoints[wp1].pose.pose.position, self.waypoints.waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_l2_distance(self, wp):
        '''
        Computes the distance between the ego car and the given waypoint.
        '''
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        return dl(self.waypoints.waypoints[wp].pose.pose.position, self.ego_pose.position)

    def get_closest_waypoint(self, begin=0, end=None):
        '''
        Returns the waypoint that is closest to the ego car.
        '''
        closest_waypoint = None
        closest_waypoint_dist = 1000000
        if end is None:
            end = len(self.waypoints.waypoints)

        if end < begin: # Wrap around after the last waypoint.
            for i in range(begin, len(self.waypoints.waypoints)):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
            for i in range(0, end):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
        else:
            for i in range(begin, end):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
        return closest_waypoint

    def get_next_waypoint(self):
        if self.next_waypoint is None: # If this is the first time we're computing the next waypoint, we have to iterate over all waypoints.
            closest_waypoint = self.get_closest_waypoint()
        else:
            closest_waypoint = self.get_closest_waypoint(begin=self.next_waypoint, end=((self.next_waypoint + 100) % len(self.waypoints.waypoints)))
        # Check whether the closest waypoint is ahead of the ego car or behind it.
        if self.is_ahead(closest_waypoint):
            # If it is ahead, that's our guy.
            return closest_waypoint
        else:
            # If not, then the next waypoint after it must be our guy.
            if (closest_waypoint + 1) < len(self.waypoints.waypoints):
                return closest_waypoint + 1
            else: # Wrap around after the last waypoint.
                return 0

    def is_ahead(self, index):
        '''
        Returns `True` if the waypoint that `index` references lies ahead of the car and `False` otherwise.
        '''
        # Get the ego car's orientation quaternion...
        quaternion = (self.ego_pose.orientation.x,
                      self.ego_pose.orientation.y,
                      self.ego_pose.orientation.z,
                      self.ego_pose.orientation.w)
        # ...and compute the yaw from the quaternion.
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        # Compute the angle of the waypoint relative to the ego car's heading.
        dx = self.waypoints.waypoints[index].pose.pose.position.x - self.ego_pose.position.x
        dy = self.waypoints.waypoints[index].pose.pose.position.y - self.ego_pose.position.y

        waypoint_angle = math.atan2(dy, dx)

        diff_angle = abs(yaw - waypoint_angle)
        if diff_angle > math.pi / 4:
            return False
        else:
            return True

if __name__ == '__main__':
    try:
        waypoint_updater = WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
