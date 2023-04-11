#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from duckietown_msgs.msg import BoolStamped, VehicleCorners


ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DUCKS_WALKING_MASK = [(0, 33, 124), (24, 255, 255)]
DEBUG = False
ENGLISH = False

class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        self.distance_sub = rospy.Subscriber("/csc22919/duckiebot_distance_node/distance", 
                                        Float32, 
                                        self.distance_callback, 
                                        queue_size=1)
        
        self.detection_sub = rospy.Subscriber("/csc22919/duckiebot_detection_node/detection", 
                                         BoolStamped, 
                                         self.detection_callback, 
                                         queue_size=1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -240
        else:
            self.offset = 240
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.04
        self.D = -0.004
        self.I = 0.008
        self.last_error = 0
        self.last_time = rospy.get_time()


        # stop variables
        self.stop = False
        self.stop_duck_area = 5000
        self.stop_duration = 3  # stop for 3 seconds
        self.stop_starttime = 0
        self.stop_cooldown = 3
        self.last_stop_time = None

        # duckie breakdown variables
        self.duckie_down = False
        
        
        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def distance_callback(self, msg):
        self.vehicle_distance = msg.data

    def detection_callback(self, msg):
        self.detecting = msg.data

    def callback(self, msg):
        self.img = self.jpeg.decode(msg.data)
        crop = self.img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)


        # Detect ducks

        max_duck_area = self.stop_duck_area
        max_duck_idx = -1

        yellow_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        #TODO: Add yellow HSV values

        yellow_mask = cv2.inRange(hsv, DUCKS_WALKING_MASK[0], DUCKS_WALKING_MASK[1])
        yellow_crop = cv2.bitwise_and(crop, crop, mask=mask)
        yellow_contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        
        for i in range(len(yellow_contours)):
            yellow_area = cv2.contourArea(contours[i])
            if yellow_area > max_area:
                max_duck_idx = i
                max_duck_area = area

        if max_duck_idx != -1:
            duck_M = cv2.moments(yellow_contours[max_duck_idx])
            try:
                cx = int(duck_M['m10'] / duck_M['m00'])
                cy = int(duck_M['m01'] / duck_M['m00'])
                self.stop = True
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

        else:
            self.stop = False

        if self.detecting == True and self.vehicle_distance < 0.50:
            self.duckie_down = True
            self.stop_starttime = rospy.get_time()

        else:
            self.duckie_down = False

    def drive(self):
        if self.stop:
            rospy.loginfo("LET DUCKIES CROSS SAFELY")
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

        elif self.duckie_down:
            rospy.loginfo("DUCKIE DOWN, I REPEAT DUCKIE DOWN")
            if rospy.get_time() - self.stop_starttime < self.stop_duration:
                # Stop
                self.twist.v = 0
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)
            

        else:
            rospy.loginfo("LANE FOLLOWING NOW")
            if self.proportional is None:
                self.twist.omega = 0
                self.last_error = 0
            else:
                # P Term
                P = -self.proportional * self.P

                # D Term
                d_time = (rospy.get_time() - self.last_time)
                d_error = (self.proportional - self.last_error) / d_time
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D

                # I Term
                I = -self.proportional * self.I * d_time

                self.twist.v = self.velocity
                self.twist.omega = P + I + D
                if DEBUG:
                    self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

            self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()

