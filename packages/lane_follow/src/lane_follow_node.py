#!/usr/bin/env python3

import rospy
import cv2

from duckietown.dtros import DTROS, NodeType
from turbojpeg import TurboJPEG, TJPF_GRAY
from duckietown_msgs.msg import Twist2DStamped
from dt_apriltags import Detector
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo, Range
from image_geometry import PinholeCameraModel

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(0, 128, 161), (10, 225, 225)]
DUCKS_WALKING_MASK = [(0, 33, 124), (24, 255, 255)]
DEBUG = False
ENGLISH = False
DEFAULT_VELOCITY = 0.28
PARKING_1_3_DISTANCE = 0.1
PARKING_2_4_DISTANCE = 0.33

"""
  Template for lane follow code was taken from eclass "Lane Follow Package".
  Author: Justin Francis
  Link: https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
"""
class LaneFollowNode(DTROS):

  def __init__(self, node_name):
    self.stop = False  # true if it detected a stop line

    super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.parking_stall_number = rospy.get_param("~parking_stall_number")
    self.veh = rospy.get_param("~veh")

    self.desired_parking_tag_dist = PARKING_1_3_DISTANCE
    if self.parking_stall_number == 2 or self.parking_stall_number == 4:
      self.desired_parking_tag_dist = PARKING_2_4_DISTANCE

    self.parking_tag_dist = None

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

    self.jpeg = TurboJPEG()

    self.loginfo("Initialized")

    # PID Variables
    self.proportional = None
    if ENGLISH:
        self.offset = -240
    else:
        self.offset = 240
    self.velocity = DEFAULT_VELOCITY
    # self.P = 0.035
    # self.I = 0.008
    # self.D = -0.0025
    self.P = 0.04
    self.D = -0.004
    self.I = 0.008
    # adjust vars for celina's robot
    # if self.veh == "csc22905": 
    #   self.P = 0.049
    #   self.D = -0.004
    #   self.offset = 200

    self.twist = Twist2DStamped(v=self.velocity, omega=0)

    self.last_error = 0
    self.last_time = rospy.get_time()

    # ====== April tag variables ======
    self.last_message = None

    self.apriltag_legend = {
      # for the other room
      # TODO: delete later
      21: "duckwalk",
      94: "right",
      200: "straight",
      169: "left",

      # for csc 229, the main room
      48: "right",
      50: "left",
      56: "straight",
      163: "duckwalk",
      207: "parking 1",
      226: "parking 2",
      228: "parking 3",
      75: "parking 4",
      227: "parking entrance",  # traffic light sign
      38: "parking entrance"  # stop sign
    }

    self.last_detected_apriltag = None

    # Get static parameters    
    self.tag_size = 0.065
    self.rectify_alpha = 0.0

    # Initialize detector
    self.at_detector = Detector(
      searchpath = ['apriltags'],
      families = 'tag36h11',
      nthreads = 1,
      quad_decimate = 1.0,
      quad_sigma = 0.0,
      refine_edges = 1,
      decode_sharpening = 0.25,
      debug = 0
    )

    self.tof_sub = rospy.Subscriber(
        f"/{self.veh}/front_center_tof_driver_node/range",
        Range,
        self.tof_callback,
        queue_size=1,
        buff_size="20MB"
      )

    self.tof_range = None

    # Initialize static parameters from camera info message
    camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)
    self.camera_model = PinholeCameraModel()
    self.camera_model.fromCameraInfo(camera_info_msg)
    H, W = camera_info_msg.height, camera_info_msg.width

    # find optimal rectified pinhole camera
    rect_K, _ = cv2.getOptimalNewCameraMatrix(
      self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
    )

    # store new camera parameters
    self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])

    self._mapx, self._mapy = cv2.initUndistortRectifyMap(
      self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
    )

    # Apriltag detection timer
    self.apriltag_hz = 2
    self.last_message = None
    self.timer = rospy.Timer(rospy.Duration(1 / self.apriltag_hz), self.cb_apriltag_timer)

    self.stop_duration = 3  # stop for 3 seconds
    self.stop_starttime = 0
    self.stop_cooldown = 3.5
    self.stop_threshold_area = 5000 # minimun area of red to stop at
    self.last_stop_time = None

    # Turn & action variables
    self.next_action = None
    self.left_turn_duration = 3.7
    self.right_turn_duration = 1
    self.straight_duration = 4
    self.started_action = None
    self.parking_turn_duration = 1
    self.start_parking = False
    self.start_parking_time = None
    self.parking_duration = 1.3

    # Wait a little while before sending motor commands
    rospy.Rate(0.20).sleep()

    # Shutdown hook
    rospy.on_shutdown(self.hook)

  def callback(self, msg):
    # message for the april tag
    self.last_message = msg

    img = self.jpeg.decode(msg.data)
    crop = img[300:-1, :, :]
    crop_width = crop.shape[1]
    self.frame_center = crop_width/2
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
        # print("THIS IS THE original PROP: ", self.proportional)
        if DEBUG:
          cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.proportional = None

    # STOP LINE HANDLING

    # See if we need to look for stop lines
    if self.stop or (self.last_stop_time and rospy.get_time() - self.last_stop_time < self.stop_cooldown):
      if DEBUG:
        rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
        self.pub.publish(rect_img_msg)
      return
    
    # Mask for stop lines
    stopMask = cv2.inRange(hsv, STOP_LINE_MASK[0], STOP_LINE_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=stopMask)
    stopContours, _ = cv2.findContours(
      stopMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    max_area = self.stop_threshold_area
    max_idx = -1
    for i in range(len(stopContours)):
      area = cv2.contourArea(stopContours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(stopContours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.stop = True
        self.stop_starttime = rospy.get_time()
        if DEBUG:
          cv2.drawContours(crop, stopContours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.stop = False

    if DEBUG:
      rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
      self.pub.publish(rect_img_msg)

  def cb_apriltag_timer(self, msg):
    msg = self.last_message

    if not msg:
      return

    self.last_detected_apriltag = None
    # turn image message into grayscale image
    img = self.jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
    # run input image through the rectification map
    img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)

    # detect tags
    tags = self.at_detector.detect(img, True, self._camera_parameters, self.tag_size)

    # Only save the april tag if it's within a close distance
    min_tag_distance = 0.6

    # Maximum distance from the parking stall april tags
    max_parking_distance = 0.095  # using tof sensor

    for tag in tags:
      self.at_distance = tag.pose_t[2][0]
      self.x_dist = tag.center[0]
      self.drift = self.frame_center - self.x_dist
      rospy.loginfo(self.at_distance)
      # self.proportional = self.drift

      # get apriltag if it is close
      if self.at_distance > min_tag_distance:
        continue
      
      self.last_detected_apriltag = tag.tag_id

      # if we're parking, stop the robot once it gets close the tag
      if (self.last_detected_apriltag == 207 or self.last_detected_apriltag == 226 or
          self.last_detected_apriltag == 228 or self.last_detected_apriltag == 75):
        if self.at_distance <= max_parking_distance:
          self.stop = True
      
      # rospy.loginfo("last detected: ")
      rospy.loginfo(self.last_detected_apriltag)

    

  def tof_callback(self, msg):
    if not msg:
      return
    # rospy.loginfo("================")
    # rospy.loginfo("min range: ")
    # rospy.loginfo(msg.min_range)
    # rospy.loginfo("max range:")
    # rospy.loginfo(msg.max_range)
    # rospy.loginfo("range")
    # rospy.loginfo(msg.range)

    self.tof_range = msg.range


  def drive(self):
    # Determine next action, if we haven't already
    # Get available action from last detected april tag
    if self.next_action is None and self.last_detected_apriltag is not None:
      self.next_action = self.apriltag_legend[self.last_detected_apriltag]

    if self.stop:
      if rospy.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        self.velocity = DEFAULT_VELOCITY
      else:
        # Do next action
        if self.next_action == "left":
          # Go left
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.left_turn_duration:
            self.twist.v = self.velocity
            self.twist.omega = 2.5
            self.vel_pub.publish(self.twist)
          else:
            self.started_action = None
            self.next_action = None
        elif self.next_action == "right":
          # lane following defaults to right turn
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.right_turn_duration:
            self._compute_lane_follow_PID()  # lane follow defaults to right turn
          else:
            self.started_action = None
            self.next_action = None
            self.velocity = 0.25  # lower velocity so that we can see the next apriltag
        elif self.next_action == "straight":
          # Go straight
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.straight_duration:
            self.twist.v = self.velocity
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)
          else:
            self.started_action = None
            self.next_action = None
        elif self.next_action == "parking entrance":
          # print("saw the parking tag")
          if self.at_distance > self.desired_parking_tag_dist:
            # print("change prop")
            self.proportional = -self.drift
            # print("THIS IS THE new PROP: ", self.proportional)

            self._compute_lane_follow_PID()
          else:
            print("stopping")
            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

            
          # if self.tof_range > self.desired_parking_tag_dist:
          #   # go straight for a bit
          #   self.twist.v = self.velocity
          #   self.twist.omega = 0
          #   self.vel_pub.publish(self.twist)
          # else:
          #   if self.started_action == None:
          #     # start parking action
          #     self.started_action = rospy.get_time()
          #   elif rospy.get_time() - self.started_action < self.parking_turn_duration:

          #   # turn into parking stall
          #     if self.parking_stall_number == 1 or self.parking_stall_number == 2:
          #       self.twist.v = 0
          #       self.twist.omega = 3
          #       self.vel_pub.publish(self.twist)
          #     elif self.parking_stall_number == 3 or self.parking_stall_number == 4:
          #       self.twist.v = 0
          #       self.twist.omega = -3
          #       self.vel_pub.publish(self.twist)
          #   else:
          #   # done parking action
          #     self.started_action = None
          #     self.next_action = None
          #     self.start_parking = True

        elif self.next_action == "parking 1":
          rospy.loginfo("I've parked in stall 1!")
          # rospy.signal_shutdown("I've parked in stall 1!")
        elif self.next_action == "parking 2":
          rospy.loginfo("I've parked in stall 2!")
          # rospy.signal_shutdown("I've parked in stall 2!")
        elif self.next_action == "parking 3":
          rospy.loginfo("I've parked in stall 3!")
          # rospy.signal_shutdown("I've parked in stall 3!")
        elif self.next_action == "parking 4":
          rospy.loginfo("I've parked in stall 4!")
          # rospy.signal_shutdown("I've parked in stall 4!")
        else:
          self.stop = False
          self.last_stop_time = rospy.get_time()
    else: # do lane following
      if self.start_parking:
        if self.tof_range > 0.1:
          self.twist.v = self.velocity
          self.twist.omega = 0
          self.vel_pub.publish(self.twist)

      # Determine Omega - based on lane-following
      if self.proportional is None:
        self.twist.omega = 0
        self.last_error = 0
      else:
        self._compute_lane_follow_PID()

    # self.vel_pub.publish(self.twist)

  def _compute_lane_follow_PID(self):
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
    for _ in range(8):
      self.vel_pub.publish(self.twist)


if __name__ == "__main__":
  node = LaneFollowNode("lanefollow_node")
  rate = rospy.Rate(8)  # 8hz

  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()
