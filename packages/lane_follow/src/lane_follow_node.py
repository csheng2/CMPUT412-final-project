#!/usr/bin/env python3

import rospy
import cv2

from duckietown.dtros import DTROS, NodeType
from turbojpeg import TurboJPEG, TJPF_GRAY
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from dt_apriltags import Detector
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from image_geometry import PinholeCameraModel
from std_msgs.msg import Float32

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(0, 128, 161), (10, 225, 225)]
DUCKS_WALKING_MASK = [(0, 33, 124), (24, 255, 255)]
DEBUG = False
ENGLISH = False
DEFAULT_VELOCITY = 0.28

"""
  Template for lane follow code was taken from eclass "Lane Follow Package".
  Author: Justin Francis
  Link: https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
"""
class LaneFollowNode(DTROS):

  def __init__(self, node_name):
    super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    self.stop = False  # true if it detected a stop line
    self.stop_duration = 3  # stop for 3 seconds
    self.stop_starttime = 0
    self.stop_cooldown = 3.5
    self.stop_threshold_area = 5000 # minimun area of red to stop at
    self.last_stop_time = None

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
    self.offset = 220
    self.velocity = DEFAULT_VELOCITY
    self.P = 0.035
    self.I = 0.008
    self.D = -0.0025

    # adjust vars for celina's robot
    if self.veh == "csc22905": 
      self.P = 0.049
      self.D = -0.004
      self.offset = 200

    self.twist = Twist2DStamped(v=self.velocity, omega=0)

    self.last_error = 0
    self.last_time = rospy.get_time()

    # STAGE 2 VARIABLES
    self.ducks_crossing = False
    self.check_duckie_down = False
    self.drive_around_bot = False
    self.drive_around_duration = 3.2
    self.stop_duck_area = 3000
    
    # Bot detection variables
    self.detecting_bot = False
    self.vehicle_distance = None

    self.distance_sub = rospy.Subscriber(f"/{self.veh}/duckiebot_distance_node/distance", 
                                    Float32, 
                                    self.distance_callback, 
                                    queue_size=1)
    
    self.detection_sub = rospy.Subscriber(f"/{self.veh}/duckiebot_detection_node/detection", 
                                      BoolStamped, 
                                      self.detection_callback, 
                                      queue_size=1)


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
      227: "parking entrance"
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

    # Turn & action variables
    self.next_action = None
    self.left_turn_duration = 3.7
    self.right_turn_duration = 1
    self.straight_duration = 4
    self.started_action = None

    # Wait a little while before sending motor commands
    rospy.Rate(0.20).sleep()

    # Shutdown hook
    rospy.on_shutdown(self.hook)


  def distance_callback(self, msg):
    self.vehicle_distance = msg.data

  def detection_callback(self, msg):
    self.detecting_bot = msg.data

  def callback(self, msg):
    # message for the april tag
    self.last_message = msg

    img = self.jpeg.decode(msg.data)
    crop = img[300:-1, :, :]
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


    # ==================================================================
    # DETECT DUCKS
    # ONLY IF WE'RE AT THE DUCKWALK APRIL TAG
    if ( self.last_detected_apriltag is not None
        and self.apriltag_legend[self.last_detected_apriltag] == "duckwalk"):
      max_duck_area = self.stop_duck_area
      max_duck_idx = -1
      duck_crop = img[: , 230:370, :]
      yellow_hsv = cv2.cvtColor(duck_crop, cv2.COLOR_BGR2HSV)
      yellow_mask = cv2.inRange(yellow_hsv, DUCKS_WALKING_MASK[0], DUCKS_WALKING_MASK[1])
      yellow_crop = cv2.bitwise_and(duck_crop, duck_crop, mask=yellow_mask)

      yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)

                                                                        
      for i in range(len(yellow_contours)):
        yellow_area = cv2.contourArea(yellow_contours[i])
        if yellow_area > max_duck_area:
          max_duck_idx = i
          max_duck_area = yellow_area

      if max_duck_idx != -1:
        duck_M = cv2.moments(yellow_contours[max_duck_idx])
        try:
          cx = int(duck_M['m10'] / duck_M['m00'])
          cy = int(duck_M['m01'] / duck_M['m00'])
          self.ducks_crossing = True

          if DEBUG:
            cv2.drawContours(yellow_crop, yellow_contours, max_duck_idx, (0, 255, 0), 3)
            cv2.circle(yellow_crop, (cx, cy), 7, (0, 0, 255), -1)
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(yellow_crop))
            self.pub.publish(rect_img_msg)
        except:
          pass
      else:
        self.ducks_crossing = False

    if (self.vehicle_distance is not None and self.detecting_bot == True
        and self.vehicle_distance < 0.5
        and not self.check_duckie_down):
      self.check_duckie_down = True
      self.stop_starttime = rospy.get_time()

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
    for tag in tags:
      distance = tag.pose_t[2][0]
      # get apriltag if it is close
      if distance > min_tag_distance:
        continue
      
      self.last_detected_apriltag = tag.tag_id

  def drive(self):
    # Determine next action, if we haven't already
    # Get available action from last detected april tag
    if self.next_action is None and self.last_detected_apriltag is not None:
      self.next_action = self.apriltag_legend[self.last_detected_apriltag]
      rospy.loginfo(self.next_action)


    if self.ducks_crossing:
      self.twist.v = 0
      self.twist.omega = 0
      self.vel_pub.publish(self.twist)

    elif self.check_duckie_down:
      if rospy.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
      else:
        self.check_duckie_down = False
        self.last_stop_time = rospy.get_time()
        self.offset *= -1
        self.drive_around_bot = True
        self.start_backup = None

    elif self.stop:
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
        else:
          self.stop = False
          self.last_stop_time = rospy.get_time()
    else: # do lane following
      # Determine Omega - based on lane-following
      if self.proportional is None:
        self.twist.omega = 0
        self.last_error = 0
      else:
        self._compute_lane_follow_PID()

    self.vel_pub.publish(self.twist)

  def _compute_lane_follow_PID(self):
    if self.drive_around_bot and rospy.get_time() - self.last_stop_time > self.drive_around_duration:
      self.drive_around_bot = False
      self.offset *= -1

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
