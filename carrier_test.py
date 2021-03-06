#  -*- coding: UTF-8 -*-

# MindPlus
# Python
import numpy as np

# 自定义函数
def forward(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('forward '+str(a)));time.sleep(3)
def led(r, g, b):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('EXT led '+str(r)+' '+str(g)+' '+str(b)));time.sleep(1)
def back(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('back '+str(a)));time.sleep(3)
def moff():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('moff'));time.sleep(3)
def shutdown():
   cv2.destroyAllWindows(); tl_camera.stop_video_stream() ;tl_drone.close()
def jump_mark(x, y, z, yaw, speed, mid, mid2):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('jump '+str(x)+' '+str(y)+' '+str(z) +' '+str(yaw)+' '+str(speed)+' m'+str(mid)+' m'+str(mid2)));time.sleep(8)
def go_mark(x, y, z, speed, mid):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('go '+str(x)+' '+str(y)+' '+str(z) +' '+str(speed)+' m'+str(mid)));time.sleep(5)
def mon():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('mon'));time.sleep(3)
def land():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('land'));time.sleep(5)
def send_rc(a, b, c, d):
  tl_drone.flight.rc(a = a, b = b, c = c, d = d)
def downvision():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('downvision 1'));time.sleep(1)
def delay(secs):
  time.sleep(secs)
def down(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('down '+str(a)));time.sleep(3)
def left(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('left  '+str(a)));time.sleep(3)
def right(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('right '+str(a)));time.sleep(3)
def up(a):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('up '+str(a)));time.sleep(3)
def takeoff():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('takeoff'));time.sleep(10)
def grab(roll_offset, pitch_offset, threshold, down_speed_scale):
  tl_drone.action_dispatcher.send_action(flight.FlightAction('downvision 1')) ; time.sleep(4)  ;cnt = 0
  while True:
    img = tl_camera.read_cv2_image() ;  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);  ret, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV) ;     img_binary = cv2.erode(img_binary, None, iterations=1) ; cv2.imshow('bin', img_binary)  ;  cv2.waitKey(1)
    color = [] ;white_thre = 10 ;   white_center = 0 ;  white_center_y = 0 ;  width = 0
    for i in range(30, 291, 2):
      color.append(img_binary[20:220, i])
    for i in range(0, 130):
      white_sum = np.sum(color[i] == 255) ;white_index = np.where(color[i] == 255)
      if  white_sum > white_thre:
        white_thre = white_sum ;  width = white_index[0][white_sum - 1] - white_index[0][0];  white_center = (white_index[0][white_sum - 1] + white_index[0][0]) / 2 - roll_offset;   white_center_y = i * 2 - pitch_offset
    if white_thre <= 5:
      tl_drone.action_dispatcher.send_action(flight.FlightAction('down 20')) ; time.sleep(4)
    roll = int(-0.3 * white_center) ; picth = int(-0.3 * white_center_y)
    if abs(roll) < 4 and abs(picth) < 4:
      cnt += down_speed_scale; tl_drone.flight.rc(int(-0.3 * white_center), int(-0.3 * white_center_y), int(-cnt), 0) ;print("send:", int(-0.3 * white_center), int(-0.3 * white_center_y), int(-cnt))
      if int(cnt) == 40:
        tl_drone.action_dispatcher.send_action(flight.FlightAction('land')) ;time.sleep(8); tl_drone.flight.rc(0, 0, 0, 0);   time.sleep(1) ;tl_drone.action_dispatcher.send_action(flight.FlightAction('motoron'));  time.sleep(3) ;  tl_drone.action_dispatcher.send_action(flight.FlightAction('takeoff')); time.sleep(10);
        break
    else:
      tl_drone.flight.rc(int(-0.3 * white_center), int(-0.3 * white_center_y),0, 0) ; print("send:", int(-0.3 * white_center), int(-0.3 * white_center_y), 0)
    time.sleep(0.01)
def throw():
  tl_drone.action_dispatcher.send_action(flight.FlightAction('EXT DIY throw'));time.sleep(6)


import time, cv2, robomaster
from robomaster import robot, flight
robomaster.config.LOCAL_IP_STR = "192.168.10.2"; tl_drone = robot.Drone(); tl_drone.initialize(); tl_flight = tl_drone.flight ; tl_camera = tl_drone.camera
tl_camera.start_video_stream(display=True)
takeoff()
down(20)
grab(100, 100, 20, 1)
forward(100)
delay(3)
throw()
land()
shutdown()
