# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import robomaster
from robomaster import robot, flight
from numpy import *

from imgbinary import *

ROLL_PARAM_P = -0.3
YAW_FEEDBACK = -0.7
FORWARD_SPEED = 15
THROTTLE = 0

robomaster.config.LOCAL_IP_STR = "192.168.10.2"
tl_drone = robot.Drone()
tl_drone.initialize()
tl_flight = tl_drone.flight

tl_camera = tl_drone.camera

def send_ctrl_cmd(cmd):
    tl_drone.action_dispatcher.send_action(flight.FlightAction(cmd))

def send_rc_cmd(a, b, c, d):
    tl_flight.rc(a = a, b = b, c = c, d = d)

def out_limit(val, min, max):
    if val > max:
        val = max
    elif val < min:
        val = min
    return val


if __name__ == '__main__':
    send_ctrl_cmd('takeoff')
    time.sleep(8)
    tl_camera.start_video_stream(display=False)
    send_ctrl_cmd('downvision 1')
    time.sleep(3)

    while True:
        img = tl_camera.read_cv2_image()
        t = time.time()

        # 二值化处理，得到线的轨迹
        ret, imgbinary = get_line_pos(img)

        cv2.imshow("Drone", img)
        cv2.imshow("BIN", imgbinary)
        cv2.waitKey(1)

        # 60列 根据线的轨迹进行 YAW 前馈
        if ret[0][0] == 0:
            yaw_out = 0
        else:
            yaw_out = YAW_FEEDBACK * ret[0][1]

        # 160列 根据线的轨迹进行 ROLL P 控制
        roll_out = ROLL_PARAM_P * ret[2][1]

        # 判断前面和中间都没线后， rc 0 0 0 0， 下降
        if ret[0][0] == 0 and ret[2][0] == 0:
            send_rc_cmd(0, 0, 0, 0)
            send_ctrl_cmd('land')
            break

        # RC 指令 rc a    b    c    d
        #            横滚 俯仰 油门 偏航 
        #            P    固定 0    前馈

        roll_out = out_limit(roll_out, -20, 20) 
        yaw_out = out_limit(yaw_out, -40, 40) 

        send_rc_cmd(int(roll_out), int(FORWARD_SPEED), int(THROTTLE), int(yaw_out))
        print('%f, %d, %d'%((time.time() - t)*1000, roll_out, yaw_out))
        time.sleep(0.01)

cv2.destroyAllWindows()
tl_camera.stop_video_stream()
tl_drone.close()
