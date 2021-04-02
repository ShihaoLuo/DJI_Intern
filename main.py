import socket
import threading
import numpy as np
import multiprocessing
import h264decoder
import cv2 as cv
import time
import math
import imgbinary

tello_address = ('192.168.10.1', 8889)
local_address = ('', 9000)
video_address = ('', 11111)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(local_address)
vsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
vsock.bind(video_address)
decoder = h264decoder.H264Decoder()
queue = multiprocessing.Queue()
response = multiprocessing.Queue()
pose = multiprocessing.Queue()
color_flag = multiprocessing.Value('i', 9)
marker1_wpoints = np.array([[100, -100, 0], [-100, -100, 0], [-100, 100, 0], [100, 100, 0]], dtype=np.float64)
dataset = {"marker1":marker1_wpoints}


def show_pic(_queue):
    print("show pic thread start.\n")
    while True:
        a = cv.waitKey(1)
        if a == ord('q'):
            cv.destroyAllWindows()
            break
        if _queue.empty() is False:
            f = _queue.get()
            _queue.put(f)
            cv.imshow('', f)
        time.sleep(0.06)

def send(message):
    try:
        sock.sendto(message.encode(), tello_address)
        print("Sending message: " + message)
    except Exception as e:
        print("Error sending: " + str(e))

def receive(response):
    while True:
        try:
            res, ip_address = sock.recvfrom(128)
            res = res.decode(encoding='utf-8')
            print("Received message: " + res)
            while response.empty() is False:
                response.get()
            response.put(res)
        except Exception as e:
            sock.close()
            print("Error receiving: " + str(e))
            break

def _h264_decode(decoder, packet_data, queue):
    frames = decoder.decode(packet_data)
    for frame_data in frames:
        (frame, w, h, ls) = frame_data
        if frame is not None:
            frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
            frame = (frame.reshape((h, ls // 3, 3)))
            frame = frame[:, :w, :]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            img = cv.filter2D(frame, -1, kernel)
            img = cv.flip(img, 0)
            while queue.qsize() > 1:
                queue.get()
            queue.put(img)

def _receive_video_thread(_sock, _queue, decoder):
    pack_data = ''
    print("receive video thread start....")
    while True:
        try:
            res_string, ip = _sock.recvfrom(2048)
            pack_data += res_string.hex()
            if len(res_string) != 1460:
                tmp = bytes.fromhex(pack_data)
                _h264_decode(decoder, tmp, _queue)
                pack_data = ''
        except socket.error as exc:
            print("Caught exception socket.error(video_thread): %s" % exc)

def _marker_detecter(_queue, dataset, _pose):
    print("marker thread start.")
    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    parameter = cv.aruco.DetectorParameters_create()
    camera_matrix = np.load('camera_matrix_tello.npy')
    distor_matrix = np.load('distor_matrix_tello.npy')
    while True:
        time.sleep(0.1)
        if _queue.empty() is False:
            img = _queue.get()
            _queue.put(img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameter)
            if ids is not None:
                key = 'marker'+str(ids[0][0])
                objp = np.array(dataset[key])
                pixp = np.array(corners[0][0])
                ret, rvecs, tvecs = cv.solvePnP(objp, pixp,camera_matrix, distor_matrix)
                R = np.array(cv.Rodrigues(rvecs)[0])
                sy = math.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])
                singular = sy < 1e-6
                if not singular:
                    x = math.atan2(R[1, 2], R[2, 2])
                    y = math.atan2(-R[0, 2], sy)
                    z = math.atan2(math.sin(x) * R[2, 0] - math.cos(x) * R[1, 0],
                                   math.cos(x) * R[1, 1] - math.sin(x) * R[2, 1])
                else:
                    x = math.atan2(-R[1, 2], R[1, 1])
                    y = math.atan2(-R[2, 0], sy)
                    z = 0
                tmp = np.dot(np.linalg.inv(-R), tvecs)
                yaw = z * 180 / 3.1416
                if yaw<0:
                    yaw+=180
                else:
                    yaw-=180
                # print('pose', tmp)
                # print('yaw', yaw)
                while _pose.empty() is False:
                    _pose.get()
                _pose.put(np.append(tmp, yaw))
                # print(np.append(tmp, yaw))

def cnt_area(cnt):
    area = cv.contourArea(cnt)
    return area

def _color_detect(_queue, _color_flag):
    lowerrange = np.array([[156, 43, 46],
                  [26, 43, 46],
                  [35, 43, 46],
                  [100, 43, 46]])
    upperrange = np.array([[180, 255, 255],
                  [34, 255, 255],
                  [77, 255, 255],
                  [124, 255, 255]])
    while True:
        area = 20000
        color = None
        time.sleep(1)
        if _queue.empty() is False:
            img = _queue.get()
            _queue.put(img)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            for idx, i in enumerate(zip(upperrange, lowerrange)):
                mask = cv.inRange(hsv, i[1], i[0])
                cv.imshow('contours', mask)
                cv.waitKey(50)
                im, contours, hierarchy = cv.findContours(mask, 1, 2)
                contours.sort(key=cnt_area, reverse=True)
                try:
                    cnt = contours[0]
                    if cv.contourArea(cnt) > 8000 and cv.contourArea(cnt) > area:
                        color = idx
                        area = cv.contourArea(cnt)
                except IndexError as e:
                    print(e)
            if color is not None:
                _color_flag.value = color
            else:
                _color_flag.value = 9


def out_limit(val, min, max):
    if val > max:
        val = max
    elif val < min:
        val = min
    return val


def line_track(_queue):
    ROLL_PARAM_P = -0.3
    YAW_FEEDBACK = -0.7
    FORWARD_SPEED = 15
    THROTTLE = 0
    while True:
        t = time.time()
        while _queue.empty() is True:
            time.sleep(0.1)
        img = _queue.get()
        _queue.put(img)
        ret, binaryimg = imgbinary.get_line_pos(img)
        if ret[0][0] == 0:
            yaw_out = 0
        else:
            yaw_out = YAW_FEEDBACK * ret[0][1]

        # 160列 根据线的轨迹进行 ROLL P 控制
        roll_out = ROLL_PARAM_P * ret[2][1]

        # 判断前面和中间都没线后， rc 0 0 0 0， 下降
        if ret[0][0] == 0 and ret[2][0] == 0:
            send('rc 0 0 0 0')
            time.sleep(0.5)
            send('land')
            break
        roll_out = out_limit(roll_out, -20, 20)
        yaw_out = out_limit(yaw_out, -40, 40)

        send('rc'+str(int(roll_out))+str(int(FORWARD_SPEED))+str(int(THROTTLE))+str(int(yaw_out)))
        print('%f, %d, %d' % ((time.time() - t) * 1000, roll_out, yaw_out))
        time.sleep(0.01)


color_thread = multiprocessing.Process(target=_color_detect, args=(queue, color_flag, ), daemon=True)
color_thread.start()
marker_thread = multiprocessing.Process(target=_marker_detecter, args=(queue, dataset, pose, ), daemon=True)
marker_thread.start()
receiveThread = threading.Thread(target=receive, args=(response,))
receiveThread.daemon = True
receiveThread.start()
video_thread = multiprocessing.Process(target=_receive_video_thread, args=(vsock, queue, decoder, ), daemon=True)
video_thread.start()
show_thread = multiprocessing.Process(target=show_pic, args=(queue,), daemon=True)
show_thread.start()

def run():
    while True:
        send('command')
        time.sleep(0.5)
        if response.empty() is False:
            t = response.get()
            if t.upper() == 'OK':
                print("connected!")
                break
    while True:
        # send('battery?')
        # time.sleep(0.1)
        # send('EXT DIY hold')
        # time.sleep(0.1)
        # send('EXT led 0 0 255')
        # time.sleep(5)
        # send('EXT DIY throw')
        # time.sleep(0.1)
        # send('EXT DIY 255 0 0')
        # time.sleep(1)
        # send('EXT DIY 0 0 255')
        time.sleep(1)
        if pose.empty() is False:
            tmp = pose.get()
            print(tmp)
        if color_flag.value != 9:
            t = color_flag.value
            print('get color:', t)

if __name__=='__main__':
    run()