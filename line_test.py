import time
import imgbinary
import cv2 as cv
import multiprocessing
import socket
import h264decoder
import numpy as np

video_address = ('', 11111)
vsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
vsock.bind(video_address)
decoder = h264decoder.H264Decoder()
queue = multiprocessing.Queue()
tello_address = ('192.168.10.1', 8889)
local_address = ('', 9000)
video_address = ('', 11111)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(local_address)

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

def send(message):
    try:
        sock.sendto(message.encode(), tello_address)
        print("Sending message: " + message)
    except Exception as e:
        print("Error sending: " + str(e))

video_thread = multiprocessing.Process(target=_receive_video_thread, args=(vsock, queue, decoder, ), daemon=True)
video_thread.start()

send("command")
time.sleep(1)
send("streamon")
time.sleep(1)
while True:
    t = time.time()
    while queue.empty() is True:
        time.sleep(0.1)
    img = queue.get()
    queue.put(img)
    ret, binaryimg = imgbinary.get_line_pos(img)
    cv.imshow("line", binaryimg)
    cv.waitKey(10)