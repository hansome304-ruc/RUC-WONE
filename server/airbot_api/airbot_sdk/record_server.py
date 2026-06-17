import socket
import struct
import json
import pickle
import time
from io import BytesIO
import cv2
import numpy as np
import threading
import os
import argparse
from datetime import datetime
from airbot_sdk.utils.video_util import H264VideoWriter, DepthVideoWriter
import rerun as rr
import rerun.blueprint as rrb
import av
from rerun.blueprint import Horizontal, Vertical
import atexit
import signal
import mysql.connector
from multiprocessing import Process, Queue

def setup_output_stream(width: int, height: int) -> av.video.VideoStream:
    """Setup the output stream which encodes the video stream to H.264."""
    
    output_container = av.open("/dev/null", "w", format="h264")
    output_stream = output_container.add_stream("libx264")
    assert isinstance(output_stream, av.video.stream.VideoStream)
    output_stream.width = width
    output_stream.height = height

    output_stream.codec_context.options = {
        "tune": "zerolatency",
        "preset": "veryfast",
    }
    output_stream.max_b_frames = 0

    return output_stream


def images_consumer(images_queue: Queue, name="left"):
    while True:
        try:
            item = images_queue.get()
            if isinstance(item, str) and item == "exit":
                break
            image, date_time = item
            # image = cv2.resize(image, (128,96), interpolation=cv2.INTER_AREA)
            image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, buffer = cv2.imencode('.jpg', image, encode_param)
            if not success:
                raise ValueError("Could not encode image as JPEG")
            # Decode back to numpy array
            img_jpeg_np = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            image = img_jpeg_np
            rr.set_time("timestamp", timestamp=date_time)
            rr.log(f"images/{name}", rr.Image(image, color_model="bgr"))
        except Exception as e:
            print("images_consumer error", e)
            continue

def joints_consumer(joints_queue: Queue, arm_name="left_arm"):
    while True:
        try:
            data = joints_queue.get()
            if isinstance(data, str) and data == "exit":
                break
            
            data_time, joint_data = data
            cur_joint = joint_data[:6]
            cur_gripper = float(joint_data[-1])
            
            rr.set_time("timestamp", timestamp=data_time)
            for i in range(6):
                rr.log(f"{arm_name}/cur_joint/joint_{i+1}", rr.Scalars(joint_data[i]))
            
            rr.log(f"{arm_name}/cur_gripper", rr.Scalars(joint_data[-1]))
            
        except Exception as e:
            print(f"joints_consumer error for {arm_name}: {e}")
            continue

# @rr.shutdown_at_exit
def data_process(left_q, right_q, front_q, left_joints_q, right_joints_q) -> None:
    rr.init("realtime_aloha")
    # rr.connect_grpc("rerun+http://10.170.21.199:9876/proxy")
    rr.connect_grpc("rerun+http://localhost:9892/proxy")
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    *[
                        rrb.Spatial2DView(
                            name=image_name,
                            origin=f"images/{image_name}",
                        )
                        for image_name in ["left", "front", "right"]
                    ]
                )
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="left_arm", origin="left_arm/cur_joint"
                        ),
                        rrb.TimeSeriesView(
                            name="left_arm_gripper", origin="left_arm/cur_gripper"
                        ),
                        rrb.TimeSeriesView(
                            name="right_arm", origin="right_arm/cur_joint"
                        ),
                        rrb.TimeSeriesView(
                            name="right_arm_gripper", origin="right_arm/cur_gripper"
                        ),
                    )
                )
            ),
        ),
        rrb.TimePanel(expanded=True),
    ),)
    
    left_thread = threading.Thread(target=images_consumer, args=(left_q, "left"), daemon=True)
    left_thread.start()
    right_thread = threading.Thread(target=images_consumer, args=(right_q, "right"), daemon=True)
    right_thread.start()
    front_thread = threading.Thread(target=images_consumer, args=(front_q, "front"), daemon=True)
    front_thread.start()
    
    left_joints_thread = threading.Thread(target=joints_consumer, args=(left_joints_q, "left_arm"), daemon=True)
    left_joints_thread.start()
    right_joints_thread = threading.Thread(target=joints_consumer, args=(right_joints_q, "right_arm"), daemon=True)
    right_joints_thread.start()
    
    while True:
        time.sleep(0.01)

# 创建所有的queues
left_q = Queue()
right_q = Queue()
front_q = Queue()
left_joints_q = Queue()
right_joints_q = Queue()

# 启动数据处理进程
global p
p = Process(target=data_process, args=(left_q, right_q, front_q, left_joints_q, right_joints_q), daemon=True)
p.start()


global save_file_name
save_file_name = None
job_id = None
rr.init("record_server", spawn=False)

videos_origins = ["videos_left", "videos_right", "videos_front"]

# 在主进程中发送blueprint
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    *[
                        rrb.Spatial2DView(
                            name=videos_origin.split("_")[1],
                            origin=videos_origin,
                        )
                        for videos_origin in videos_origins
                    ]
                )
            ),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="left_arm", origin="left_arm/cur_joint"
                        ),
                        rrb.TimeSeriesView(
                            name="left_arm_gripper", origin="left_arm/cur_gripper"
                        ),
                        rrb.TimeSeriesView(
                            name="right_arm", origin="right_arm/cur_joint"
                        ),
                        rrb.TimeSeriesView(
                            name="right_arm_gripper", origin="right_arm/cur_gripper"
                        ),
                    )
                )
            ),
        ),
        rrb.TimePanel(expanded=True),
    ),
)

left_output_stream = setup_output_stream(640, 480)
right_output_stream = setup_output_stream(640, 480)
front_output_stream = setup_output_stream(640, 480)

rr.log("videos_left", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
rr.log("videos_right", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
rr.log("videos_front", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

HOST = '127.0.0.1'
PORT = 7788
BUFFER_SIZE = 4096
SOCKET_PATH_1 = '/tmp/socket_test_1.sock' # for left cur joint
SOCKET_PATH_2 = '/tmp/socket_test_2.sock' # for right cur joint
SOCKET_PATH_3 = '/tmp/socket_test_3.sock' # for left img
SOCKET_PATH_4 = '/tmp/socket_test_4.sock' # for right img
SOCKET_PATH_5 = '/tmp/socket_test_5.sock' # for front img
SOCKET_PATH_11 = '/tmp/socket_test_11.sock' # for check
SOCKET_PATH_12 = '/tmp/socket_test_12.sock' # for stop record cmd


TYPE_INT = b'0'
TYPE_DICT = b'1'
TYPE_IMAGE_LEFT = b'2'
TYPE_IMAGE_RIGHT = b'3'
TYPE_IMAGE_FRONT = b'4'

def send_data(sock, data_type, data_time, data_bytes):
    data_len = len(data_bytes)
    header = data_type + struct.pack('i', data_len) + data_time
    sock.sendall(header)
    sock.sendall(data_bytes)

global left_color_list
global right_color_list
global front_color_list
global left_color_ts_list
global right_color_ts_list
global front_color_ts_list

left_color_list = []
right_color_list = []
front_color_list = []
left_color_ts_list = []
right_color_ts_list = []
front_color_ts_list = []

shutdown_lock = threading.Lock()
shutdown_event = threading.Event()


def _notify_stop_ack():
    global save_file_name
    if save_file_name is None:
        return
    for _ in range(30):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                data_time = int(time.time()*1000)
                s.connect(SOCKET_PATH_11)
                data_bytes = save_file_name.encode('utf-8')
                time_bytes = data_time.to_bytes(length=8, byteorder='big')
                send_data(s, TYPE_DICT, time_bytes, data_bytes)
                return
        except Exception:
            time.sleep(0.1)
    print("notify stop ack failed")


def _shutdown_record_server(reason="stop_socket"):
    global left_color_list
    global left_color_ts_list
    global right_color_list
    global right_color_ts_list
    global front_color_list
    global front_color_ts_list
    global left_output_stream
    global right_output_stream
    global front_output_stream
    global job_id
    global p
    global save_file_name

    with shutdown_lock:
        if shutdown_event.is_set():
            return
        shutdown_event.set()

    print(f'Received stop request: {reason}, exiting gracefully.')
    left_color_list = []
    left_color_ts_list = []
    right_color_list = []
    right_color_ts_list = []
    front_color_list = []
    front_color_ts_list = []

    left_q.put('exit')
    right_q.put('exit')
    front_q.put('exit')
    left_joints_q.put('exit')
    right_joints_q.put('exit')
    print("queue clear")

    if save_file_name is not None:
        try:
            conn = mysql.connector.connect(
                db='hq_aloha_db',
                host='localhost',
                port=3306,
                user='hq_aloha',
                password='12345678'
            )
            print("sql connect done")
            cursor = conn.cursor()
            sql_cmd = "INSERT INTO uploads (file_path, object_key, is_upload, upload_error, job_id) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql_cmd, (save_file_name, '0', False, None, job_id))
            conn.commit()
            conn.close()
            print("sql insert done")
        except Exception as e:
            print(f"sql insert failed: {e}")

    print("save done")
    _notify_stop_ack()

    try:
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(f"kill child process failed: {e}")
    os._exit(0)

def main():
    global save_file_name
    global job_id
    parser = argparse.ArgumentParser()
    parser.add_argument("save_file")
    parser.add_argument("job_id", default="0", nargs='?')
    args = parser.parse_args()
    save_file_name = args.save_file
    job_id = args.job_id
    rr.set_sinks(rr.GrpcSink("rerun+http://localhost:9876/proxy"), rr.FileSink(save_file_name))
    while True:
        time.sleep(0.01)

def _record_func_1():
    global left_joints_q
    if os.path.exists(SOCKET_PATH_1):
        os.unlink(SOCKET_PATH_1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_1)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                header = conn.recv(13)
                if not header: # if client disconnect with server
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                data_bytes = b''
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                if data_type == TYPE_DICT:
                    data = json.loads(data_bytes.decode('utf-8'))
                    key_ = list(data.keys())[0]
                    if key_ == 'left_cur_joint':
                        val = data[key_]
                        cur_joint = val[:6]
                        cur_gripper = float(val[-1])
                        try:
                            left_joints_q.put((data_time, val))
                        except Exception as e:
                            print(f"Error putting left joint data to queue: {e}")
                        
                        rr.set_time("timestamp", timestamp=data_time)
                        for i in range(6):
                            rr.log("left_arm/cur_joint/joint_{}".format(i+1), rr.Scalars(val[i]))
                        rr.log("left_arm/cur_gripper", rr.Scalars(val[-1]))
        time.sleep(0.005)

def _record_func_2():
    global right_joints_q
    if os.path.exists(SOCKET_PATH_2):
        os.unlink(SOCKET_PATH_2)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_2)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                header = conn.recv(13)
                if not header: # if client disconnect with server
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                
                data_bytes = b''
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                if data_type == TYPE_DICT:
                    data = json.loads(data_bytes.decode('utf-8'))
                    key_ = list(data.keys())[0]
                    if key_ == 'right_cur_joint':
                        val = data[key_]
                        cur_joint = val[:6]
                        cur_gripper = float(val[-1])
                        try:
                            right_joints_q.put((data_time, val))
                        except Exception as e:
                            print(f"Error putting right joint data to queue: {e}")
                        rr.set_time("timestamp", timestamp=data_time)
                        for i in range(6):
                            rr.log("right_arm/cur_joint/joint_{}".format(i+1), rr.Scalars(val[i]))
                        rr.log("right_arm/cur_gripper", rr.Scalars(val[-1]))
        time.sleep(0.005)

def _record_func_3():
    global left_output_stream
    if os.path.exists(SOCKET_PATH_3):
        os.unlink(SOCKET_PATH_3)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_3)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                header = conn.recv(13)
                if not header: # if client disconnect with server
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                data_bytes = b''
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                # deal with different type of data
                if data_type == TYPE_IMAGE_LEFT:
                    img_data = BytesIO(data_bytes)
                    img = cv2.imdecode(np.frombuffer(img_data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    img_record = np.asanyarray(img)
                    left_q.put((img_record, data_time))
                    try:
                        av_frame = av.VideoFrame.from_ndarray(img_record, format="bgr24")
                        av_frame.pict_type = av.video.frame.PictureType.NONE
                        
                        for packet in left_output_stream.encode(av_frame):
                            if packet.pts is None:
                                continue
                            rr.set_time("timestamp", timestamp=data_time)
                            rr.log("videos_left", rr.VideoStream.from_fields(sample=bytes(packet)))
                    except Exception as e:
                        print(f"Left camera encoding error: {e}")
        time.sleep(0.02)

def _record_func_4():
    global right_output_stream
    if os.path.exists(SOCKET_PATH_4):
        os.unlink(SOCKET_PATH_4)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_4)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                header = conn.recv(13)
                if not header: # if client disconnect with server
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                data_bytes = b''
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                # deal with different type of data
                if data_type == TYPE_IMAGE_RIGHT:
                    img_data = BytesIO(data_bytes)
                    img = cv2.imdecode(np.frombuffer(img_data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    img_record = np.asanyarray(img)
                    right_q.put((img_record, data_time))
                    try:
                        av_frame = av.VideoFrame.from_ndarray(img_record, format="bgr24")
                        av_frame.pict_type = av.video.frame.PictureType.NONE
                        
                        for packet in right_output_stream.encode(av_frame):
                            if packet.pts is None:
                                continue
                            rr.set_time("timestamp", timestamp=data_time)
                            rr.log("videos_right", rr.VideoStream.from_fields(sample=bytes(packet)))
                    except Exception as e:
                        print(f"Right camera encoding error: {e}")
        time.sleep(0.02)

def _record_func_5():
    global front_output_stream
    if os.path.exists(SOCKET_PATH_5):
        os.unlink(SOCKET_PATH_5)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_5)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            # print ("connection got")
            while True:
                header = conn.recv(13)
                if not header: # if client disconnect with server
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                data_bytes = b''
                # print (data_type)
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                # deal with different type of data
                if data_type == TYPE_IMAGE_FRONT:
                    img_data = BytesIO(data_bytes)
                    img = cv2.imdecode(np.frombuffer(img_data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    img_record = np.asanyarray(img)
                    front_q.put((img_record, data_time))
                    try:
                        av_frame = av.VideoFrame.from_ndarray(img_record, format='bgr24')
                        av_frame.pict_type = av.video.frame.PictureType.NONE
                        
                        for packet in front_output_stream.encode(av_frame):
                            if packet.pts is None:
                                continue
                            rr.set_time("timestamp", timestamp=data_time)
                            rr.log("videos_front", rr.VideoStream.from_fields(sample=bytes(packet)))
                    except Exception as e:
                        print(f"Front camera encoding error: {e}")
        time.sleep(0.02)

def _record_func_11_stop():
    if os.path.exists(SOCKET_PATH_12):
        os.unlink(SOCKET_PATH_12)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH_12)
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                header = conn.recv(13)
                if not header:
                    time.sleep(0.001)
                    break
                data_type = header[0:1]
                data_len = struct.unpack('i', header[1:5])[0]
                data_time = struct.unpack('>q', header[5:13])[0]
                data_time *= 0.001
                data_bytes = b''
                while len(data_bytes) < data_len:
                    chunk = conn.recv(min(BUFFER_SIZE, data_len-len(data_bytes)))
                    if not chunk:
                        break
                    data_bytes += chunk
                if data_type == TYPE_DICT:
                    try:
                        data = json.loads(data_bytes.decode('utf-8'))
                    except Exception:
                        continue
                    if data.get('cmd') == 'stop_record':
                        _shutdown_record_server("stop_socket")
        time.sleep(0.01)

def handler(signum, frame):
    _shutdown_record_server(f"signal:{signum}")

signal.signal(signal.SIGTERM, handler)

if __name__ == "__main__":
    record_thread_1 = threading.Thread(target=_record_func_1)
    record_thread_1.daemon = True
    record_thread_1.start()
    record_thread_2 = threading.Thread(target=_record_func_2)
    record_thread_2.daemon = True
    record_thread_2.start()
    record_thread_3 = threading.Thread(target=_record_func_3)
    record_thread_3.daemon = True
    record_thread_3.start()
    record_thread_4 = threading.Thread(target=_record_func_4)
    record_thread_4.daemon = True
    record_thread_4.start()
    record_thread_5 = threading.Thread(target=_record_func_5)
    record_thread_5.daemon = True
    record_thread_5.start()
    record_thread_11 = threading.Thread(target=_record_func_11_stop)
    record_thread_11.daemon = True
    record_thread_11.start()

    main()