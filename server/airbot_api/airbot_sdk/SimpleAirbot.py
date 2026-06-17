# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
import collections
from collections import deque
import cv2
import time
import math
import sys
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
import airbot_sdk
import threading
import multiprocessing
import json
import logging
import os
import queue
from datetime import datetime
from enum import Enum
from typing import Dict, List
import queue
import pyrealsense2 as rs
import signal
import argparse
import subprocess
import PyKDL
from kdl_parser.urdf import treeFromFile, treeFromUrdfModel
from urdf_parser_py.urdf import URDF
from airbot_sdk.utils.transformations import quaternion_matrix

import socket
import struct
from io import BytesIO

ANGLE2RAD = np.pi / 180.0
INTERP_POSITION = 0.005
INTERP_ROTATION = 0.02

def send_data(sock, data_type, data_time, data_bytes):
    data_len = len(data_bytes)
    header = data_type + struct.pack('i', data_len) + data_time
    sock.sendall(header)
    sock.sendall(data_bytes)


def quaternion_slerp(start, end, t):
    cosa = start[0] * end[0] + start[1] * end[1] + start[2] * end[2] + start[3] * end[3]
    # to make sure that the trace go along the "short circle", not the "long circle"
    if cosa < 0:
        end[0] = -end[0]
        end[1] = -end[1]
        end[2] = -end[2]
        end[3] = -end[3]
        cosa = - cosa
    # if a is too small that sina is close to a, use a instead
    if cosa > 0.9995:
        k0 = 1.0 - t
        k1 = t
    else:
        sina = math.sqrt(1.0 - cosa * cosa)
        a = math.atan2(sina, cosa)
        k0 = math.sin((1.0 - t) * a) / sina
        k1 = math.sin(t * a)/sina
    res = [0, 0, 0, 0]
    res[0] = start[0] * k0 + end[0] * k1
    res[1] = start[1] * k0 + end[1] * k1
    res[2] = start[2] * k0 + end[2] * k1
    res[3] = start[3] * k0 + end[3] * k1
    return res

def position_interpolation(start, end, t):
    start = np.array(start)
    end = np.array(end)
    position = (start + t*(end - start))
    return list(position)

def gripper_interpolation(start, end, t):
    ret = (start + t*(end - start))
    return ret

def cartesian_extend_with_gripper(  pose_start,
                                    pose_target,
                                    gripper_start,
                                    gripper_target,
                                    trans_step = 0.002,
                                    rot_step = 0.02):

    start_position = pose_start[:3]
    start_orient = pose_start[3:]
    target_position = pose_target[:3]
    target_orient = pose_target[3:]
    start_gripper = gripper_start
    target_gripper = gripper_target

    position_trace = list()
    orient_trace = list()
    pose_trace = list()
    gripper_trace = list()
    dis = np.linalg.norm(np.array(start_position) - np.array(target_position))
    position_inter_num = int(dis / trans_step)
    costheta = start_orient[0] * target_orient[0] + start_orient[1] * target_orient[1] + start_orient[2] * target_orient[2] + start_orient[3] * target_orient[3]
    if costheta > 0.9995:
        theta = 0
    else:
        sintheta = math.sqrt(1- costheta * costheta)
        theta = math.atan2(sintheta, costheta)
    orient_inter_num = int(theta / rot_step) # if use rot_step, the number of total interpolate point
    inter_num = max(position_inter_num, orient_inter_num)
    if inter_num <= 1:
        inter_num = 2
    t_ = 1.0 / (inter_num - 1)

    for i in range(inter_num):
        t = i * t_
        temp_orient = quaternion_slerp(list(start_orient), list(target_orient), t) #list
        temp_position = position_interpolation(list(start_position), list(target_position), t)
        temp_gripper = gripper_interpolation(start_gripper, target_gripper, t)
        orient_trace.append(temp_orient)#list
        position_trace.append(temp_position)
        gripper_trace.append(temp_gripper)
    if len(orient_trace) != inter_num:
        raise Exception("Match Error!")

    position_trace = np.array(position_trace)
    orient_trace = np.array(orient_trace)
    pose_trace = np.concatenate((position_trace,orient_trace), axis = 1)
    pose_trace = pose_trace.tolist()

    return pose_trace, gripper_trace

def load_urdf_to_kdl(urdf_file_path):
    if not os.path.exists(urdf_file_path):
        print(f"urdf not exists - {urdf_file_path}")
        return False, None
    
    try:
        success, tree = treeFromFile(urdf_file_path)
        if success:
            print(f"load sucess: {urdf_file_path}")
            print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
            return success, tree
    except Exception as e:
        print(f"methods1 failed: {e}")
    
    # 使用二进制模式读取文件并移除编码声明
    try:
        with open(urdf_file_path, 'rb') as f:
            # 读取二进制数据
            xml_content = f.read()
            # 尝试解码为字符串
            try:
                xml_str = xml_content.decode('utf-8')
            except UnicodeDecodeError:
                # 尝试其他编码
                xml_str = xml_content.decode('latin-1')
            # 移除XML编码声明
            if xml_str.startswith('<?xml'):
                xml_str = xml_str[xml_str.find('?>') + 2:]
            # 从字符串创建URDF模型
            robot_model = URDF.from_xml_string(xml_str)
            # 从URDF模型创建KDL树
            success, tree = treeFromUrdfModel(robot_model)
            if success:
                print(f"method2 sucess: {urdf_file_path}")
                print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
                return success, tree
            else:
                print("method2 failed")
    except Exception as e:
        print(f"method2 failed: {e}")
    
    try:
        robot_model = URDF.from_xml_file(urdf_file_path)
        success, tree = treeFromUrdfModel(robot_model)
        
        if success:
            print(f"method3 sucess: {urdf_file_path}")
            print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
            return success, tree
        else:
            print("method3 failed")
    except Exception as e:
        print(f"method3 failed: {e}")
    
    print(f"can not load: {urdf_file_path}")
    return False, None

def create_chain_from_tree(tree, base_link, tip_link):
    try:
        chain = tree.getChain(base_link, tip_link)
        print(f"create chain from {base_link} to {tip_link}")
        print(f"chain has {chain.getNrOfSegments()} segs and {chain.getNrOfJoints()} joints")
        return chain
    except Exception as e:
        print(f"create chain failed: {e}")
        return None

def setup_kinematics(chain):
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    max_iterations = 100
    eps = 1e-6
    # ik_pos_solver = PyKDL.ChainIkSolverPos_NR(
    #     chain, fk_solver, ik_vel_solver, max_iterations, eps
    # )
    min_jnt_array = PyKDL.JntArray(6)
    max_jnt_array = PyKDL.JntArray(6)
    for i in range(6):
        min_jnt_array[i] = -3.14
    for i in range(6):
        max_jnt_array[i] = 3.14
    ik_pos_solver = PyKDL.ChainIkSolverPos_NR_JL(
        chain, min_jnt_array, max_jnt_array, fk_solver, ik_vel_solver, max_iterations, eps
    )
    return fk_solver, ik_vel_solver, ik_pos_solver, chain

def perform_forward_kinematics(fk_solver, joint_positions):
    q = PyKDL.JntArray(len(joint_positions))
    for i in range(len(joint_positions)):
        q[i] = joint_positions[i]
    frame = PyKDL.Frame()
    status = fk_solver.JntToCart(q, frame)
    if status >= 0:
        position = [frame.p.x(), frame.p.y(), frame.p.z()]
        rotation = frame.M.GetQuaternion() 
        return frame
    else:
        print("fk failed")
        return None

def perform_inverse_kinematics(ik_pos_solver, chain, target_frame, initial_joints=None):
    num_joints = chain.getNrOfJoints()
    
    if initial_joints is None:
        q_init = PyKDL.JntArray(num_joints)
    else:
        q_init = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            q_init[i] = initial_joints[i]
    q_out = PyKDL.JntArray(num_joints)
    status = ik_pos_solver.CartToJnt(q_init, target_frame, q_out)
    
    if status >= 0:
        joint_positions = [q_out[i] for i in range(num_joints)]
        return joint_positions
    else:
        print("ik failed")
        return None

class SimpleAirbotRobot():
    def __init__(self,
                left_port=50051, left_url='localhost', right_port=50053, right_url='localhost',
                left_lead_port=50050, left_lead_url='localhost', right_lead_port=50052, right_lead_url='localhost',
                ):
        self.running = True
        # for robot
        self.left_port = left_port # 50051 left, 50053 right
        self.left_url = left_url
        self.right_port = right_port
        self.right_url = right_url
        self.left_lead_port = left_lead_port
        self.left_lead_url = left_lead_url
        self.right_lead_port = right_lead_port
        self.right_lead_url = right_lead_url
        self.left_arm = AIRBOTPlay(url=self.left_url, port=self.left_port)
        self.left_arm.connect()
        time.sleep(2.0)
        self.right_arm = AIRBOTPlay(url=self.right_url, port=self.right_port)
        self.right_arm.connect()
        time.sleep(2.0)
        self.left_lead_arm = AIRBOTPlay(url=self.left_lead_url, port=self.left_lead_port)
        self.left_lead_arm.connect()
        time.sleep(2.0)
        self.right_lead_arm = AIRBOTPlay(url=self.right_lead_url, port=self.right_lead_port)
        self.right_lead_arm.connect()
        # time.sleep(2.0)
        self.left_cur_joint = []
        self.right_cur_joint = []
        self.left_lead_joint = []
        self.right_lead_joint = []
        self.left_cur_pose = []
        self.right_cur_pose = []
        # for kinematics
        tmp_dir = os.path.dirname(os.path.abspath(airbot_sdk.__file__))
        urdf_file = "/home/ubuntu/dos_w1/airbot/airbot_api/config/airbot_urdf/play_g2/urdf/play_g2.urdf"
        lead_urdf_file = "/home/ubuntu/dos_w1/airbot/airbot_api/config/airbot_urdf/replay/urdf/replay.urdf"
        ret, tree = load_urdf_to_kdl(urdf_file)
        self.chain_ = create_chain_from_tree(tree, "base_link", "end_link")
        self.fk_solver, self.ik_vel_solver, self.ik_pos_solver, _ = setup_kinematics(self.chain_)
        # for lead arm
        ret, tree = load_urdf_to_kdl(lead_urdf_file)
        self.lead_chain_ = create_chain_from_tree(tree, "base_link", "end_link")
        self.lead_fk_solver, self.lead_ik_vel_solver, self.lead_ik_pos_solver, _ = setup_kinematics(self.lead_chain_)
        # for robot move
        self.left_lead_arm.switch_mode(RobotMode.GRAVITY_COMP)
        self.right_lead_arm.switch_mode(RobotMode.GRAVITY_COMP)
        self.left_arm.set_speed_profile(SpeedProfile.FAST)
        self.left_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self.right_arm.set_speed_profile(SpeedProfile.FAST)
        self.right_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self.left_move_lock = threading.Lock()
        self.right_move_lock = threading.Lock()
        self.left_target_joint_list = []
        self.right_target_joint_list = []
        self.left_target_gripper_list = []
        self.right_target_gripper_list = []
        self._control_thread = threading.Thread(target=self._control_func, daemon=True)
        self._control_thread.start()

    def _control_func(self):
        while self.running:
            # first get
            s_t = time.time()
            # get lead joint
            pos = self.left_lead_arm.get_joint_pos()
            eef_pos = self.left_lead_arm.get_eef_pos()
            pos.extend(eef_pos)
            self.left_lead_joint = pos
            pos = self.right_lead_arm.get_joint_pos()
            eef_pos = self.right_lead_arm.get_eef_pos()
            pos.extend(eef_pos)
            self.right_lead_joint = pos
            # get follow joint
            pos = self.left_arm.get_joint_pos()
            eef_pos = self.left_arm.get_eef_pos()
            pos.extend(eef_pos)
            e_t = time.time()
            self.left_cur_joint = pos
            pos = self.right_arm.get_joint_pos()
            eef_pos = self.right_arm.get_eef_pos()
            pos.extend(eef_pos)
            self.right_cur_joint = pos
            # get pose
            position, quat = self.left_arm.get_end_pose()
            position.extend(quat)
            eef_pos = self.left_arm.get_eef_pos()
            position.extend(eef_pos)
            self.left_cur_pose = position
            position, quat = self.right_arm.get_end_pose()
            position.extend(quat)
            eef_pos = self.right_arm.get_eef_pos()
            position.extend(eef_pos)
            self.right_cur_pose = position
            # for left
            if len(self.left_target_joint_list) and len(self.left_target_gripper_list):
                self._left_go_joint_servo(self.left_target_joint_list[0], self.left_target_gripper_list[0])
                self.left_target_joint_list.pop(0)
            # for right
            if len(self.right_target_joint_list) and len(self.right_target_gripper_list):
                self._right_go_joint_servo(self.right_target_joint_list[0], self.right_target_gripper_list[0])
                self.right_target_joint_list.pop(0)
                self.right_target_gripper_list.pop(0)
            e_t = time.time()
            # print ("total t: ", e_t - s_t)
            time.sleep(0.02)
    
    def __del__(self):
        self.running = False
        self.left_arm.disconnect()
        time.sleep(2.0)
        self.right_arm.disconnect()
        time.sleep(2.0)
        self.left_lead_arm.disconnect()
        time.sleep(1.0)
        self.right_lead_arm.disconnect()
        time.sleep(1.0)
        print ("del")

    def fk(self, joint):
        frame_ = perform_forward_kinematics(self.fk_solver, joint)
        pose_ = [frame_.p.x(), frame_.p.y(), frame_.p.z()] + list(frame_.M.GetQuaternion())
        return pose_

    def lead_fk(self, joint):
        frame_ = perform_forward_kinematics(self.lead_fk_solver, joint)
        pose_ = [frame_.p.x(), frame_.p.y(), frame_.p.z()] + list(frame_.M.GetQuaternion())
        return pose_

    def ik(self, target_pose, init_joint=[0,0,0,0,0,0]):
        target_position = PyKDL.Vector(target_pose[0], target_pose[1], target_pose[2])
        m_ = quaternion_matrix(target_pose[3:])
        target_R = PyKDL.Rotation(
            m_[0,0], m_[0,1],m_[0,2],
            m_[1,0], m_[1,1],m_[1,2],
            m_[2,0], m_[2,1],m_[2,2],
        )
        target_frame = PyKDL.Frame(target_R, target_position)
        ik_res = perform_inverse_kinematics(self.ik_pos_solver, self.chain_, target_frame, init_joint)
        return ik_res

    def ik_far(self, target_pose, init_joint):
        init_joint = np.round(init_joint, decimals=3)
        init_pose = self.fk(init_joint)
        pose_trace, _ = cartesian_extend_with_gripper(init_pose, target_pose, 0.0, 0.0)
        ik_res_list = []
        for pose in pose_trace:
            ik_res = self.ik(pose, init_joint)
            init_joint = ik_res
            ik_res_list.append(ik_res)
        return ik_res_list[-1]

    
    def left_get_joint(self):
        return self.left_cur_joint

    def right_get_joint(self):
        return self.right_cur_joint

    def left_get_pose(self):
        return self.left_cur_pose

    def right_get_pose(self):
        return self.right_cur_pose

    def left_go_joint(self, joint, gripper, interp=True):
        if interp:
        # first, interpolate
            interp_target_joint_list = []
            cur_joint_state = self.left_get_joint()
            cur_joint = cur_joint_state[:6]
            cur_gripper = cur_joint_state[6]
            start_list = np.array(cur_joint)
            end_list = np.array(joint)
            num = int(np.linalg.norm(end_list - start_list) / 0.02)
            num = max(num, 1)
            start_list_with_gripper = np.concatenate((start_list, [gripper]))
            end_list_with_gripper = np.concatenate((end_list, [gripper]))
            interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num, endpoint=True)
            interp_target_joint_list.extend(interp_list.tolist())
            self.left_target_joint_list = [item[:6] for item in interp_target_joint_list]
            self.left_target_gripper_list = [item[6] for item in interp_target_joint_list]
        else:
            self.left_target_joint_list = [joint]
            self.left_target_gripper_list = [gripper]

    def _left_go_joint_servo(self, joint, gripper):
        self.left_arm.servo_eef_pos([gripper])
        self.left_arm.servo_joint_pos(joint)

    def right_go_joint(self, joint, gripper, interp=True):
        if interp:
            # first, interpolate
            interp_target_joint_list = []
            cur_joint_state = self.right_get_joint()
            cur_joint = cur_joint_state[:6]
            cur_gripper = cur_joint_state[6]
            start_list = np.array(cur_joint)
            end_list = np.array(joint)
            num = int(np.linalg.norm(end_list - start_list) / 0.02)
            num = max(num, 1)
            start_list_with_gripper = np.concatenate((start_list, [gripper]))
            end_list_with_gripper = np.concatenate((end_list, [gripper]))
            interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num, endpoint=True)
            interp_target_joint_list.extend(interp_list.tolist())
            self.right_target_joint_list = [item[:6] for item in interp_target_joint_list]
            self.right_target_gripper_list = [item[6] for item in interp_target_joint_list]
        else:
            self.right_target_joint_list = [joint]
            self.right_target_gripper_list = [gripper]

    def _right_go_joint_servo(self, joint, gripper):
        self.right_arm.servo_eef_pos([gripper])
        self.right_arm.servo_joint_pos(joint)

    def left_go_pose(self, pose, gripper, interp=True):
        if interp:
            target_joint = self.ik_far(pose, self.left_cur_joint[:6])
            if target_joint:
                self.left_go_joint(target_joint, gripper, interp)
        else:
            target_joint = self.ik(pose, self.left_cur_joint[:6])
            if target_joint:
                self.left_go_joint(target_joint, gripper, interp)

    def right_go_pose(self, pose, gripper, interp=True):
        if interp:
            target_joint = self.ik_far(pose, self.right_cur_joint[:6])
            if target_joint:
                self.right_go_joint(target_joint, gripper, interp)
        else:
            target_joint = self.ik(pose, self.right_cur_joint[:6])
            if target_joint:
                self.right_go_joint(target_joint, gripper, interp)

    def left_go_joint_list(self, joint_list, gripper_list):
        '''
        notice: interp threshold cannot be too large
        '''
        if len(joint_list) != len(gripper_list):
            return
        interp_target_joint_list = []
        cur_joint_state = self.left_get_joint()
        cur_joint = cur_joint_state[:6]
        cur_gripper = cur_joint_state[6]
        joint_list = [cur_joint] + joint_list
        gripper_list = [cur_gripper] + gripper_list
        for i in range(len(joint_list) - 1):
            start_list = np.array(joint_list[i])
            end_list = np.array(joint_list[i+1])
            num = int(np.linalg.norm(end_list - start_list) / 0.02)
            num = max(num, 1)
            start_list_with_gripper = np.concatenate((start_list, [gripper_list[i]]))
            end_list_with_gripper = np.concatenate((end_list, [gripper_list[i+1]]))
            if i == len(joint_list) - 2:
                interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num)
            else:
                interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num, endpoint=False)
            interp_target_joint_list.extend(interp_list.tolist())
        # with self.left_move_lock:
        self.left_target_joint_list = [item[:6] for item in interp_target_joint_list]
        self.left_target_gripper_list = [item[6] for item in interp_target_joint_list]

    def right_go_joint_list(self, joint_list, gripper_list):
        '''
        notice: interp threshold cannot be too large
        '''
        if len(joint_list) != len(gripper_list):
            return
        interp_target_joint_list = []
        cur_joint_state = self.right_get_joint()
        cur_joint = cur_joint_state[:6]
        cur_gripper = cur_joint_state[6]
        joint_list = [cur_joint] + joint_list
        gripper_list = [cur_gripper] + gripper_list
        for i in range(len(joint_list) - 1):
            start_list = np.array(joint_list[i])
            end_list = np.array(joint_list[i+1])
            num = int(np.linalg.norm(end_list - start_list) / 0.02)
            num = max(num, 1)
            start_list_with_gripper = np.concatenate((start_list, [gripper_list[i]]))
            end_list_with_gripper = np.concatenate((end_list, [gripper_list[i+1]]))
            if i == len(joint_list) - 2:
                interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num)
            else:
                interp_list = np.linspace(start_list_with_gripper, end_list_with_gripper, num, endpoint=False)
            interp_target_joint_list.extend(interp_list.tolist())
        # with self.right_move_lock:
        self.right_target_joint_list = [item[:6] for item in interp_target_joint_list]
        self.right_target_gripper_list = [item[6] for item in interp_target_joint_list]

    def left_go_pose_list(self, pose_list, gripper_list):
        '''
        notice: interp threshold cannot be too large
        '''
        if len(pose_list) != len(gripper_list):
            return
        init_joint = self.left_cur_joint[:6]
        target_joint_list = []
        for i in range(len(pose_list)):
            tmp_joint = self.ik_far(pose_list[i], init_joint)
            init_joint = tmp_joint
            target_joint_list.append(tmp_joint)
        self.left_go_joint_list(target_joint_list, gripper_list)

    def right_go_pose_list(self, pose_list, gripper_list):
        '''
        notice: interp threshold cannot be too large
        '''
        if len(pose_list) != len(gripper_list):
            return
        init_joint = self.right_cur_joint[:6]
        target_joint_list = []
        for i in range(len(pose_list)):
            tmp_joint = self.ik_far(pose_list[i], init_joint)
            init_joint = tmp_joint
            target_joint_list.append(tmp_joint)
        self.right_go_joint_list(target_joint_list, gripper_list)

# if __name__ == "__main__":
#     ar = AirbotRobot(port=50051)
#     print (ar.get_joint())
#     # test go joint
#     ar.go_joint([0,0,0,0,0,0], 0.01)
#     time.sleep(1.0)

    # # test go joint list
    # target_joint_list = [
    #     [0.37, -0.6, 0.5, 0.132, -0.30, -0.13],
    #     [0.2, -2.0, 2.0, 0.0, 0.01, 0.0],
    #     [0.2, 0.1, 0.3, 0.0, 0.01, 0.0],
    #     [-0.2, 0.15, 0.0, 0.0, 0.01, 0.0],
    #     [0,0,0,0,0,0]
    # ]
    # target_gripper_list = [0.01, 0.02, 0.03, 0.04, 0.01]
    # ar.go_joint_list(target_joint_list, target_gripper_list)
    # time.sleep(2.0)
    # print (ar.get_joint())

    # # test go pose
    # target_pose = [0.45, 0.20, 0.3, 0, 0, 0, 1]
    # target_gripper = 0.01
    # ar.go_pose(target_pose, target_gripper)
    # print (ar.get_pose())

    # # test go pose list
    # target_pose_list = [
    #     [0.28635771926232956, -0.00010167414761488679, 0.21330598726999278, -0.00019073775689503993, 0.00038149372158484076, -0.00019066499510091497, 0.9999998908642481],
    #     [0.32, 0.1, 0.2, 0, 0, 0, 1],
    #     [0.45, 0.20, 0.3, 0, 0, 0, 1]
    # ]
    # target_gripper_list = [0.01, 0.02, 0.03]
    # ar.go_pose_list(target_pose_list, target_gripper_list)
    # time.sleep(2.0)
    # print (ar.get_pose())