# airbot_api

airbot臂的api接口

## Install
```
pip install .
```
在python中
```
import airbot_sdk
```
成功即为调用成功

## Usage
1. 启动驱动  
```
bash start.sh
```
2. 接口示例
* 可参考test.py中的使用  
```
ar = AirbotRobot(port=50051) # port为端口号，左臂为50051，右臂为50053
```
* 获取状态
```
j = ar.get_joint() # 返回6个关节角+夹爪张开距离，关节角单位为rad，gripper单位为米

p = ar.get_pose() # 返回xyz和四元数
```
* 单点运动
```
ar.go_joint(joint_list, gripper) # joint_list为6个关节角度的list，gripper为开合距离（m）
ar.go_pose(target_pose, target_gripper) # target_pose为xyz+quaternion的list，gripper为夹爪开合距离
```
* 伺服多点运动
```
ar.go_joint_list(target_joint_list, target_gripper_list)
ar.go_pose_list(target_pose_list, target_gripper_list)
```
该模式允许输入目标位置的list，机械臂将以最快速度连续运动到各个路点，中间不会停止