#!/usr/bin/env python3
"""Bridge Unitree SDK2 DDS low-level I/O to HoloMotion ROS2 topics."""

from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as DdsLowCmd
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as DdsLowState

import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowCmd as RosLowCmd
from unitree_hg.msg import LowState as RosLowState


def copy_sequence(dst, src) -> None:
    for i, value in enumerate(src):
        dst[i] = value


class UnitreeRos2Bridge(Node):
    def __init__(self) -> None:
        super().__init__("wbc_unitree_ros2_bridge")
        self.lowstate_pub = self.create_publisher(RosLowState, "/lowstate", 10)
        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", DdsLowCmd)
        self.lowcmd_pub.Init()
        self.lowcmd_sub = self.create_subscription(RosLowCmd, "/lowcmd", self.on_ros_lowcmd, 10)
        self.dds_lowstate_sub = ChannelSubscriber("rt/lowstate", DdsLowState)
        self.dds_lowstate_sub.Init(self.on_dds_lowstate, 10)
        self.last_lowstate_wall = 0.0
        self.forwarded_dpad_bits = 0

    def on_dds_lowstate(self, msg: DdsLowState) -> None:
        out = RosLowState()
        copy_sequence(out.version, msg.version)
        out.mode_pr = int(msg.mode_pr)
        out.mode_machine = int(msg.mode_machine)
        out.tick = int(msg.tick)
        copy_sequence(out.imu_state.quaternion, msg.imu_state.quaternion)
        copy_sequence(out.imu_state.gyroscope, msg.imu_state.gyroscope)
        copy_sequence(out.imu_state.accelerometer, msg.imu_state.accelerometer)
        copy_sequence(out.imu_state.rpy, msg.imu_state.rpy)
        out.imu_state.temperature = int(msg.imu_state.temperature)
        for i, motor in enumerate(msg.motor_state):
            out.motor_state[i].mode = int(motor.mode)
            out.motor_state[i].q = float(motor.q)
            out.motor_state[i].dq = float(motor.dq)
            out.motor_state[i].ddq = float(getattr(motor, "ddq", 0.0))
            out.motor_state[i].tau_est = float(motor.tau_est)
            copy_sequence(out.motor_state[i].temperature, motor.temperature)
            out.motor_state[i].vol = float(motor.vol)
            copy_sequence(out.motor_state[i].sensor, motor.sensor)
            out.motor_state[i].motorstate = int(motor.motorstate)
            copy_sequence(out.motor_state[i].reserve, motor.reserve)
        copy_sequence(out.wireless_remote, msg.wireless_remote)
        key_bits = int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3]) << 8)
        dpad_bits = key_bits & 0xF000
        if dpad_bits == 0:
            self.forwarded_dpad_bits = 0
        elif self.forwarded_dpad_bits == dpad_bits:
            filtered = key_bits & ~dpad_bits
            out.wireless_remote[2] = filtered & 0xFF
            out.wireless_remote[3] = (filtered >> 8) & 0xFF
        else:
            self.forwarded_dpad_bits = dpad_bits
        copy_sequence(out.reserve, msg.reserve)
        out.crc = int(msg.crc)
        self.lowstate_pub.publish(out)
        self.last_lowstate_wall = time.time()

    def on_ros_lowcmd(self, msg: RosLowCmd) -> None:
        out = unitree_hg_msg_dds__LowCmd_()
        out.mode_pr = int(msg.mode_pr)
        out.mode_machine = int(msg.mode_machine)
        for i, motor in enumerate(msg.motor_cmd):
            out.motor_cmd[i].mode = int(motor.mode)
            out.motor_cmd[i].q = float(motor.q)
            out.motor_cmd[i].dq = float(motor.dq)
            out.motor_cmd[i].tau = float(motor.tau)
            out.motor_cmd[i].kp = float(motor.kp)
            out.motor_cmd[i].kd = float(motor.kd)
            out.motor_cmd[i].reserve = int(motor.reserve)
        copy_sequence(out.reserve, msg.reserve)
        out.crc = int(msg.crc)
        self.lowcmd_pub.Write(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=1)
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    rclpy.init()
    node = UnitreeRos2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
