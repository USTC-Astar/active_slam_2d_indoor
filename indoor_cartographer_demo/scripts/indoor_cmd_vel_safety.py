#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def clamp(value, low, high):
    return max(low, min(high, value))


class CmdVelSafety:
    def __init__(self):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.status_pub = rospy.Publisher("/cmd_vel_safety/status", String, queue_size=1, latch=True)
        self.raw_sub = rospy.Subscriber("/move_base/cmd_vel_raw", Twist, self.raw_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)

        self.max_forward_speed = rospy.get_param("~max_forward_speed", 1.15)
        self.max_reverse_speed = rospy.get_param("~max_reverse_speed", 0.28)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.95)
        self.slow_distance = rospy.get_param("~slow_distance", 0.64)
        self.stop_distance = rospy.get_param("~stop_distance", 0.38)
        self.emergency_distance = rospy.get_param("~emergency_distance", 0.27)
        self.side_stop_distance = rospy.get_param("~side_stop_distance", 0.24)
        self.clear_speed_distance = rospy.get_param("~clear_speed_distance", 0.85)
        self.speed_boost = rospy.get_param("~speed_boost", 1.6)
        self.min_clear_speed = rospy.get_param("~min_clear_speed", 0.32)
        self.boost_angular_limit = rospy.get_param("~boost_angular_limit", 0.45)
        self.backup_seconds = rospy.Duration(rospy.get_param("~backup_seconds", 0.85))
        self.turn_seconds = rospy.Duration(rospy.get_param("~turn_seconds", 0.75))
        self.stuck_seconds = rospy.Duration(rospy.get_param("~stuck_seconds", 3.0))
        self.stuck_distance = rospy.get_param("~stuck_distance", 0.045)
        self.raw_timeout = rospy.Duration(rospy.get_param("~raw_timeout", 0.6))

        self.raw_cmd = Twist()
        self.raw_stamp = rospy.Time(0)
        self.scan = None
        self.odom = None
        self.motion_reference = None
        self.motion_reference_stamp = rospy.Time.now()
        self.recovery_mode = None
        self.recovery_until = rospy.Time(0)
        self.turn_sign = 1.0

        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)

    def raw_callback(self, msg):
        self.raw_cmd = msg
        self.raw_stamp = rospy.Time.now()

    def scan_callback(self, msg):
        self.scan = msg

    def odom_callback(self, msg):
        self.odom = msg

    def sector_min(self, low, high):
        if self.scan is None:
            return float("inf")
        best = float("inf")
        angle = self.scan.angle_min
        for value in self.scan.ranges:
            if low <= angle <= high and math.isfinite(value):
                best = min(best, value)
            angle += self.scan.angle_increment
        return best

    def pose_xy(self):
        if self.odom is None:
            return None
        pose = self.odom.pose.pose.position
        return pose.x, pose.y

    def begin_recovery(self, reason):
        left = self.sector_min(0.35, 1.45)
        right = self.sector_min(-1.45, -0.35)
        self.turn_sign = 1.0 if left >= right else -1.0
        self.recovery_mode = "backup"
        self.recovery_until = rospy.Time.now() + self.backup_seconds
        self.status_pub.publish("%s: backing up" % reason)

    def recovery_cmd(self):
        now = rospy.Time.now()
        cmd = Twist()
        if self.recovery_mode == "backup":
            if now < self.recovery_until:
                cmd.linear.x = -self.max_reverse_speed
                cmd.angular.z = -0.25 * self.turn_sign
                return cmd
            self.recovery_mode = "turn"
            self.recovery_until = now + self.turn_seconds

        if self.recovery_mode == "turn":
            if now < self.recovery_until:
                cmd.angular.z = self.turn_sign * min(1.05, self.max_angular_speed)
                return cmd
            self.recovery_mode = None
            self.motion_reference = self.pose_xy()
            self.motion_reference_stamp = now

        return None

    def stuck_detected(self, cmd):
        xy = self.pose_xy()
        now = rospy.Time.now()
        if xy is None:
            return False
        if cmd.linear.x <= 0.12:
            self.motion_reference = xy
            self.motion_reference_stamp = now
            return False
        if self.motion_reference is None:
            self.motion_reference = xy
            self.motion_reference_stamp = now
            return False
        if now - self.motion_reference_stamp < self.stuck_seconds:
            return False
        moved = math.hypot(xy[0] - self.motion_reference[0], xy[1] - self.motion_reference[1])
        self.motion_reference = xy
        self.motion_reference_stamp = now
        return moved < self.stuck_distance

    def filtered_cmd(self):
        now = rospy.Time.now()
        if now - self.raw_stamp > self.raw_timeout:
            return Twist(), "waiting for move_base cmd_vel"

        cmd = Twist()
        cmd.linear.x = clamp(self.raw_cmd.linear.x, -self.max_reverse_speed, self.max_forward_speed)
        cmd.angular.z = clamp(self.raw_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)

        front = self.sector_min(-0.34, 0.34)
        left = self.sector_min(0.35, 1.35)
        right = self.sector_min(-1.35, -0.35)

        if cmd.linear.x > 0.02 and front < self.emergency_distance:
            self.begin_recovery("emergency obstacle")
            return Twist(), "emergency stop"

        if cmd.linear.x > 0.02 and front < self.stop_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.9 if left >= right else -0.9
            return cmd, "blocked ahead: turning"

        slowed = False
        if cmd.linear.x > 0.02 and front < self.slow_distance:
            scale = clamp((front - self.stop_distance) / max(0.01, self.slow_distance - self.stop_distance), 0.35, 1.0)
            cmd.linear.x *= scale
            slowed = True

        boosted = False
        if cmd.linear.x > 0.08 and not slowed and front > self.clear_speed_distance and abs(cmd.angular.z) < self.boost_angular_limit:
            angular_ratio = clamp(abs(cmd.angular.z) / max(0.01, self.boost_angular_limit), 0.0, 1.0)
            turn_factor = 1.0 - 0.45 * angular_ratio
            boosted_speed = max(cmd.linear.x * self.speed_boost, self.min_clear_speed * turn_factor)
            cmd.linear.x = clamp(boosted_speed, 0.0, self.max_forward_speed)
            boosted = True

        if left < self.side_stop_distance and cmd.angular.z > 0.0:
            cmd.angular.z = min(cmd.angular.z, 0.0)
        if right < self.side_stop_distance and cmd.angular.z < 0.0:
            cmd.angular.z = max(cmd.angular.z, 0.0)

        if self.stuck_detected(cmd):
            self.begin_recovery("stuck")
            return Twist(), "stuck recovery"

        if boosted:
            return cmd, "clear path speed boost"
        return cmd, "passing move_base command"

    def update(self, _event):
        recovery = self.recovery_cmd()
        if recovery is not None:
            self.cmd_pub.publish(recovery)
            self.status_pub.publish("recovery: %s" % self.recovery_mode)
            return

        cmd, status = self.filtered_cmd()
        self.cmd_pub.publish(cmd)
        self.status_pub.publish(status)


if __name__ == "__main__":
    rospy.init_node("indoor_cmd_vel_safety")
    CmdVelSafety()
    rospy.spin()
