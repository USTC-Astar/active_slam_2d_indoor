#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry


class StrictOdomFilter:
    def __init__(self):
        self.last_stamp = rospy.Time(0)
        self.dropped = 0
        self.publisher = rospy.Publisher("/cartographer/odom", Odometry, queue_size=50)
        self.subscriber = rospy.Subscriber("/odom", Odometry, self.callback, queue_size=100)

    def callback(self, msg):
        if msg.header.stamp <= self.last_stamp:
            self.dropped += 1
            rospy.logwarn_throttle(
                5.0,
                "odom filter dropped %d duplicate or out-of-order messages",
                self.dropped,
            )
            return
        self.last_stamp = msg.header.stamp
        self.publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node("indoor_odom_filter")
    StrictOdomFilter()
    rospy.spin()
