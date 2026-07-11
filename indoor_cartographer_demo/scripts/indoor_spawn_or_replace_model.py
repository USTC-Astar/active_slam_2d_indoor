#!/usr/bin/env python3
import math

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


def main():
    rospy.init_node("indoor_spawn_or_replace_model")

    model_name = rospy.get_param("~model_name", "indoor_mapper_bot")
    model_xml = rospy.get_param("robot_description", "")
    x = rospy.get_param("~x", 0.0)
    y = rospy.get_param("~y", 0.0)
    z = rospy.get_param("~z", 0.12)
    yaw = rospy.get_param("~yaw", 0.0)

    if not model_xml:
        rospy.logfatal("robot_description is empty; cannot spawn %s.", model_name)
        return 1

    rospy.wait_for_service("/gazebo/get_world_properties")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    get_world = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)

    world = get_world()
    if model_name in world.model_names:
        try:
            delete_model(model_name)
            rospy.sleep(0.5)
        except rospy.ServiceException as exc:
            rospy.logwarn("Could not delete existing model %s: %s", model_name, exc)

    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)

    result = spawn_model(model_name, model_xml, "", pose, "world")
    if not result.success:
        rospy.logfatal("Spawn failed: %s", result.status_message)
        return 1

    rospy.loginfo("Spawned %s at x=%.2f y=%.2f yaw=%.2f", model_name, x, y, yaw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
