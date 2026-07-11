#!/usr/bin/env python3
import collections
import math
import threading

import rospy
import tf2_ros
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class ActiveSlamExplorer:
    def __init__(self):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.status_pub = rospy.Publisher("/active_slam/status", String, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/active_slam/path", Path, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher("/active_slam/target", PoseStamped, queue_size=1, latch=True)
        self.frontier_pub = rospy.Publisher("/active_slam/frontiers", MarkerArray, queue_size=1, latch=True)

        self.map_lock = threading.RLock()
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        self.contact_sub = rospy.Subscriber("/bumper", ContactsState, self.contact_callback, queue_size=1)

        self.map_msg = None
        self.scan = None
        self.scan_points = []
        self.scan_observations = []
        self.odom = None
        self.last_contact_time = rospy.Time(0)
        self.current_plan = []
        self.current_goal_cell = None
        self.current_goal_source = None
        self.local_target = None
        self.blacklist = {}
        self.recent_patrol_goals = collections.deque(maxlen=10)

        self.free_threshold = int(rospy.get_param("~free_threshold", 35))
        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 58))
        self.inflate_radius = rospy.get_param("~inflate_radius", 0.22)
        self.min_frontier_cells = int(rospy.get_param("~min_frontier_cells", 10))
        self.min_frontier_path_distance = rospy.get_param("~min_frontier_path_distance", 1.2)
        self.frontier_gain_weight = rospy.get_param("~frontier_gain_weight", 2.4)
        self.frontier_distance_weight = rospy.get_param("~frontier_distance_weight", 0.45)
        self.frontier_heading_weight = rospy.get_param("~frontier_heading_weight", 0.45)
        self.frontier_far_bonus = rospy.get_param("~frontier_far_bonus", 0.20)
        self.frontier_goal_hysteresis = rospy.get_param("~frontier_goal_hysteresis", 3.0)
        self.patrol_min_distance = rospy.get_param("~patrol_min_distance", 1.5)
        self.patrol_max_distance = rospy.get_param("~patrol_max_distance", 5.0)
        self.patrol_unknown_radius = int(rospy.get_param("~patrol_unknown_radius", 5))
        self.patrol_cell_stride = int(rospy.get_param("~patrol_cell_stride", 4))
        self.max_bfs_cells = int(rospy.get_param("~max_bfs_cells", 90000))
        self.replan_interval = rospy.Duration(rospy.get_param("~replan_interval", 2.8))
        self.goal_timeout = rospy.Duration(rospy.get_param("~goal_timeout", 16.0))
        self.blacklist_seconds = rospy.Duration(rospy.get_param("~blacklist_seconds", 18.0))
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.35)
        self.min_local_target_distance = rospy.get_param("~min_local_target_distance", 0.38)
        self.goal_reached_distance = rospy.get_param("~goal_reached_distance", 0.48)

        self.cruise_speed = rospy.get_param("~cruise_speed", 0.62)
        self.min_drive_speed = rospy.get_param("~min_drive_speed", 0.24)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.75)
        self.turn_gain = rospy.get_param("~turn_gain", 1.85)
        self.local_heading_samples = int(rospy.get_param("~local_heading_samples", 17))
        self.local_heading_limit = rospy.get_param("~local_heading_limit", 1.35)
        self.local_clear_distance = rospy.get_param("~local_clear_distance", 1.35)
        self.local_stop_margin = rospy.get_param("~local_stop_margin", 0.12)
        self.obstacle_clearance_weight = rospy.get_param("~obstacle_clearance_weight", 1.25)
        self.heading_tracking_weight = rospy.get_param("~heading_tracking_weight", 2.2)
        self.forward_heading_weight = rospy.get_param("~forward_heading_weight", 0.35)
        self.front_slow_distance = rospy.get_param("~front_slow_distance", 0.82)
        self.front_stop_distance = rospy.get_param("~front_stop_distance", 0.42)
        self.emergency_distance = rospy.get_param("~emergency_distance", 0.25)
        self.side_stop_distance = rospy.get_param("~side_stop_distance", 0.25)
        self.rear_stop_distance = rospy.get_param("~rear_stop_distance", 0.31)
        self.robot_half_width = rospy.get_param("~robot_half_width", 0.235)
        self.robot_front_radius = rospy.get_param("~robot_front_radius", 0.25)
        self.laser_offset_x = rospy.get_param("~laser_offset_x", 0.11)
        self.scan_self_filter_radius = rospy.get_param("~scan_self_filter_radius", 0.20)
        self.corridor_margin = rospy.get_param("~corridor_margin", 0.055)
        self.max_linear_deceleration = rospy.get_param("~max_linear_deceleration", 1.45)
        self.stuck_timeout = rospy.Duration(rospy.get_param("~stuck_timeout", 4.0))
        self.stuck_distance = rospy.get_param("~stuck_distance", 0.06)
        self.blocked_timeout = rospy.Duration(rospy.get_param("~blocked_timeout", 2.0))
        self.contact_hold = rospy.Duration(rospy.get_param("~contact_hold", 0.45))
        self.brake_seconds = rospy.Duration(rospy.get_param("~brake_seconds", 0.16))
        self.backup_seconds = rospy.Duration(rospy.get_param("~backup_seconds", 1.05))
        self.turn_seconds = rospy.Duration(rospy.get_param("~turn_seconds", 1.15))
        self.probe_seconds = rospy.Duration(rospy.get_param("~probe_seconds", 0.9))
        self.recovery_reset_seconds = rospy.Duration(rospy.get_param("~recovery_reset_seconds", 12.0))
        self.backup_speed = rospy.get_param("~backup_speed", 0.30)
        self.recovery_turn_speed = rospy.get_param("~recovery_turn_speed", 1.25)
        self.spin_timeout = rospy.Duration(rospy.get_param("~spin_timeout", 4.2))
        self.spin_min_angle = rospy.get_param("~spin_min_angle", 4.5)
        self.spin_max_translation = rospy.get_param("~spin_max_translation", 0.16)
        self.control_rate = rospy.get_param("~control_rate", 24.0)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.last_plan_time = rospy.Time(0)
        self.goal_started = rospy.Time(0)
        self.progress_pose = None
        self.progress_time = rospy.Time.now()
        self.recovery_mode = None
        self.recovery_until = rospy.Time(0)
        self.recovery_turn_sign = 1.0
        self.recovery_attempt = 0
        self.last_recovery_time = rospy.Time(0)
        self.blocked_since = None
        self.blocked_turn_sign = 1.0
        self.last_command = Twist()
        self.spin_start_time = None
        self.spin_start_position = None
        self.spin_last_yaw = None
        self.spin_accumulated_angle = 0.0

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.control_rate)), self.update)

    def map_callback(self, msg):
        with self.map_lock:
            self.map_msg = msg

    def scan_callback(self, msg):
        self.scan = msg
        points = []
        observations = []
        angle = msg.angle_min
        for value in msg.ranges:
            if math.isfinite(value) and self.scan_self_filter_radius <= value <= msg.range_max:
                x = self.laser_offset_x + value * math.cos(angle)
                y = value * math.sin(angle)
                inside_robot = (
                    -self.robot_front_radius - 0.03 <= x <= self.robot_front_radius + 0.03 and
                    abs(y) <= self.robot_half_width + 0.025
                )
                if not inside_robot:
                    points.append((x, y))
                    observations.append((value, angle))
            angle += msg.angle_increment
        self.scan_points = points
        self.scan_observations = observations

    def odom_callback(self, msg):
        self.odom = msg

    def contact_callback(self, msg):
        for state in msg.states:
            other_name = state.collision2_name.lower()
            if "indoor_mapper_bot" not in state.collision1_name:
                other_name = state.collision1_name.lower()
            if "floor" in other_name or "ground_plane" in other_name:
                continue
            horizontal_contact = any(abs(normal.z) < 0.65 for normal in state.contact_normals)
            if horizontal_contact or not state.contact_normals:
                self.last_contact_time = rospy.Time.now()
                return

    def pose(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.05))
        except Exception:
            return None
        t = tf.transform.translation
        yaw = yaw_from_quaternion(tf.transform.rotation)
        return t.x, t.y, yaw

    def sector_min(self, low, high):
        if self.scan is None:
            return float("inf")
        best = float("inf")
        for value, angle in self.scan_observations:
            if low <= angle <= high:
                best = min(best, value)
        return best

    def rear_min(self):
        return min(
            self.sector_min(2.55, math.pi),
            self.sector_min(-math.pi, -2.55),
        )

    def heading_clearance(self, heading):
        """Distance along a candidate corridor, including the robot footprint."""
        if self.scan is None:
            return float("inf")
        corridor_half_width = self.robot_half_width + self.corridor_margin
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        best = float("inf")
        for x, y in self.scan_points:
            longitudinal = cos_h * x + sin_h * y
            lateral = -sin_h * x + cos_h * y
            if longitudinal > 0.0 and abs(lateral) <= corridor_half_width:
                best = min(best, longitudinal)
        return best

    def motion_position(self, map_pose):
        if self.odom is not None:
            position = self.odom.pose.pose.position
            return position.x, position.y
        return map_pose[:2]

    def motion_yaw(self, map_pose):
        if self.odom is not None:
            return yaw_from_quaternion(self.odom.pose.pose.orientation)
        return map_pose[2]

    def reset_spin_monitor(self):
        self.spin_start_time = None
        self.spin_start_position = None
        self.spin_last_yaw = None
        self.spin_accumulated_angle = 0.0

    def rotation_loop_detected(self, pose):
        now = rospy.Time.now()
        position = self.motion_position(pose)
        yaw = self.motion_yaw(pose)
        if self.spin_start_time is None:
            if abs(self.last_command.angular.z) < 0.55:
                return False
            self.spin_start_time = now
            self.spin_start_position = position
            self.spin_last_yaw = yaw
            return False

        self.spin_accumulated_angle += abs(angle_diff(yaw, self.spin_last_yaw))
        self.spin_last_yaw = yaw
        moved = math.hypot(
            position[0] - self.spin_start_position[0],
            position[1] - self.spin_start_position[1],
        )
        if moved > self.spin_max_translation:
            self.reset_spin_monitor()
            return False
        return (
            now - self.spin_start_time > self.spin_timeout and
            self.spin_accumulated_angle > self.spin_min_angle
        )

    def choose_local_heading(self, desired_heading):
        samples = max(5, self.local_heading_samples)
        limit = max(0.4, self.local_heading_limit)
        candidates = [0.0, clamp(desired_heading, -limit, limit)]
        for index in range(samples):
            ratio = index / float(max(1, samples - 1))
            candidates.append(-limit + 2.0 * limit * ratio)

        best_heading = 0.0
        best_clearance = 0.0
        best_score = -float("inf")
        fallback_heading = 0.0
        fallback_clearance = 0.0

        for heading in candidates:
            clearance = self.heading_clearance(heading)
            if not math.isfinite(clearance):
                clearance = self.local_clear_distance
            clearance_norm = clamp(clearance / max(0.1, self.local_clear_distance), 0.0, 1.0)
            if clearance > fallback_clearance:
                fallback_clearance = clearance
                fallback_heading = heading

            safe = clearance > max(
                self.front_stop_distance + self.local_stop_margin,
                self.robot_front_radius + self.corridor_margin,
            )
            tracking = math.cos(angle_diff(heading, desired_heading))
            forward = math.cos(heading)
            score = (
                self.heading_tracking_weight * tracking +
                self.obstacle_clearance_weight * clearance_norm +
                self.forward_heading_weight * forward
            )
            if not safe:
                score -= 4.0
            if abs(heading) > 1.1:
                score -= 0.25 * (abs(heading) - 1.1)
            if score > best_score:
                best_score = score
                best_heading = heading
                best_clearance = clearance

        if best_clearance <= self.front_stop_distance + self.local_stop_margin:
            return fallback_heading, fallback_clearance, False
        return best_heading, best_clearance, True

    def world_to_cell(self, x, y):
        info = self.map_msg.info
        return int((x - info.origin.position.x) / info.resolution), int((y - info.origin.position.y) / info.resolution)

    def cell_to_world(self, cell):
        gx, gy = cell
        info = self.map_msg.info
        return (
            info.origin.position.x + (gx + 0.5) * info.resolution,
            info.origin.position.y + (gy + 0.5) * info.resolution,
        )

    def inside(self, gx, gy):
        return 0 <= gx < self.map_msg.info.width and 0 <= gy < self.map_msg.info.height

    def index(self, cell):
        return cell[1] * self.map_msg.info.width + cell[0]

    def build_traversable(self):
        width = self.map_msg.info.width
        height = self.map_msg.info.height
        data = self.map_msg.data
        traversable = [False] * len(data)
        obstacles = []
        for index, value in enumerate(data):
            if 0 <= value <= self.free_threshold:
                traversable[index] = True
            if value >= self.occupied_threshold:
                obstacles.append((index % width, index // width))

        radius = int(math.ceil(self.inflate_radius / max(self.map_msg.info.resolution, 0.01)))
        if radius <= 0:
            return traversable
        for ox, oy in obstacles:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if self.inside(nx, ny):
                        traversable[ny * width + nx] = False
        return traversable

    def nearest_traversable(self, start, traversable, max_radius=12):
        sx, sy = start
        for radius in range(max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cell = sx + dx, sy + dy
                    if self.inside(*cell) and traversable[self.index(cell)]:
                        return cell
        return None

    def bfs(self, start, traversable):
        width = self.map_msg.info.width
        parent = {}
        distance = {start: 0}
        queue = collections.deque([start])
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        expansions = 0
        while queue and expansions < self.max_bfs_cells:
            cell = queue.popleft()
            expansions += 1
            for dx, dy in neighbors:
                nxt = cell[0] + dx, cell[1] + dy
                if not self.inside(*nxt) or nxt in distance:
                    continue
                if not traversable[nxt[1] * width + nxt[0]]:
                    continue
                if dx and dy:
                    side_a = (cell[0] + dx, cell[1])
                    side_b = (cell[0], cell[1] + dy)
                    if not self.inside(*side_a) or not self.inside(*side_b):
                        continue
                    if not traversable[self.index(side_a)] or not traversable[self.index(side_b)]:
                        continue
                parent[nxt] = cell
                distance[nxt] = distance[cell] + (14 if dx and dy else 10)
                queue.append(nxt)
        return parent, distance

    def is_frontier(self, cell, traversable):
        if not traversable[self.index(cell)]:
            return False
        data = self.map_msg.data
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cell[0] + dx, cell[1] + dy
                if self.inside(nx, ny) and data[ny * self.map_msg.info.width + nx] < 0:
                    return True
        return False

    def cluster_frontiers(self, frontier_cells):
        frontier_set = set(frontier_cells)
        clusters = []
        while frontier_set:
            seed = frontier_set.pop()
            cluster = [seed]
            queue = collections.deque([seed])
            while queue:
                cx, cy = queue.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nxt = cx + dx, cy + dy
                        if nxt in frontier_set:
                            frontier_set.remove(nxt)
                            cluster.append(nxt)
                            queue.append(nxt)
            if len(cluster) >= self.min_frontier_cells:
                clusters.append(cluster)
        return clusters

    def reconstruct_path(self, goal, parent):
        path = [goal]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def cluster_target(self, cluster, distance):
        centroid_x = sum(cell[0] for cell in cluster) / float(len(cluster))
        centroid_y = sum(cell[1] for cell in cluster) / float(len(cluster))
        reachable = [cell for cell in cluster if cell in distance]
        if not reachable:
            return None
        return min(
            reachable,
            key=lambda cell: (
                (cell[0] - centroid_x) * (cell[0] - centroid_x) +
                (cell[1] - centroid_y) * (cell[1] - centroid_y)
            ),
        )

    def choose_frontier(self, pose, parent, distance, traversable):
        now = rospy.Time.now()
        for key, expiry in list(self.blacklist.items()):
            if now > expiry:
                del self.blacklist[key]

        reachable = list(distance.keys())
        frontiers = [cell for cell in reachable if self.is_frontier(cell, traversable)]
        clusters = self.cluster_frontiers(frontiers)
        self.publish_frontiers(clusters)
        if not clusters:
            return None, []

        rx, ry, yaw = pose
        best = None
        best_score = -float("inf")
        for cluster in clusters:
            target = self.cluster_target(cluster, distance)
            if target is None:
                continue
            wx, wy = self.cell_to_world(target)
            key = (round(wx, 1), round(wy, 1))
            if key in self.blacklist:
                continue
            path_cost = distance.get(target, 10 ** 9) * self.map_msg.info.resolution / 10.0
            if path_cost < self.min_frontier_path_distance:
                continue
            heading = math.atan2(wy - ry, wx - rx)
            alignment = math.cos(angle_diff(heading, yaw))
            gain = math.sqrt(len(cluster))
            score = (
                self.frontier_gain_weight * gain -
                self.frontier_distance_weight * path_cost +
                self.frontier_heading_weight * alignment +
                self.frontier_far_bonus * min(path_cost, 8.0)
            )
            if self.current_goal_cell is not None:
                goal_delta = math.hypot(
                    target[0] - self.current_goal_cell[0],
                    target[1] - self.current_goal_cell[1],
                ) * self.map_msg.info.resolution
                if goal_delta < 0.7:
                    score += self.frontier_goal_hysteresis
            if score > best_score:
                best_score = score
                best = target

        if best is None:
            return None, []
        return best, self.reconstruct_path(best, parent)

    def choose_patrol_goal(self, pose, parent, distance):
        """Pick a reachable free-space waypoint when frontier clusters are weak."""
        rx, ry, yaw = pose
        data = self.map_msg.data
        width = self.map_msg.info.width
        radius = max(2, self.patrol_unknown_radius)
        stride = max(2, self.patrol_cell_stride)
        best = None
        best_score = -float("inf")

        for cell, grid_cost in distance.items():
            if cell[0] % stride or cell[1] % stride:
                continue
            path_cost = grid_cost * self.map_msg.info.resolution / 10.0
            if path_cost < self.patrol_min_distance or path_cost > self.patrol_max_distance:
                continue

            unknown_gain = 0
            for dy in range(-radius, radius + 1):
                ny = cell[1] + dy
                if ny < 0 or ny >= self.map_msg.info.height:
                    continue
                row = ny * width
                for dx in range(-radius, radius + 1):
                    nx = cell[0] + dx
                    if 0 <= nx < width and data[row + nx] < 0:
                        unknown_gain += 1

            wx, wy = self.cell_to_world(cell)
            key = (round(wx, 1), round(wy, 1))
            if key in self.blacklist:
                continue
            heading = math.atan2(wy - ry, wx - rx)
            alignment = math.cos(angle_diff(heading, yaw))
            repeat_penalty = 0.0
            for old_x, old_y in self.recent_patrol_goals:
                repeat_penalty = max(
                    repeat_penalty,
                    clamp(1.4 - math.hypot(wx - old_x, wy - old_y), 0.0, 1.4),
                )
            score = (
                0.16 * unknown_gain +
                0.34 * min(path_cost, 4.5) +
                0.40 * alignment -
                2.2 * repeat_penalty
            )
            if self.current_goal_cell is not None and self.current_goal_source == "patrol":
                goal_delta = math.hypot(
                    cell[0] - self.current_goal_cell[0],
                    cell[1] - self.current_goal_cell[1],
                ) * self.map_msg.info.resolution
                if goal_delta < 0.7:
                    score += self.frontier_goal_hysteresis
            if score > best_score:
                best_score = score
                best = cell
        return best

    def publish_frontiers(self, clusters):
        msg = MarkerArray()
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = rospy.Time.now()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        for marker_id, cluster in enumerate(clusters[:40]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = clear.header.stamp
            marker.ns = "active_frontiers"
            marker.id = marker_id
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.color.r = 0.05
            marker.color.g = 0.85
            marker.color.b = 1.0
            marker.color.a = 0.9
            for cell in cluster[::max(1, len(cluster) // 120)]:
                point = Point()
                point.x, point.y = self.cell_to_world(cell)
                point.z = 0.03
                marker.points.append(point)
            msg.markers.append(marker)
        self.frontier_pub.publish(msg)

    def publish_path(self, cell_path):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        for cell in cell_path:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = self.cell_to_world(cell)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def publish_target(self, target):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = target[0]
        msg.pose.position.y = target[1]
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    def replan(self, pose):
        if self.map_msg is None:
            return False
        start = self.world_to_cell(pose[0], pose[1])
        traversable = self.build_traversable()
        start = self.nearest_traversable(start, traversable)
        if start is None:
            self.status_pub.publish("no traversable start cell")
            return False
        parent, distance = self.bfs(start, traversable)
        goal, plan = self.choose_frontier(pose, parent, distance, traversable)
        goal_source = "frontier"
        if goal is None or len(plan) < 2:
            goal = self.choose_patrol_goal(pose, parent, distance)
            plan = self.reconstruct_path(goal, parent) if goal is not None else []
            goal_source = "patrol"
        self.last_plan_time = rospy.Time.now()
        if goal is None or len(plan) < 2:
            self.current_plan = []
            self.local_target = None
            self.status_pub.publish("no reachable frontier or patrol waypoint")
            return False
        same_goal = (
            self.current_goal_cell is not None and
            math.hypot(
                goal[0] - self.current_goal_cell[0],
                goal[1] - self.current_goal_cell[1],
            ) * self.map_msg.info.resolution < 0.65
        )
        self.current_goal_cell = goal
        self.current_goal_source = goal_source
        self.current_plan = plan
        if not same_goal:
            self.goal_started = rospy.Time.now()
            self.progress_pose = self.motion_position(pose)
            self.progress_time = rospy.Time.now()
        self.publish_path(plan)
        self.status_pub.publish(
            "%s goal, cells=%d" % (goal_source, len(plan))
        )
        return True

    def select_local_target(self, pose):
        if not self.current_plan:
            return None
        rx, ry = pose[0], pose[1]
        best_index = 0
        best_distance = float("inf")
        for index, cell in enumerate(self.current_plan[:80]):
            wx, wy = self.cell_to_world(cell)
            dist = math.hypot(wx - rx, wy - ry)
            if dist < best_distance:
                best_distance = dist
                best_index = index
        self.current_plan = self.current_plan[best_index:]

        accumulated = 0.0
        last = (rx, ry)
        for cell in self.current_plan:
            point = self.cell_to_world(cell)
            accumulated += math.hypot(point[0] - last[0], point[1] - last[1])
            direct = math.hypot(point[0] - rx, point[1] - ry)
            if accumulated >= self.lookahead_distance and direct >= self.min_local_target_distance:
                return point
            last = point
        final = self.cell_to_world(self.current_plan[-1])
        if math.hypot(final[0] - rx, final[1] - ry) < self.min_local_target_distance:
            self.current_plan = []
            self.last_plan_time = rospy.Time(0)
            return None
        return final

    def start_recovery(self, reason):
        if self.recovery_mode is not None:
            return
        now = rospy.Time.now()
        if now - self.last_recovery_time > self.recovery_reset_seconds:
            self.recovery_attempt = 0
        self.recovery_attempt += 1
        self.last_recovery_time = now
        self.maybe_blacklist_goal()
        self.remember_patrol_goal()
        left = self.sector_min(0.35, 1.45)
        right = self.sector_min(-1.45, -0.35)
        self.recovery_turn_sign = 1.0 if left >= right else -1.0
        if self.recovery_attempt % 3 == 0:
            self.recovery_turn_sign *= -1.0
        self.recovery_mode = "brake"
        self.recovery_until = now + self.brake_seconds
        self.blocked_since = None
        self.reset_spin_monitor()
        self.status_pub.publish("%s: recovery attempt %d" % (reason, self.recovery_attempt))

    def begin_recovery_turn(self, now):
        left = self.sector_min(0.25, 1.65)
        right = self.sector_min(-1.65, -0.25)
        preferred = 1.0 if left >= right else -1.0
        if self.recovery_attempt % 3 != 0:
            self.recovery_turn_sign = preferred
        duration_scale = 1.0 + 0.28 * min(self.recovery_attempt - 1, 3)
        self.recovery_mode = "turn"
        self.recovery_until = now + rospy.Duration(self.turn_seconds.to_sec() * duration_scale)

    def recovery_cmd(self):
        now = rospy.Time.now()
        cmd = Twist()
        if self.recovery_mode is not None:
            self.status_pub.publish(
                "recovery %s, attempt=%d" % (self.recovery_mode, self.recovery_attempt)
            )
        if self.recovery_mode == "brake":
            if now < self.recovery_until:
                return cmd
            if self.rear_min() > self.rear_stop_distance:
                self.recovery_mode = "backup"
                backup_scale = 1.0 + 0.18 * min(self.recovery_attempt - 1, 2)
                self.recovery_until = now + rospy.Duration(self.backup_seconds.to_sec() * backup_scale)
            else:
                self.begin_recovery_turn(now)
        if self.recovery_mode == "backup":
            if now < self.recovery_until:
                if self.rear_min() <= self.rear_stop_distance:
                    self.begin_recovery_turn(now)
                else:
                    cmd.linear.x = -self.backup_speed
                    cmd.angular.z = -0.22 * self.recovery_turn_sign
                    return cmd
            else:
                self.begin_recovery_turn(now)
        if self.recovery_mode == "turn":
            turn_side = self.sector_min(0.20, 1.55) if self.recovery_turn_sign > 0.0 else self.sector_min(-1.55, -0.20)
            other_side = self.sector_min(-1.55, -0.20) if self.recovery_turn_sign > 0.0 else self.sector_min(0.20, 1.55)
            if turn_side < self.side_stop_distance + 0.05 and other_side > turn_side + 0.08:
                self.recovery_turn_sign *= -1.0
            if now < self.recovery_until:
                cmd.angular.z = self.recovery_turn_sign * self.recovery_turn_speed
                return cmd
            self.recovery_mode = "probe"
            self.recovery_until = now + self.probe_seconds
        if self.recovery_mode == "probe":
            if now < self.recovery_until:
                front = self.heading_clearance(0.0)
                if front > self.front_stop_distance + self.local_stop_margin + 0.12:
                    cmd.linear.x = min(0.28, self.min_drive_speed)
                    cmd.angular.z = 0.18 * self.recovery_turn_sign
                else:
                    cmd.angular.z = self.recovery_turn_sign * self.recovery_turn_speed
                return cmd
            self.recovery_mode = None
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.progress_pose = None
            self.progress_time = now
            self.reset_spin_monitor()
        return None

    def maybe_blacklist_goal(self):
        if self.current_goal_cell is None:
            return
        wx, wy = self.cell_to_world(self.current_goal_cell)
        self.blacklist[(round(wx, 1), round(wy, 1))] = rospy.Time.now() + self.blacklist_seconds

    def remember_patrol_goal(self):
        if self.current_goal_cell is None or self.current_goal_source != "patrol":
            return
        point = self.cell_to_world(self.current_goal_cell)
        if not self.recent_patrol_goals or math.hypot(
                point[0] - self.recent_patrol_goals[-1][0],
                point[1] - self.recent_patrol_goals[-1][1]) > 0.3:
            self.recent_patrol_goals.append(point)

    def goal_reached(self, pose):
        if self.current_goal_cell is None:
            return False
        gx, gy = self.cell_to_world(self.current_goal_cell)
        return math.hypot(gx - pose[0], gy - pose[1]) < self.goal_reached_distance

    def progress_failed(self, pose):
        now = rospy.Time.now()
        if self.last_command.linear.x < 0.16:
            self.progress_pose = self.motion_position(pose)
            self.progress_time = now
            return False
        if self.progress_pose is None:
            self.progress_pose = self.motion_position(pose)
            self.progress_time = now
            return False
        current = self.motion_position(pose)
        moved = math.hypot(current[0] - self.progress_pose[0], current[1] - self.progress_pose[1])
        if moved > self.stuck_distance:
            self.progress_pose = current
            self.progress_time = now
            return False
        return now - self.progress_time > self.stuck_timeout

    def drive_command(self, pose, target):
        rx, ry, yaw = pose
        tx, ty = target
        dx, dy = tx - rx, ty - ry
        distance = math.hypot(dx, dy)
        bearing = angle_diff(math.atan2(dy, dx), yaw)
        turn_abs = abs(bearing)
        turn_factor = clamp(1.0 - turn_abs / 1.85, 0.28, 1.0)
        desired_speed = clamp(self.cruise_speed * turn_factor, self.min_drive_speed, self.cruise_speed)
        if distance < self.min_local_target_distance:
            desired_speed = min(desired_speed, max(0.12, distance))
        if turn_abs > 2.25:
            desired_speed = 0.0
        return self.local_avoidance_command(bearing, desired_speed)

    def local_avoidance_command(self, desired_heading, desired_speed):
        heading, clearance, safe = self.choose_local_heading(desired_heading)
        cmd = Twist()
        front_clearance = self.heading_clearance(0.0)
        left = self.sector_min(0.38, 1.40)
        right = self.sector_min(-1.40, -0.38)
        if front_clearance < self.emergency_distance:
            now = rospy.Time.now()
            if self.blocked_since is None:
                self.blocked_since = now
                self.blocked_turn_sign = 1.0 if left >= right else -1.0
            elif now - self.blocked_since > self.blocked_timeout:
                self.start_recovery("emergency path blocked")
            cmd.angular.z = 1.15 * self.blocked_turn_sign
            return cmd
        if not safe:
            now = rospy.Time.now()
            if self.blocked_since is None:
                self.blocked_since = now
                self.blocked_turn_sign = 1.0 if heading >= 0.0 else -1.0
            elif now - self.blocked_since > self.blocked_timeout:
                self.start_recovery("local path blocked")
            turn_speed = clamp(abs(self.turn_gain * heading), 0.85, self.max_angular_speed)
            cmd.angular.z = self.blocked_turn_sign * turn_speed
            return cmd

        braking_clearance = (
            self.robot_front_radius + self.local_stop_margin +
            desired_speed * desired_speed / max(0.2, 2.0 * self.max_linear_deceleration)
        )
        if clearance <= braking_clearance:
            now = rospy.Time.now()
            if self.blocked_since is None:
                self.blocked_since = now
                self.blocked_turn_sign = 1.0 if heading >= 0.0 else -1.0
            cmd.angular.z = self.blocked_turn_sign * clamp(
                abs(self.turn_gain * heading), 0.65, self.max_angular_speed
            )
            if now - self.blocked_since > self.blocked_timeout:
                self.start_recovery("braking path blocked")
            return cmd

        self.blocked_since = None

        clearance_scale = clamp(
            (clearance - self.front_stop_distance) /
            max(0.05, self.front_slow_distance - self.front_stop_distance),
            0.20,
            1.0,
        )
        heading_scale = clamp(math.cos(abs(heading)), 0.22, 1.0)
        cmd.linear.x = clamp(desired_speed * clearance_scale * heading_scale, 0.0, self.cruise_speed)
        if (desired_speed > 0.01 and clearance > self.front_stop_distance + 0.22 and
                cmd.linear.x < self.min_drive_speed * 0.75 and abs(heading) < 1.2):
            cmd.linear.x = self.min_drive_speed * 0.75
        cmd.angular.z = clamp(self.turn_gain * heading, -self.max_angular_speed, self.max_angular_speed)
        if left < self.side_stop_distance and left < right and cmd.angular.z > -0.55:
            cmd.angular.z = min(-0.55, cmd.angular.z)
            cmd.linear.x = min(cmd.linear.x, self.min_drive_speed * 0.65)
        elif right < self.side_stop_distance and right < left and cmd.angular.z < 0.55:
            cmd.angular.z = max(0.55, cmd.angular.z)
            cmd.linear.x = min(cmd.linear.x, self.min_drive_speed * 0.65)
        return cmd

    def open_space_command(self):
        front = self.heading_clearance(0.0)
        left = self.sector_min(0.35, 1.35)
        right = self.sector_min(-1.35, -0.35)
        openness_turn = clamp((left - right) * 0.45, -0.75, 0.75)
        if front < self.front_slow_distance:
            desired_speed = 0.34
        else:
            desired_speed = self.cruise_speed
        return self.local_avoidance_command(openness_turn, desired_speed)

    def update(self, event):
        try:
            self.update_control(event)
        except Exception as exc:
            rospy.logerr_throttle(1.0, "active slam control error: %s", exc)
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.reset_spin_monitor()
            self.publish_cmd(Twist())
            self.status_pub.publish("control error: stopped and retrying")

    def update_control(self, _event):
        recovery = self.recovery_cmd()
        if recovery is not None:
            self.publish_cmd(recovery)
            return

        pose = self.pose()
        if pose is None or self.scan is None or self.map_msg is None:
            self.status_pub.publish("waiting for map, scan, and tf")
            self.publish_cmd(Twist())
            return

        with self.map_lock:
            self.update_navigation(pose)

    def update_navigation(self, pose):

        if rospy.Time.now() - self.last_contact_time < self.contact_hold:
            self.start_recovery("physical contact")
            self.publish_cmd(Twist())
            return

        if self.rotation_loop_detected(pose):
            self.start_recovery("rotation loop")
            self.publish_cmd(Twist())
            return

        now = rospy.Time.now()
        if self.goal_reached(pose):
            reached_source = self.current_goal_source or "exploration"
            self.remember_patrol_goal()
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.status_pub.publish("%s goal reached" % reached_source)

        if self.current_goal_cell is not None and now - self.goal_started > self.goal_timeout:
            self.maybe_blacklist_goal()
            self.remember_patrol_goal()
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.status_pub.publish("frontier timeout: replanning")

        if self.current_goal_cell is not None and self.progress_failed(pose):
            self.maybe_blacklist_goal()
            self.start_recovery("stuck")
            self.publish_cmd(Twist())
            return

        if now - self.last_plan_time > self.replan_interval:
            self.replan(pose)

        target = self.select_local_target(pose)
        if target is None:
            cmd = self.open_space_command()
            self.publish_cmd(cmd)
            if self.recovery_mode is None:
                self.status_pub.publish("active slam fallback %.2f %.2f" % (cmd.linear.x, cmd.angular.z))
            return

        self.publish_target(target)
        cmd = self.drive_command(pose, target)
        self.publish_cmd(cmd)
        if self.recovery_mode is None:
            self.status_pub.publish("active slam driving %.2f %.2f" % (cmd.linear.x, cmd.angular.z))

    def publish_cmd(self, cmd):
        self.last_command = cmd
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("active_slam_explorer")
    ActiveSlamExplorer()
    rospy.spin()
