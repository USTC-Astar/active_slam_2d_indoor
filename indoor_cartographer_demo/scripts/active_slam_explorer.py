#!/usr/bin/env python3
import collections
import heapq
import math
import threading

import rospy
import tf2_ros
from cartographer_ros_msgs.srv import FinishTrajectory
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
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
        self.costmap_pub = rospy.Publisher("/active_slam/costmap", OccupancyGrid, queue_size=1, latch=True)
        self.completed_pub = rospy.Publisher("/active_slam/completed", Bool, queue_size=1, latch=True)

        self.map_lock = threading.RLock()
        self.last_known_cell_count = 0
        self.current_known_cell_count = 0
        self.last_map_growth_time = rospy.Time.now()
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
        self.current_goal_world = None
        self.current_goal_source = None
        self.local_target = None
        self.blacklist = {}
        self.recent_patrol_goals = collections.deque(maxlen=10)
        self.recent_closure_goals = collections.deque(maxlen=16)
        self.visited_positions = collections.deque(maxlen=120)

        self.free_threshold = int(rospy.get_param("~free_threshold", 35))
        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 58))
        self.inflate_radius = rospy.get_param("~inflate_radius", 0.22)
        self.costmap_radius = rospy.get_param("~costmap_radius", 0.52)
        self.costmap_weight = rospy.get_param("~costmap_weight", 0.10)
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
        self.patrol_min_visible_unknown = int(rospy.get_param("~patrol_min_visible_unknown", 6))
        self.patrol_cell_stride = int(rospy.get_param("~patrol_cell_stride", 4))
        self.closure_start_ratio = rospy.get_param("~closure_start_ratio", 0.72)
        self.closure_min_runtime = rospy.Duration(rospy.get_param("~closure_min_runtime", 90.0))
        self.closure_unknown_radius = int(rospy.get_param("~closure_unknown_radius", 24))
        self.closure_cell_stride = int(rospy.get_param("~closure_cell_stride", 4))
        self.closure_min_visible_unknown = int(rospy.get_param("~closure_min_visible_unknown", 10))
        self.closure_min_distance = rospy.get_param("~closure_min_distance", 0.35)
        self.closure_max_distance = rospy.get_param("~closure_max_distance", 8.0)
        self.closure_bbox_margin = int(rospy.get_param("~closure_bbox_margin", 4))
        self.closure_bounds = (
            rospy.get_param("~closure_bounds_min_x", float("nan")),
            rospy.get_param("~closure_bounds_min_y", float("nan")),
            rospy.get_param("~closure_bounds_max_x", float("nan")),
            rospy.get_param("~closure_bounds_max_y", float("nan")),
        )
        self.completion_no_frontier_cycles = int(rospy.get_param("~completion_no_frontier_cycles", 6))
        self.completion_stable_seconds = rospy.Duration(rospy.get_param("~completion_stable_seconds", 12.0))
        self.completion_min_runtime = rospy.Duration(rospy.get_param("~completion_min_runtime", 45.0))
        self.completion_min_travel_distance = rospy.get_param("~completion_min_travel_distance", 24.0)
        self.completion_min_known_cells = int(rospy.get_param("~completion_min_known_cells", 0))
        self.completion_max_interior_unknown = int(
            rospy.get_param("~completion_max_interior_unknown", 10 ** 9)
        )
        self.completion_known_growth_cells = int(rospy.get_param("~completion_known_growth_cells", 20))
        self.visit_record_distance = rospy.get_param("~visit_record_distance", 0.75)
        self.home_reached_distance = rospy.get_param("~home_reached_distance", 0.48)
        self.home_timeout = rospy.Duration(rospy.get_param("~home_timeout", 90.0))
        self.max_bfs_cells = int(rospy.get_param("~max_bfs_cells", 90000))
        self.replan_interval = rospy.Duration(rospy.get_param("~replan_interval", 2.8))
        self.goal_timeout = rospy.Duration(rospy.get_param("~goal_timeout", 16.0))
        self.goal_commitment = rospy.Duration(rospy.get_param("~goal_commitment", 8.0))
        self.blacklist_seconds = rospy.Duration(rospy.get_param("~blacklist_seconds", 18.0))
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.35)
        self.min_local_target_distance = rospy.get_param("~min_local_target_distance", 0.16)
        self.goal_reached_distance = rospy.get_param("~goal_reached_distance", 0.26)
        self.frontier_probe_distance = rospy.get_param("~frontier_probe_distance", 1.4)
        self.frontier_probe_seconds = rospy.Duration(rospy.get_param("~frontier_probe_seconds", 5.0))
        self.frontier_probe_speed = rospy.get_param("~frontier_probe_speed", 0.42)
        self.frontier_probe_unknown_radius = int(rospy.get_param("~frontier_probe_unknown_radius", 18))
        self.finish_cartographer_on_complete = rospy.get_param("~finish_cartographer_on_complete", True)

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
        self.robot_half_width = rospy.get_param("~robot_half_width", 0.15)
        self.robot_front_radius = rospy.get_param("~robot_front_radius", 0.16)
        self.laser_offset_x = rospy.get_param("~laser_offset_x", 0.065)
        self.scan_self_filter_radius = rospy.get_param("~scan_self_filter_radius", 0.13)
        self.corridor_margin = rospy.get_param("~corridor_margin", 0.04)
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
        self.dead_end_front_distance = rospy.get_param("~dead_end_front_distance", 0.58)
        self.dead_end_side_distance = rospy.get_param("~dead_end_side_distance", 0.52)
        self.dead_end_backup_extra = rospy.Duration(rospy.get_param("~dead_end_backup_extra", 0.55))
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
        self.recovery_reason = ""
        self.last_recovery_time = rospy.Time(0)
        self.blocked_since = None
        self.blocked_turn_sign = 1.0
        self.last_command = Twist()
        self.spin_start_time = None
        self.spin_start_position = None
        self.spin_last_yaw = None
        self.spin_accumulated_angle = 0.0
        self.home_position = None
        self.home_yaw = 0.0
        self.mapping_start_time = None
        self.no_frontier_cycles = 0
        self.last_frontier_cell_count = 0
        self.returning_home = False
        self.mapping_completed = False
        self.total_travel_distance = 0.0
        self.travel_reference = None
        self.last_visit_position = None
        self.frontier_probe_active = False
        self.frontier_probe_start = None
        self.frontier_probe_heading = 0.0
        self.frontier_probe_until = rospy.Time(0)
        self.finish_trajectory_started = False
        self.last_interior_unknown_count = 10 ** 9
        self.completed_pub.publish(False)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.control_rate)), self.update)

    def map_callback(self, msg):
        with self.map_lock:
            self.map_msg = msg
            known_cells = sum(1 for value in msg.data if value >= 0)
            self.current_known_cell_count = known_cells
            growth_threshold = getattr(self, "completion_known_growth_cells", 20)
            if known_cells > self.last_known_cell_count + growth_threshold:
                self.last_map_growth_time = rospy.Time.now()
            self.last_known_cell_count = max(self.last_known_cell_count, known_cells)

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

    def build_navigation_grid(self):
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

        resolution = max(self.map_msg.info.resolution, 0.01)
        lethal_cells = self.inflate_radius / resolution
        cost_cells = max(lethal_cells + 1.0, self.costmap_radius / resolution)
        obstacle_distance = [10 ** 9] * len(data)
        queue = collections.deque()
        for ox, oy in obstacles:
            index = oy * width + ox
            obstacle_distance[index] = 0
            queue.append((ox, oy))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            cx, cy = queue.popleft()
            index = cy * width + cx
            distance_cells = obstacle_distance[index]
            if distance_cells >= cost_cells:
                continue
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not self.inside(nx, ny):
                    continue
                nxt = ny * width + nx
                candidate = distance_cells + 1
                if candidate < obstacle_distance[nxt] and candidate <= cost_cells:
                    obstacle_distance[nxt] = candidate
                    queue.append((nx, ny))

        costs = [0] * len(data)
        visualization = [-1] * len(data)
        span = max(1.0, cost_cells - lethal_cells)
        for index, value in enumerate(data):
            if value < 0:
                continue
            distance_cells = obstacle_distance[index]
            if value >= self.occupied_threshold or distance_cells <= lethal_cells:
                traversable[index] = False
                costs[index] = 100
                visualization[index] = 100
            elif distance_cells < cost_cells:
                ratio = (cost_cells - distance_cells) / span
                costs[index] = int(clamp(99.0 * ratio * ratio, 1.0, 99.0))
                visualization[index] = costs[index]
            else:
                visualization[index] = 0

        costmap = OccupancyGrid()
        costmap.header = self.map_msg.header
        costmap.header.stamp = rospy.Time.now()
        costmap.info = self.map_msg.info
        costmap.data = visualization
        self.costmap_pub.publish(costmap)
        return traversable, costs

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

    def dijkstra(self, start, traversable, costs):
        width = self.map_msg.info.width
        parent = {}
        distance = {start: 0}
        travel_distance = {start: 0}
        queue = [(0, start)]
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        expansions = 0
        while queue and expansions < self.max_bfs_cells:
            current_cost, cell = heapq.heappop(queue)
            if current_cost != distance.get(cell):
                continue
            expansions += 1
            for dx, dy in neighbors:
                nxt = cell[0] + dx, cell[1] + dy
                if not self.inside(*nxt):
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
                step = 14 if dx and dy else 10
                penalty = int(costs[nxt[1] * width + nxt[0]] * self.costmap_weight)
                candidate = current_cost + step + penalty
                if candidate >= distance.get(nxt, 10 ** 18):
                    continue
                parent[nxt] = cell
                distance[nxt] = candidate
                travel_distance[nxt] = travel_distance[cell] + step
                heapq.heappush(queue, (candidate, nxt))
        return parent, distance, travel_distance

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

    def choose_frontier(self, pose, parent, distance, travel_distance, traversable):
        now = rospy.Time.now()
        for key, expiry in list(self.blacklist.items()):
            if now > expiry:
                del self.blacklist[key]

        reachable = list(distance.keys())
        frontiers = [cell for cell in reachable if self.is_frontier(cell, traversable)]
        clusters = self.cluster_frontiers(frontiers)
        self.last_frontier_cell_count = sum(len(cluster) for cluster in clusters)
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
            path_cost = travel_distance.get(target, 10 ** 9) * self.map_msg.info.resolution / 10.0
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

    def choose_patrol_goal(self, pose, parent, distance, travel_distance):
        """Pick a reachable free-space waypoint when frontier clusters are weak."""
        rx, ry, yaw = pose
        data = self.map_msg.data
        width = self.map_msg.info.width
        radius = max(2, self.patrol_unknown_radius)
        stride = max(2, self.patrol_cell_stride)
        best = None
        best_score = -float("inf")
        integral_width = width + 1
        unknown_integral = [0] * ((self.map_msg.info.height + 1) * integral_width)
        for y in range(self.map_msg.info.height):
            row_sum = 0
            source_row = y * width
            integral_row = (y + 1) * integral_width
            previous_row = y * integral_width
            for x in range(width):
                if data[source_row + x] < 0:
                    row_sum += 1
                unknown_integral[integral_row + x + 1] = (
                    unknown_integral[previous_row + x + 1] + row_sum
                )

        for cell in distance:
            if cell[0] % stride or cell[1] % stride:
                continue
            path_cost = travel_distance[cell] * self.map_msg.info.resolution / 10.0
            if path_cost < self.patrol_min_distance or path_cost > self.patrol_max_distance:
                continue

            x0 = max(0, cell[0] - radius)
            y0 = max(0, cell[1] - radius)
            x1 = min(width - 1, cell[0] + radius)
            y1 = min(self.map_msg.info.height - 1, cell[1] + radius)
            unknown_gain = (
                unknown_integral[(y1 + 1) * integral_width + x1 + 1] -
                unknown_integral[y0 * integral_width + x1 + 1] -
                unknown_integral[(y1 + 1) * integral_width + x0] +
                unknown_integral[y0 * integral_width + x0]
            )
            if unknown_gain < self.patrol_min_visible_unknown:
                continue

            visible_unknown = self.visible_unknown_gain(cell, radius)
            if visible_unknown < self.patrol_min_visible_unknown:
                continue

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
            nearest_visit = 2.5
            for old_x, old_y in self.visited_positions:
                nearest_visit = min(nearest_visit, math.hypot(wx - old_x, wy - old_y))
            score = (
                0.72 * math.sqrt(visible_unknown) +
                0.38 * min(path_cost, 5.5) +
                0.40 * alignment -
                2.8 * repeat_penalty +
                0.85 * min(nearest_visit, 2.5)
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

    def visible_unknown_gain(self, cell, radius):
        """Count unknown cells visible from a patrol cell without looking through walls."""
        return self.visible_unknown_cells(cell, radius, None)

    def visible_unknown_cells(self, cell, radius, bounds):
        data = self.map_msg.data
        width = self.map_msg.info.width
        visible = set()
        ray_count = max(24, radius * 2)
        for ray in range(ray_count):
            angle = 2.0 * math.pi * ray / ray_count
            last = None
            for step in range(1, radius + 1):
                sample = (
                    cell[0] + int(round(math.cos(angle) * step)),
                    cell[1] + int(round(math.sin(angle) * step)),
                )
                if sample == last:
                    continue
                last = sample
                if not self.inside(*sample):
                    break
                value = data[sample[1] * width + sample[0]]
                if value >= self.occupied_threshold:
                    break
                inside_bounds = (
                    bounds is None or
                    (bounds[0] <= sample[0] <= bounds[2] and
                     bounds[1] <= sample[1] <= bounds[3])
                )
                if value < 0 and inside_bounds:
                    visible.add(sample)
        return len(visible)

    def interior_unknown_region(self):
        data = self.map_msg.data
        width = self.map_msg.info.width
        if self.home_position is not None and all(math.isfinite(value) for value in self.closure_bounds):
            min_cell = self.world_to_cell(
                self.home_position[0] + self.closure_bounds[0],
                self.home_position[1] + self.closure_bounds[1],
            )
            max_cell = self.world_to_cell(
                self.home_position[0] + self.closure_bounds[2],
                self.home_position[1] + self.closure_bounds[3],
            )
            min_x = max(0, min(min_cell[0], max_cell[0]))
            max_x = min(width - 1, max(min_cell[0], max_cell[0]))
            min_y = max(0, min(min_cell[1], max_cell[1]))
            max_y = min(self.map_msg.info.height - 1, max(min_cell[1], max_cell[1]))
            if min_x < max_x and min_y < max_y:
                bounds = (min_x, min_y, max_x, max_y)
                self.last_interior_unknown_count = sum(
                    1
                    for y in range(min_y, max_y + 1)
                    for x in range(min_x, max_x + 1)
                    if data[y * width + x] < 0
                )
                return bounds

        occupied = [index for index, value in enumerate(data) if value >= self.occupied_threshold]
        if len(occupied) < 40:
            self.last_interior_unknown_count = 10 ** 9
            return None
        margin = max(1, self.closure_bbox_margin)
        min_x = min(index % width for index in occupied) + margin
        max_x = max(index % width for index in occupied) - margin
        min_y = min(index // width for index in occupied) + margin
        max_y = max(index // width for index in occupied) - margin
        if min_x >= max_x or min_y >= max_y:
            self.last_interior_unknown_count = 10 ** 9
            return None
        bounds = (min_x, min_y, max_x, max_y)
        self.last_interior_unknown_count = sum(
            1
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if data[y * width + x] < 0
        )
        return bounds

    def choose_closure_goal(self, pose, distance, travel_distance):
        """Choose a new free-space viewpoint for interior lidar shadow gaps."""
        if self.completion_min_known_cells <= 0:
            return None
        if (self.mapping_start_time is None or
                rospy.Time.now() - self.mapping_start_time < self.closure_min_runtime):
            return None
        if self.current_known_cell_count < self.completion_min_known_cells * self.closure_start_ratio:
            return None
        bounds = self.interior_unknown_region()
        if bounds is None or self.last_interior_unknown_count <= self.completion_max_interior_unknown:
            return None

        rx, ry, yaw = pose
        radius = max(4, self.closure_unknown_radius)
        stride = max(2, self.closure_cell_stride)
        best = None
        best_score = -float("inf")
        for cell in distance:
            if cell[0] % stride or cell[1] % stride:
                continue
            path_cost = travel_distance[cell] * self.map_msg.info.resolution / 10.0
            if path_cost < self.closure_min_distance or path_cost > self.closure_max_distance:
                continue
            visible_unknown = self.visible_unknown_cells(cell, radius, bounds)
            if visible_unknown < self.closure_min_visible_unknown:
                continue
            wx, wy = self.cell_to_world(cell)
            key = (round(wx, 1), round(wy, 1))
            if key in self.blacklist:
                continue
            repeat_penalty = 0.0
            for old_x, old_y in self.recent_closure_goals:
                repeat_penalty = max(
                    repeat_penalty,
                    clamp(1.2 - math.hypot(wx - old_x, wy - old_y), 0.0, 1.2),
                )
            heading = math.atan2(wy - ry, wx - rx)
            alignment = math.cos(angle_diff(heading, yaw))
            score = (
                1.05 * math.sqrt(visible_unknown) +
                0.16 * min(path_cost, 6.0) +
                0.22 * alignment -
                3.4 * repeat_penalty
            )
            if score > best_score:
                best_score = score
                best = cell
        return best

    def choose_home_goal(self, parent, distance, traversable):
        if self.home_position is None:
            return None, []
        home = self.world_to_cell(self.home_position[0], self.home_position[1])
        home = self.nearest_traversable(home, traversable, max_radius=20)
        if home is None or home not in distance:
            return None, []
        return home, self.reconstruct_path(home, parent)

    def frontier_unknown_heading(self, cell, fallback_yaw):
        radius = max(2, self.frontier_probe_unknown_radius)
        vector_x = 0.0
        vector_y = 0.0
        width = self.map_msg.info.width
        data = self.map_msg.data
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cell[0] + dx, cell[1] + dy
                if not self.inside(nx, ny) or data[ny * width + nx] >= 0:
                    continue
                weight = 1.0 / max(1.0, math.hypot(dx, dy))
                vector_x += dx * weight
                vector_y += dy * weight
        if math.hypot(vector_x, vector_y) < 0.5:
            return fallback_yaw
        return math.atan2(vector_y, vector_x)

    def unknown_cells_near(self, cell, radius=None):
        radius = radius or self.frontier_probe_unknown_radius
        count = 0
        width = self.map_msg.info.width
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cell[0] + dx, cell[1] + dy
                if self.inside(nx, ny) and self.map_msg.data[ny * width + nx] < 0:
                    count += 1
        return count

    def start_frontier_probe(self, pose, goal_cell):
        self.frontier_probe_active = True
        self.frontier_probe_start = self.motion_position(pose)
        self.frontier_probe_heading = self.frontier_unknown_heading(goal_cell, pose[2])
        self.frontier_probe_until = rospy.Time.now() + self.frontier_probe_seconds
        self.current_plan = []
        self.current_goal_cell = None
        self.current_goal_world = None
        self.current_goal_source = None
        self.last_plan_time = rospy.Time.now()
        self.reset_spin_monitor()

    def frontier_probe_command(self, pose):
        if not self.frontier_probe_active:
            return None
        now = rospy.Time.now()
        current = self.motion_position(pose)
        moved = math.hypot(
            current[0] - self.frontier_probe_start[0],
            current[1] - self.frontier_probe_start[1],
        )
        desired_heading = angle_diff(self.frontier_probe_heading, pose[2])
        front = self.heading_clearance(0.0)
        if (moved >= self.frontier_probe_distance or now >= self.frontier_probe_until or
                (abs(desired_heading) < 0.45 and
                 front <= self.front_stop_distance + self.local_stop_margin + 0.04)):
            self.frontier_probe_active = False
            self.last_plan_time = rospy.Time(0)
            self.status_pub.publish("frontier deep probe complete: replanning")
            return None
        return self.local_avoidance_command(desired_heading, self.frontier_probe_speed)

    def finish_cartographer_trajectory(self):
        if not self.finish_cartographer_on_complete or self.finish_trajectory_started:
            return
        self.finish_trajectory_started = True

        def call_service():
            try:
                rospy.wait_for_service("/finish_trajectory", timeout=2.0)
                response = rospy.ServiceProxy("/finish_trajectory", FinishTrajectory)(0)
                rospy.loginfo("Cartographer final optimization: %s", response.status.message)
            except Exception as exc:
                rospy.logwarn("Could not finish Cartographer trajectory: %s", exc)

        thread = threading.Thread(target=call_service)
        thread.daemon = True
        thread.start()

    def completion_ready(self, now):
        if self.mapping_start_time is None:
            return False
        coverage_closed = (
            self.current_known_cell_count >= self.completion_min_known_cells and
            self.last_interior_unknown_count <= self.completion_max_interior_unknown
        )
        return (
            (self.no_frontier_cycles >= self.completion_no_frontier_cycles or coverage_closed) and
            now - self.mapping_start_time >= self.completion_min_runtime and
            now - self.last_map_growth_time >= self.completion_stable_seconds and
            self.total_travel_distance >= self.completion_min_travel_distance and
            coverage_closed
        )

    def update_coverage_progress(self, pose):
        position = self.motion_position(pose)
        if self.travel_reference is not None:
            step = math.hypot(
                position[0] - self.travel_reference[0],
                position[1] - self.travel_reference[1],
            )
            if step < 0.8:
                self.total_travel_distance += step
        self.travel_reference = position
        if (self.last_visit_position is None or math.hypot(
                position[0] - self.last_visit_position[0],
                position[1] - self.last_visit_position[1]) >= self.visit_record_distance):
            self.visited_positions.append(position)
            self.last_visit_position = position

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
        traversable, costs = self.build_navigation_grid()
        start = self.nearest_traversable(start, traversable)
        if start is None:
            self.status_pub.publish("no traversable start cell")
            return False
        parent, distance, travel_distance = self.dijkstra(start, traversable, costs)
        now = rospy.Time.now()
        if self.returning_home:
            self.interior_unknown_region()
            if (self.current_known_cell_count < self.completion_min_known_cells or
                    self.last_interior_unknown_count > self.completion_max_interior_unknown):
                self.returning_home = False
                self.current_goal_cell = None
                self.current_goal_world = None
                self.current_goal_source = None
                self.status_pub.publish("map gap reopened: resuming closure scan")
        if self.returning_home:
            goal, plan = self.choose_home_goal(parent, distance, traversable)
            goal_source = "home"
        else:
            goal = None
            plan = []
            goal_source = self.current_goal_source
            closure_priority = (
                self.current_known_cell_count >= self.completion_min_known_cells and
                self.mapping_start_time is not None and
                now - self.mapping_start_time >= self.closure_min_runtime
            )
            if closure_priority:
                self.interior_unknown_region()
            if (self.current_goal_world is not None and
                    self.current_goal_source in ("frontier", "closure", "patrol") and
                    now - self.goal_started < self.goal_commitment):
                committed = self.world_to_cell(*self.current_goal_world)
                committed = self.nearest_traversable(committed, traversable, max_radius=5)
                if committed is not None and committed in distance:
                    goal = committed
                    plan = self.reconstruct_path(goal, parent)

            if (goal is None or len(plan) < 2) and closure_priority:
                if self.completion_ready(now):
                    self.returning_home = True
                    goal, plan = self.choose_home_goal(parent, distance, traversable)
                    goal_source = "home"
                    self.status_pub.publish("closure complete: returning home")
                else:
                    goal = self.choose_closure_goal(pose, distance, travel_distance)
                    plan = self.reconstruct_path(goal, parent) if goal is not None else []
                    goal_source = "closure"

            if goal is None or len(plan) < 2:
                goal, plan = self.choose_frontier(pose, parent, distance, travel_distance, traversable)
                goal_source = "frontier"
                if goal is not None and len(plan) >= 2:
                    self.no_frontier_cycles = 0
                else:
                    self.no_frontier_cycles += 1

            if (goal is None or len(plan) < 2) and not closure_priority:
                goal = self.choose_closure_goal(pose, distance, travel_distance)
                plan = self.reconstruct_path(goal, parent) if goal is not None else []
                goal_source = "closure"

            if goal is None or len(plan) < 2:
                goal = self.choose_patrol_goal(pose, parent, distance, travel_distance)
                plan = self.reconstruct_path(goal, parent) if goal is not None else []
                goal_source = "patrol"

            if goal_source != "frontier" and self.completion_ready(now):
                self.returning_home = True
                goal, plan = self.choose_home_goal(parent, distance, traversable)
                goal_source = "home"
                self.status_pub.publish("exploration complete: returning home")
        self.last_plan_time = rospy.Time.now()
        if goal is None or len(plan) < 2:
            self.current_plan = []
            self.local_target = None
            if self.returning_home:
                self.status_pub.publish("returning home: waiting for a reachable path")
            else:
                self.status_pub.publish("no reachable frontier or patrol waypoint")
            return False
        same_goal = (
            self.current_goal_world is not None and
            self.current_goal_source == goal_source and
            math.hypot(
                self.cell_to_world(goal)[0] - self.current_goal_world[0],
                self.cell_to_world(goal)[1] - self.current_goal_world[1],
            ) < 0.65
        )
        self.current_goal_cell = goal
        self.current_goal_world = self.cell_to_world(goal)
        self.current_goal_source = goal_source
        self.current_plan = plan
        if not same_goal:
            self.goal_started = rospy.Time.now()
            self.progress_pose = self.motion_position(pose)
            self.progress_time = rospy.Time.now()
        self.publish_path(plan)
        status = "%s goal, cells=%d" % (goal_source, len(plan))
        if goal_source == "closure":
            status += ", interior_unknown=%d" % self.last_interior_unknown_count
        self.status_pub.publish(status)
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
        self.recovery_reason = reason
        self.frontier_probe_active = False
        self.last_recovery_time = now
        self.maybe_blacklist_goal()
        self.remember_patrol_goal()
        self.remember_closure_goal()
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
                backup_duration = self.backup_seconds.to_sec() * backup_scale
                if "dead end" in self.recovery_reason:
                    backup_duration += self.dead_end_backup_extra.to_sec()
                self.recovery_until = now + rospy.Duration(backup_duration)
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
            self.current_goal_world = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.progress_pose = None
            self.progress_time = now
            self.reset_spin_monitor()
        return None

    def maybe_blacklist_goal(self):
        if self.current_goal_cell is None or self.current_goal_source == "home":
            return
        wx, wy = self.current_goal_world or self.cell_to_world(self.current_goal_cell)
        self.blacklist[(round(wx, 1), round(wy, 1))] = rospy.Time.now() + self.blacklist_seconds

    def remember_patrol_goal(self):
        if self.current_goal_cell is None or self.current_goal_source != "patrol":
            return
        point = self.current_goal_world or self.cell_to_world(self.current_goal_cell)
        if not self.recent_patrol_goals or math.hypot(
                point[0] - self.recent_patrol_goals[-1][0],
                point[1] - self.recent_patrol_goals[-1][1]) > 0.3:
            self.recent_patrol_goals.append(point)

    def remember_closure_goal(self):
        if self.current_goal_cell is None or self.current_goal_source != "closure":
            return
        point = self.current_goal_world or self.cell_to_world(self.current_goal_cell)
        if not self.recent_closure_goals or math.hypot(
                point[0] - self.recent_closure_goals[-1][0],
                point[1] - self.recent_closure_goals[-1][1]) > 0.25:
            self.recent_closure_goals.append(point)

    def goal_reached(self, pose):
        if self.current_goal_cell is None:
            return False
        gx, gy = self.current_goal_world or self.cell_to_world(self.current_goal_cell)
        threshold = self.home_reached_distance if self.current_goal_source == "home" else self.goal_reached_distance
        return math.hypot(gx - pose[0], gy - pose[1]) < threshold

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
        if turn_abs > 2.15:
            front = self.heading_clearance(0.0)
            left = self.sector_min(0.38, 1.40)
            right = self.sector_min(-1.40, -0.38)
            if front < self.dead_end_front_distance or min(left, right) < self.dead_end_side_distance:
                self.start_recovery("dead end turn-around")
                return Twist()
        if turn_abs > 2.25:
            desired_speed = 0.0
        return self.local_avoidance_command(bearing, desired_speed)

    def local_avoidance_command(self, desired_heading, desired_speed):
        heading, clearance, safe = self.choose_local_heading(desired_heading)
        cmd = Twist()
        front_clearance = self.heading_clearance(0.0)
        left = self.sector_min(0.38, 1.40)
        right = self.sector_min(-1.40, -0.38)
        if (front_clearance < self.dead_end_front_distance and
                left < self.dead_end_side_distance and right < self.dead_end_side_distance):
            self.start_recovery("dead end blocked")
            return cmd
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
            self.current_goal_world = None
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

        self.update_coverage_progress(pose)

        if self.home_position is None:
            self.home_position = pose[:2]
            self.home_yaw = pose[2]
            self.mapping_start_time = rospy.Time.now()
            self.last_map_growth_time = rospy.Time.now()

        if self.mapping_completed:
            self.publish_cmd(Twist())
            self.status_pub.publish("mapping completed: home reached")
            return

        if rospy.Time.now() - self.last_contact_time < self.contact_hold:
            self.start_recovery("physical contact")
            self.publish_cmd(Twist())
            return

        if self.rotation_loop_detected(pose):
            self.start_recovery("rotation loop")
            self.publish_cmd(Twist())
            return

        probe_cmd = self.frontier_probe_command(pose)
        if probe_cmd is not None:
            self.publish_cmd(probe_cmd)
            self.status_pub.publish(
                "frontier deep probe %.2f %.2f" % (probe_cmd.linear.x, probe_cmd.angular.z)
            )
            return

        now = rospy.Time.now()
        if self.goal_reached(pose):
            reached_source = self.current_goal_source or "exploration"
            if reached_source == "home":
                self.current_plan = []
                self.current_goal_cell = None
                self.current_goal_world = None
                self.current_goal_source = None
                self.mapping_completed = True
                self.completed_pub.publish(True)
                self.publish_cmd(Twist())
                self.status_pub.publish("mapping completed: home reached")
                self.finish_cartographer_trajectory()
                return
            should_probe = (
                reached_source == "frontier" or
                (reached_source == "patrol" and self.current_goal_cell is not None and
                 self.unknown_cells_near(self.current_goal_cell) >= 5)
            )
            if should_probe and self.current_goal_cell is not None:
                goal_cell = self.current_goal_cell
                self.start_frontier_probe(pose, goal_cell)
                self.publish_cmd(Twist())
                self.status_pub.publish("%s reached: starting deep probe" % reached_source)
                return
            self.remember_patrol_goal()
            self.remember_closure_goal()
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_world = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.status_pub.publish("%s goal reached" % reached_source)

        active_timeout = self.home_timeout if self.current_goal_source == "home" else self.goal_timeout
        if self.current_goal_cell is not None and now - self.goal_started > active_timeout:
            self.maybe_blacklist_goal()
            self.remember_patrol_goal()
            self.remember_closure_goal()
            self.current_plan = []
            self.current_goal_cell = None
            self.current_goal_world = None
            self.current_goal_source = None
            self.last_plan_time = rospy.Time(0)
            self.status_pub.publish("frontier timeout: replanning")

        if self.current_goal_cell is not None and self.progress_failed(pose):
            self.maybe_blacklist_goal()
            self.start_recovery("stuck")
            self.publish_cmd(Twist())
            return

        if now - self.last_plan_time > self.replan_interval:
            self.publish_cmd(Twist())
            self.replan(pose)

        target = self.select_local_target(pose)
        if target is None:
            if self.returning_home:
                self.publish_cmd(Twist())
                self.status_pub.publish("returning home: replanning")
                return
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
