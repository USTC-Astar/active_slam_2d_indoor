#!/usr/bin/env python3
import heapq
import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


DEFAULT_WAYPOINTS = [
    (-6.0, -4.8),
    (-3.0, -4.3),
    (-0.6, -2.4),
    (1.95, -2.4),
    (5.8, -4.3),
    (6.4, -1.4),
    (3.2, -1.8),
    (1.95, -2.4),
    (0.0, -0.6),
    (-3.0, -0.5),
    (-5.2, 0.6),
    (-5.2, 1.7),
    (-6.7, 4.2),
    (-4.1, 4.5),
    (-5.2, 1.7),
    (-2.1, 0.5),
    (-0.1, 1.7),
    (-0.6, 4.4),
    (1.55, 4.0),
    (-0.1, 1.7),
    (2.0, 0.5),
    (4.0, 0.6),
    (4.0, 1.7),
    (4.0, 4.5),
    (3.45, 2.4),
    (4.0, 1.7),
    (0.0, -0.6),
    (-4.0, -4.8),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class InflationCostmap:
    def __init__(self, inflation_radius, lethal_cost, unknown_cost, cost_weight):
        self.inflation_radius = inflation_radius
        self.lethal_cost = lethal_cost
        self.unknown_cost = unknown_cost
        self.cost_weight = cost_weight
        self.grid = None
        self.costs = []
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0

    def update(self, msg):
        self.grid = msg
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.costs = self.inflate(msg.data)

    def inflate(self, data):
        costs = [0] * len(data)
        obstacle_cells = []
        for index, value in enumerate(data):
            if value < 0:
                costs[index] = self.unknown_cost
            elif value >= self.lethal_cost:
                costs[index] = 100
                obstacle_cells.append((index % self.width, index // self.width))
            else:
                costs[index] = int(clamp(value, 0, 100))

        radius_cells = max(1, int(math.ceil(self.inflation_radius / self.resolution)))
        for ox, oy in obstacle_cells:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    nx = ox + dx
                    ny = oy + dy
                    if not self.inside_cell(nx, ny):
                        continue
                    distance = math.hypot(dx, dy) * self.resolution
                    if distance > self.inflation_radius:
                        continue
                    index = ny * self.width + nx
                    if costs[index] >= 100:
                        continue
                    ratio = 1.0 - distance / max(self.inflation_radius, self.resolution)
                    inflated = int(15 + 80 * ratio)
                    costs[index] = max(costs[index], inflated)
        return costs

    def ready(self):
        return self.grid is not None and bool(self.costs)

    def inside_cell(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def world_to_cell(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def cell_to_world(self, gx, gy):
        return (
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def cost_at_cell(self, gx, gy):
        if not self.inside_cell(gx, gy):
            return 100
        return self.costs[gy * self.width + gx]

    def cost_at_world(self, x, y):
        return self.cost_at_cell(*self.world_to_cell(x, y))

    def traversable(self, gx, gy):
        return self.cost_at_cell(gx, gy) < 98

    def line_traversable(self, start, end, step=0.08):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(distance / step))
        max_cost = 0
        for index in range(steps + 1):
            ratio = index / float(steps)
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            cost = self.cost_at_world(x, y)
            if cost >= 98:
                return False, cost
            max_cost = max(max_cost, cost)
        return True, max_cost

    def astar(self, start_xy, goal_xy, max_expansions=22000):
        if not self.ready():
            return []
        start = self.world_to_cell(*start_xy)
        goal = self.world_to_cell(*goal_xy)
        if not self.traversable(*start):
            start = self.nearest_free(start)
        if not self.traversable(*goal):
            goal = self.nearest_free(goal)
        if start is None or goal is None:
            return []

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]
        open_heap = [(0.0, start)]
        came_from = {}
        costs = {start: 0.0}
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, current = heapq.heappop(open_heap)
            expansions += 1
            if current == goal:
                break
            for dx, dy, step_cost in neighbors:
                nxt = current[0] + dx, current[1] + dy
                if not self.traversable(*nxt):
                    continue
                if dx and dy and (not self.traversable(current[0] + dx, current[1]) or
                                  not self.traversable(current[0], current[1] + dy)):
                    continue
                cell_cost = self.cost_at_cell(*nxt) / 100.0
                new_cost = costs[current] + step_cost * (1.0 + self.cost_weight * cell_cost)
                if nxt not in costs or new_cost < costs[nxt]:
                    costs[nxt] = new_cost
                    priority = new_cost + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(open_heap, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return []
        cells = [goal]
        while cells[-1] != start:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        return self.simplify([self.cell_to_world(*cell) for cell in cells])

    def nearest_free(self, cell, max_radius=16):
        gx, gy = cell
        for radius in range(max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    candidate = gx + dx, gy + dy
                    if self.traversable(*candidate):
                        return candidate
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    candidate = gx + dx, gy + dy
                    if self.traversable(*candidate):
                        return candidate
        return None

    def simplify(self, path):
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor = 0
        index = 2
        while index < len(path):
            free, _ = self.line_traversable(path[anchor], path[index], step=max(0.06, self.resolution))
            if not free:
                simplified.append(path[index - 1])
                anchor = index - 1
            index += 1
        simplified.append(path[-1])
        return simplified

    def to_message(self):
        msg = OccupancyGrid()
        if self.grid is None:
            return msg
        msg.header = self.grid.header
        msg.info = self.grid.info
        msg.data = [int(clamp(value, 0, 100)) for value in self.costs]
        return msg


class IndoorWaypointMapper:
    def __init__(self):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.status_pub = rospy.Publisher("/indoor_mapper/status", String, queue_size=1, latch=True)
        self.path_pub = rospy.Publisher("/indoor_mapper/waypoints", Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher("/indoor_mapper/local_plan", Path, queue_size=1, latch=True)
        self.costmap_pub = rospy.Publisher("/indoor_mapper/costmap", OccupancyGrid, queue_size=1, latch=True)
        self.target_pub = rospy.Publisher("/indoor_mapper/current_target", PoseStamped, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        self.loop = rospy.get_param("~loop", True)
        self.reached_distance = rospy.get_param("~reached_distance", 0.42)
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.42)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.25)
        self.front_stop_distance = rospy.get_param("~front_stop_distance", 0.58)
        self.side_stop_distance = rospy.get_param("~side_stop_distance", 0.36)
        self.collision_distance = rospy.get_param("~collision_distance", 0.25)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.85)
        self.path_resolution = rospy.get_param("~path_resolution", 0.18)
        self.smoothing_passes = int(rospy.get_param("~smoothing_passes", 2))
        self.stuck_seconds = rospy.Duration(rospy.get_param("~stuck_seconds", 4.0))
        self.stuck_distance = rospy.get_param("~stuck_distance", 0.05)
        self.backup_seconds = rospy.Duration(rospy.get_param("~backup_seconds", 1.25))
        self.turn_seconds = rospy.Duration(rospy.get_param("~turn_seconds", 1.15))
        self.costmap_inflation_radius = rospy.get_param("~costmap_inflation_radius", 0.45)
        self.costmap_lethal_cost = rospy.get_param("~costmap_lethal_cost", 70)
        self.costmap_unknown_cost = rospy.get_param("~costmap_unknown_cost", 18)
        self.costmap_weight = rospy.get_param("~costmap_weight", 5.0)
        self.replan_interval = rospy.Duration(rospy.get_param("~replan_interval", 2.0))
        self.start_delay = rospy.Duration(rospy.get_param("~start_delay", 3.0))

        self.waypoints = self.load_waypoints()
        self.route = self.build_smooth_route(self.waypoints)
        self.global_waypoint_index = 1
        self.index = 0
        self.odom = None
        self.scan = None
        self.costmap = InflationCostmap(
            self.costmap_inflation_radius,
            self.costmap_lethal_cost,
            self.costmap_unknown_cost,
            self.costmap_weight,
        )
        self.last_replan_time = rospy.Time(0)
        self.last_cmd = Twist()
        self.start_time = rospy.Time.now()
        self.motion_reference = None
        self.motion_reference_time = rospy.Time.now()
        self.recovery_mode = None
        self.recovery_until = rospy.Time(0)
        self.recovery_turn_sign = 1.0
        self.detour_point = None
        self.detour_expires = rospy.Time(0)
        self.publish_path()

    def load_waypoints(self):
        raw = rospy.get_param("~waypoints", "")
        if not raw:
            return list(DEFAULT_WAYPOINTS)
        points = []
        for pair in raw.split(";"):
            if not pair.strip():
                continue
            x_str, y_str = pair.split(",")
            points.append((float(x_str), float(y_str)))
        return points or list(DEFAULT_WAYPOINTS)

    def odom_callback(self, msg):
        self.odom = msg

    def scan_callback(self, msg):
        self.scan = msg

    def map_callback(self, msg):
        self.costmap.update(msg)
        self.costmap_pub.publish(self.costmap.to_message())

    def publish_path(self):
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()
        for x, y in self.waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)
        self.publish_local_path()

    def publish_local_path(self):
        smooth = Path()
        smooth.header.frame_id = "odom"
        smooth.header.stamp = rospy.Time.now()
        for x, y in self.route:
            pose = PoseStamped()
            pose.header = smooth.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            smooth.poses.append(pose)
        self.local_path_pub.publish(smooth)

    def active_global_goal(self):
        if not self.waypoints:
            return None
        if self.global_waypoint_index >= len(self.waypoints):
            if not self.loop:
                return None
            self.global_waypoint_index = 0
        return self.waypoints[self.global_waypoint_index]

    def maybe_advance_global_goal(self, x, y):
        goal = self.active_global_goal()
        if goal is None:
            return False
        if math.hypot(goal[0] - x, goal[1] - y) > self.reached_distance:
            return False
        self.global_waypoint_index += 1
        self.index = 0
        self.last_replan_time = rospy.Time(0)
        return True

    def replan_with_costmap(self, x, y, force=False):
        if not self.costmap.ready():
            return False
        now = rospy.Time.now()
        if not force and now - self.last_replan_time < self.replan_interval:
            return False
        goal = self.active_global_goal()
        if goal is None:
            return False

        path = self.costmap.astar((x, y), goal)
        if len(path) < 2:
            self.last_replan_time = now
            return False
        self.route = self.build_smooth_route(path)
        self.index = 0
        self.last_replan_time = now
        self.publish_local_path()
        return True

    def route_needs_replan(self, x, y):
        if not self.costmap.ready() or not self.route:
            return False
        target = self.lookahead_target(x, y)
        if target is None:
            return True
        free, cost = self.costmap.line_traversable((x, y), target)
        return (not free) or cost > 88

    def build_smooth_route(self, points):
        if len(points) < 2:
            return list(points)

        dense = []
        for start, end in zip(points[:-1], points[1:]):
            sx, sy = start
            ex, ey = end
            distance = math.hypot(ex - sx, ey - sy)
            steps = max(1, int(distance / self.path_resolution))
            for step in range(steps):
                t = step / float(steps)
                dense.append((sx + (ex - sx) * t, sy + (ey - sy) * t))
        dense.append(points[-1])

        smooth = dense
        for _ in range(max(0, self.smoothing_passes)):
            if len(smooth) < 5:
                break
            next_points = [smooth[0], smooth[1]]
            for index in range(2, len(smooth) - 2):
                p0 = smooth[index - 2]
                p1 = smooth[index - 1]
                p2 = smooth[index]
                p3 = smooth[index + 1]
                p4 = smooth[index + 2]
                # Uniform cubic B-spline filter. It keeps the route smooth enough
                # for the local planner without making wide corner cuts.
                x = (p0[0] + 4.0 * p1[0] + 6.0 * p2[0] + 4.0 * p3[0] + p4[0]) / 16.0
                y = (p0[1] + 4.0 * p1[1] + 6.0 * p2[1] + 4.0 * p3[1] + p4[1]) / 16.0
                next_points.append((x, y))
            next_points.extend(smooth[-2:])
            smooth = next_points
        return smooth

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

    def clearance_for_heading(self, heading):
        return self.sector_min(heading - 0.24, heading + 0.24)

    def current_pose(self):
        pose = self.odom.pose.pose
        return pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation)

    def advance_index(self, x, y):
        if not self.route:
            return

        search_end = min(len(self.route), self.index + 35)
        best_index = self.index
        best_distance = float("inf")
        for index in range(self.index, search_end):
            distance = math.hypot(self.route[index][0] - x, self.route[index][1] - y)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        self.index = best_index

        while self.index < len(self.route) - 1:
            distance = math.hypot(self.route[self.index][0] - x, self.route[self.index][1] - y)
            if distance > self.reached_distance:
                break
            self.index += 1

    def lookahead_target(self, x, y):
        if self.detour_point is not None and rospy.Time.now() < self.detour_expires:
            return self.detour_point
        self.detour_point = None

        if not self.route:
            return None
        self.advance_index(x, y)
        if self.index >= len(self.route):
            return None

        accumulated = 0.0
        last = (x, y)
        for index in range(self.index, len(self.route)):
            point = self.route[index]
            accumulated += math.hypot(point[0] - last[0], point[1] - last[1])
            if accumulated >= self.lookahead_distance:
                return point
            last = point
        return self.route[-1]

    def publish_target(self, x, y):
        target = PoseStamped()
        target.header.frame_id = "odom"
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.orientation.w = 1.0
        self.target_pub.publish(target)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def set_recovery(self, mode, duration, turn_sign=None):
        self.recovery_mode = mode
        self.recovery_until = rospy.Time.now() + duration
        if turn_sign is not None:
            self.recovery_turn_sign = turn_sign

    def start_replan_recovery(self, x, y, yaw, reason):
        left = self.sector_min(0.35, 1.45)
        right = self.sector_min(-1.45, -0.35)
        self.recovery_turn_sign = 1.0 if left >= right else -1.0
        lateral = self.recovery_turn_sign * 0.85
        forward = 0.45
        self.detour_point = (
            x + math.cos(yaw) * forward - math.sin(yaw) * lateral,
            y + math.sin(yaw) * forward + math.cos(yaw) * lateral,
        )
        self.detour_expires = rospy.Time.now() + rospy.Duration(5.0)
        self.last_replan_time = rospy.Time(0)
        self.set_recovery("backup", self.backup_seconds)
        self.status_pub.publish("%s: backing up, then replanning around obstacle" % reason)

    def update_motion_reference(self, x, y, moving_commanded):
        now = rospy.Time.now()
        if self.motion_reference is None or not moving_commanded:
            self.motion_reference = (x, y)
            self.motion_reference_time = now
            return False

        if now - self.motion_reference_time < self.stuck_seconds:
            return False

        moved = math.hypot(x - self.motion_reference[0], y - self.motion_reference[1])
        self.motion_reference = (x, y)
        self.motion_reference_time = now
        return moved < self.stuck_distance

    def recovery_step(self):
        now = rospy.Time.now()
        twist = Twist()
        if self.recovery_mode == "backup":
            if now < self.recovery_until:
                twist.linear.x = -0.18
                twist.angular.z = -0.25 * self.recovery_turn_sign
                self.cmd_pub.publish(twist)
                self.last_cmd = twist
                self.status_pub.publish("recovery: backing up")
                return True
            self.set_recovery("turn", self.turn_seconds)

        if self.recovery_mode == "turn":
            if now < self.recovery_until:
                twist.angular.z = self.recovery_turn_sign * 0.85
                self.cmd_pub.publish(twist)
                self.last_cmd = twist
                self.status_pub.publish("recovery: turning toward clearer side")
                return True
            self.recovery_mode = None
            self.stop()
            return False

        return False

    def local_planner_cmd(self, x, y, yaw, target):
        tx, ty = target
        desired_yaw = math.atan2(ty - y, tx - x)
        heading_error = angle_diff(desired_yaw, yaw)
        front = self.sector_min(-0.34, 0.34)
        left = self.sector_min(0.35, 1.40)
        right = self.sector_min(-1.40, -0.35)

        if front < self.collision_distance:
            self.start_replan_recovery(x, y, yaw, "near collision")
            return Twist()

        best_score = -float("inf")
        best_cmd = Twist()
        linear_samples = [0.0, 0.10, 0.18, 0.28, self.max_linear_speed]
        angular_samples = [
            -self.max_angular_speed,
            -0.75,
            -0.35,
            0.0,
            0.35,
            0.75,
            self.max_angular_speed,
        ]

        for linear in linear_samples:
            for angular in angular_samples:
                predicted_heading = angle_diff(heading_error, -angular * 0.45)
                clearance = self.clearance_for_heading(angular * 0.45)
                if linear > 0.05 and clearance < self.collision_distance + 0.10:
                    continue
                predicted_x = x + math.cos(yaw + angular * 0.45) * linear * 0.9
                predicted_y = y + math.sin(yaw + angular * 0.45) * linear * 0.9
                costmap_penalty = 0.0
                if self.costmap.ready():
                    free, path_cost = self.costmap.line_traversable((x, y), (predicted_x, predicted_y))
                    if not free:
                        continue
                    costmap_penalty = 2.2 * (path_cost / 100.0)
                obstacle_cost = 0.0
                if front < self.front_stop_distance and abs(angular) < 0.25 and linear > 0.05:
                    obstacle_cost += 5.0
                if left < self.side_stop_distance and angular > 0.0:
                    obstacle_cost += 1.5
                if right < self.side_stop_distance and angular < 0.0:
                    obstacle_cost += 1.5

                progress_score = 2.8 * linear * max(0.0, math.cos(predicted_heading))
                heading_score = -2.1 * abs(predicted_heading)
                clearance_score = min(clearance, 1.5)
                smoothness_score = -0.25 * abs(angular - self.last_cmd.angular.z)
                stop_penalty = -0.15 if linear < 0.05 else 0.0
                score = progress_score + heading_score + clearance_score + smoothness_score + stop_penalty
                score -= obstacle_cost + costmap_penalty
                if score > best_score:
                    best_score = score
                    best_cmd.linear.x = linear
                    best_cmd.angular.z = angular

        if best_cmd.linear.x > 0.05 and front < self.front_stop_distance:
            best_cmd.linear.x *= clamp((front - self.collision_distance) / max(0.01, self.front_stop_distance), 0.0, 1.0)
            if abs(best_cmd.angular.z) < 0.25:
                best_cmd.angular.z = 0.65 if left >= right else -0.65

        if best_cmd.linear.x <= 0.02 and abs(best_cmd.angular.z) <= 0.02:
            best_cmd.angular.z = 0.55 if left >= right else -0.55

        return best_cmd

    def step(self):
        if self.odom is None or self.scan is None:
            self.status_pub.publish("waiting for odom and scan")
            self.stop()
            return

        if rospy.Time.now() - self.start_time < self.start_delay:
            self.status_pub.publish("letting Gazebo and Cartographer settle")
            self.stop()
            return

        if self.index >= len(self.route):
            if not self.loop:
                self.status_pub.publish("smooth route complete")
                self.stop()
                return
            self.index = 0

        if self.recovery_step():
            return

        x, y, yaw = self.current_pose()
        if self.maybe_advance_global_goal(x, y):
            self.status_pub.publish("advanced to global waypoint %d/%d" % (
                min(self.global_waypoint_index + 1, len(self.waypoints)),
                len(self.waypoints),
            ))

        if self.costmap.ready() and (self.route_needs_replan(x, y) or self.index == 0):
            self.replan_with_costmap(x, y, force=self.route_needs_replan(x, y))

        target = self.lookahead_target(x, y)
        if target is None:
            self.index = 0 if self.loop else len(self.route)
            self.stop()
            return

        target_x, target_y = target
        self.publish_target(target_x, target_y)
        twist = self.local_planner_cmd(x, y, yaw, target)
        if self.update_motion_reference(x, y, twist.linear.x > 0.06):
            self.start_replan_recovery(x, y, yaw, "stuck")
            return

        goal = self.active_global_goal()
        remaining = math.hypot(goal[0] - x, goal[1] - y) if goal is not None else 0.0
        cost_status = "costmap ready" if self.costmap.ready() else "waiting for costmap"
        status = "%s; local planning on B-spline/A* route %d/%d, goal distance %.2f" % (
            cost_status,
            min(self.index + 1, len(self.route)),
            len(self.route),
            remaining,
        )
        if self.detour_point is not None:
            status = "temporary detour active; " + status
        self.status_pub.publish(status)
        self.cmd_pub.publish(twist)
        self.last_cmd = twist

    def run(self):
        rate = rospy.Rate(15)
        try:
            while not rospy.is_shutdown():
                self.step()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.stop()


if __name__ == "__main__":
    rospy.init_node("indoor_waypoint_mapper")
    IndoorWaypointMapper().run()
