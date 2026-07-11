# Indoor Cartographer Mapping Demo

This demo adds a high-detail Gazebo apartment scene for Cartographer 2D SLAM.
The layout contains two bedrooms, one kitchen, one living room, and one
bathroom. The robot has a 2D lidar, wheel odometry, and an RGB camera published
as `/robot_view/image_raw`.

Run it from `~/cartographer`:

```bash
./run_indoor_mapping_demo.sh
```

The detailed apartment remains the default. Use the smaller, wider quick home
when iterating on exploration and recovery behavior:

```bash
./run_indoor_mapping_demo.sh scene:=quick
./run_indoor_mapping_demo.sh scene:=detailed
```

Useful launch arguments:

```bash
./run_indoor_mapping_demo.sh autonomous:=false
./run_indoor_mapping_demo.sh gui:=false
./run_indoor_mapping_demo.sh rviz:=false
```

RViz opens with a narrow left display panel, the `Robot View` camera panel
docked at the lower left, and a larger map view. The display stack includes the
Cartographer map/submaps, scan, robot model, Active SLAM frontiers, the active
path, the current target, and the lower-left camera view.

Autonomous exploration now defaults to a lightweight Active SLAM node instead
of `move_base + explore_lite`. `active_slam_explorer.py` reads `/map`, `/scan`,
and TF, runs reachable-frontier detection plus cost-aware Dijkstra planning in
known free space, follows a lookahead
point on that path, and falls back to fast open-space probing when the initial
frontiers are too close. This keeps the loop small and responsive for short
Cartographer mapping checks.

The default profile is tuned for fast coverage: control commands are published
at `24 Hz`, the cruise speed is `0.82 m/s`, and frontier scoring favors larger
unknown regions over tiny nearby frontiers. Local obstacle handling uses a
VFH-style laser heading selector, so each control step chooses a safe gap that
still points toward the active frontier whenever possible.

The local controller evaluates a footprint-width laser corridor rather than a
single center ray. A Gazebo contact sensor provides a final collision signal.
When motion is blocked, recovery brakes, checks rear clearance, backs away,
turns toward the more open side, probes forward, blacklists the failed frontier,
and forces a fresh global frontier plan.

The compact robot is approximately `0.34 m` long and `0.335 m` wide across the
wheels. Planning uses a graded costmap on `/active_slam/costmap`: lethal cells
cover the physical footprint, while a wider decaying cost band keeps Dijkstra
paths away from walls. RViz shows it as `Navigation Costmap` with low alpha so
the occupancy map remains readable. Cartographer submaps, trajectory nodes, and
constraints remain available but disabled by default to avoid visual clutter.

Exploration completion requires repeated plans with no useful reachable
frontier, a minimum mapping runtime, a minimum accumulated travel distance, and
a stable known-cell count. Patrol scoring uses a large unknown-area window and
recorded visit positions to prefer rooms and corridors that have seen less
coverage. The robot
then plans back to its recorded start position. At home it publishes
`/active_slam/completed=True` and continues publishing a zero `/cmd_vel`.
Dead ends trigger a longer checked reverse before turning and replanning.

If normal frontier clusters temporarily disappear, the explorer now selects a
reachable patrol waypoint near unknown space and follows a BFS path instead of
rotating indefinitely. Target hysteresis prevents rapid left/right goal changes.
The controller also detects high accumulated rotation with little translation
and forces recovery. Map access is synchronized so occupancy-grid expansion
cannot terminate the control timer and leave Gazebo executing a stale command.

Cartographer receives strictly increasing odometry through
`/cartographer/odom`; duplicate Gazebo timestamps are removed by
`indoor_odom_filter.py`. RViz includes Cartographer submaps and trajectory nodes,
with the denser constraint visualization available but disabled by default.

Main Active SLAM topics:

```bash
/active_slam/status
/active_slam/frontiers
/active_slam/path
/active_slam/target
```

The old navigation stack is still available for comparison:

```bash
./run_indoor_mapping_demo.sh active_slam:=false navigation:=true explore:=true
```

The active costmap is the `Navigation Costmap` display. The legacy move_base
local costmap remains disabled and is only relevant when launching the optional
navigation stack.
