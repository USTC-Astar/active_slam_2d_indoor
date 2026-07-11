#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/noetic/setup.bash

for setup_file in \
  "$HOME/cartographer_ws/install_isolated/setup.bash" \
  "$HOME/cartographer_ws/devel_isolated/setup.bash" \
  "$HOME/cartographer_ws_v3/install_isolated/setup.bash" \
  "$HOME/cartographer_ws_v3/devel_isolated/setup.bash" \
  "$HOME/cartographer_ws_v2/install_isolated/setup.bash" \
  "$HOME/cartographer_ws_v2/devel_isolated/setup.bash" \
  "$HOME/cartographer_ws_explorelite_stage/install_isolated/setup.bash" \
  "$HOME/cartographer_ws_explorelite_stage/devel_isolated/setup.bash"; do
  if [[ -f "$setup_file" ]]; then
    source "$setup_file"
  fi
done

export ROS_PACKAGE_PATH="$SCRIPT_DIR/indoor_cartographer_demo:$HOME:${ROS_PACKAGE_PATH:-}"

if ! rospack find cartographer_ros >/dev/null 2>&1; then
  echo "cartographer_ros was not found. Source a workspace containing cartographer_ros first." >&2
  exit 1
fi

roslaunch "$SCRIPT_DIR/indoor_cartographer_demo/launch/indoor_cartographer.launch" "$@"
