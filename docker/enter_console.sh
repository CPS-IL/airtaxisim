#!/bin/bash

DEFAULT_SERVICE_NAME="env_sim"

SERVICE_NAME="${1:-$DEFAULT_SERVICE_NAME}"

VALID_SERVICES=$(docker compose -f docker-compose.yml config --services)
if ! echo "$VALID_SERVICES" | grep -qw "$SERVICE_NAME"; then
    echo "Error: Invalid service name '$SERVICE_NAME'"
    echo "Usage: $0 [<service_name>]"
    echo "Available services:"
    echo "$VALID_SERVICES"
    exit 1
fi

# Required for rviz to be able to display window
xhost +local:docker 1>/dev/null 2>&1

clear
docker compose -f docker-compose.yml exec "$SERVICE_NAME" /bin/bash -c " \
cd /catkin_ws;
source /opt/ros/noetic/setup.bash;
source devel/setup.bash;
exec bash
"
