# Troubleshooting Guide


## Problem: Simulator stuck

**Solutions**:
Use docker compose manually to stop all services.
```bash
cd docker
docker compose stop
```

## Problem: No Display for GUI Applications
**Symptoms**:
- GUI applications like RViz, Pygame, or other Qt-based tools do not open when executed from inside the container.
- Error message:
  - "Authorization required, but no authorization protocol specified"
  - "qt.qpa.xcb: could not connect to display :0"

**Causes**:
- The Docker container does not have permission to access the host's X server.

**Solutions**:
1. Allow the container to access the host's X server by running the following command on the host:
   ```bash
   xhost +local:docker
   ```
2. Modify your Docker Compose file to include:
   ```yaml
   environment:
     - DISPLAY=${DISPLAY}
   volumes:
     - /tmp/.X11-unix:/tmp/.X11-unix
   network_mode: host
   ```
3. Ensure that the `DISPLAY` variable is set correctly, typically `:0` or the corresponding value on your host.

**Security Note**: The `xhost +local:docker` command can pose a security risk by allowing access to all local containers. To restrict access:
```bash
xhost +si:localuser:root
```

## Problem: Terminal unusable on exit
**Symptoms**:
- Terminal shows no inputs.


**Solutions**:
1. Reset the terminal. While your typing won't be visible, the command will still run.
   ```bash
   reset
   ```

