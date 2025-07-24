# Import required message types
std_msgs/Header header
Obstacle[] obstacles  # Array of obstacles 

# Individual obstacle definition
================
MSG: rraaa_sim/Obstacle
float32 x_min
float32 x_max
float32 y_min
float32 y_max
float32 z_min
float32 z_max
bool is_moving
float32[] velocity  # 3D velocity vector [vx, vy, vz] 