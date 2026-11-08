import numpy as np  
import time
import math
import matplotlib.pyplot as plt  
from controller import Robot, Lidar  

TIME_STEP = 32  
GRID_SIZE = 500  
pos_scalar = 100  

WHEEL_RADIUS = 0.05  
TRACK_WIDTH = 0.23    

robot = Robot()  

lidar = robot.getDevice("lidar")  
lidar.enable(TIME_STEP)  
lidar.enablePointCloud()  

us_sensors = [robot.getDevice("us0"), robot.getDevice("us1")]
for us in us_sensors:
    if us is not None:
        us.enable(TIME_STEP)

left_motor = robot.getDevice("left wheel motor")  
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))  
right_motor.setPosition(float("inf"))  
left_motor.setVelocity(0.0)  
left_motor.setVelocity(0.0)    

grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
motion_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

plt.ion()  
fig_grid, ax = plt.subplots()  
fig_motion, ex = plt.subplots()  

def clean_lidar_data(lidar_data):  
    return np.where(np.isfinite(lidar_data), lidar_data, 3.0)      

def transform_to_pos(position):  
    # Pixel transformation centered at (250, 250) (GRID_SIZE // 2 = 250)
    pos_x = int(position[0] * pos_scalar) + GRID_SIZE // 2
    pos_y = int(position[1] * pos_scalar) + GRID_SIZE // 2  
    return np.clip(pos_x, 0, GRID_SIZE - 1), np.clip(pos_y, 0, GRID_SIZE - 1)

def draw_line(grid, x0, y0, x1, y1, value=1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < grid.shape[0] and 0 <= x0 < grid.shape[1]:
            grid[x0, y0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

position = np.array([0.0, 0.0], dtype=np.float64)  
previous_pos = np.array([0.0, 0.0], dtype=np.float64)
last_mapped_pos = np.array([0.0, 0.0], dtype=np.float64)
orientation = 0.0  

OBSTACLE_THRESHOLD = 0.25  
dt = float(TIME_STEP / 1000.0)    

# Variable for pure LiDAR-based reference tracking
initial_ref_dist_0 = None

while robot.step(TIME_STEP) != -1:  
    us_data = [us.getValue() if us is not None and us.getValue() > 0 else 99.9 for us in us_sensors]
    left_obs = us_data[0] < OBSTACLE_THRESHOLD
    right_obs = us_data[1] < OBSTACLE_THRESHOLD

    prev_pos_x, prev_pos_y = transform_to_pos(position)

    # Obstacle Avoidance and Velocity Commands
    if left_obs and right_obs:
        left_speed = -2.0  
        right_speed = 2.0  
        position[0] = previous_pos[0]
        position[1] = previous_pos[1]
    elif right_obs:
        left_speed = -2.0  
        right_speed = 2.0  
        position[0] = previous_pos[0]
        position[1] = previous_pos[1]
    elif left_obs:
        left_speed = 2.0  
        right_speed = -2.0  
        position[0] = previous_pos[0]
        position[1] = previous_pos[1]
    else:  
        left_speed = 2.0  
        right_speed = 2.0  
        
        linear_velocity = ((left_speed + right_speed) / 2.0) * WHEEL_RADIUS
        position += linear_velocity * dt * np.array([np.cos(orientation), np.sin(orientation)])
        
        previous_pos[0] = position[0]
        previous_pos[1] = position[1]

    # Orientation Update
    omega = ((right_speed - left_speed) * WHEEL_RADIUS) / TRACK_WIDTH
    orientation = (orientation + omega * dt) % (2 * np.pi)

    left_motor.setVelocity(left_speed)  
    right_motor.setVelocity(right_speed)  

    # Print Real-time Values to Console
    print(f"Left Speed: {left_speed:5.2f} | Right Speed: {right_speed:5.2f} | Orientation: {orientation:6.4f} rad")

    # LiDAR Data Processing and Pure LiDAR Reference Logic
    is_turning = left_obs or right_obs
    raw_lidar = lidar.getRangeImage()  
    
    if len(raw_lidar) > 0 and not all(np.isinf(raw_lidar)):
        curr_clean = clean_lidar_data(raw_lidar)
        n = len(curr_clean)
        
        # Pure LiDAR distance measurement of index 0 (Odometry decoupled)
        dist_idx_0 = curr_clean[0]
        if 0 < dist_idx_0 < lidar.getMaxRange():
            if initial_ref_dist_0 is None:
                initial_ref_dist_0 = dist_idx_0
                print(f"Pure LiDAR Reference Distance Locked: Distance={initial_ref_dist_0:.2f}m")
            else:
                # Change in pure LiDAR distance (drift/deviation observation)
                distance_drift = abs(dist_idx_0 - initial_ref_dist_0)
                # You can use this drift data later for pure LiDAR-based corrections independent of odometry

        # Grid map update only during straight movement and centered at (250,250)
        if not is_turning:
            angles = 2 * np.pi - np.linspace(0, 2 * np.pi, n, endpoint=False)  
            x_coords = []
            y_coords = []
            
            for i in range(n):
                dist = curr_clean[i]
                if 0.05 < dist < 3.0:
                    local_x = dist * np.cos(angles[i])
                    local_y = dist * np.sin(angles[i])
                    
                    global_x = position[0] + (local_x * np.cos(orientation) - local_y * np.sin(orientation))
                    global_y = position[1] + (local_x * np.sin(orientation) + local_y * np.cos(orientation))
                    
                    gx = int(global_x * pos_scalar) + GRID_SIZE // 2
                    gy = int(global_y * pos_scalar) + GRID_SIZE // 2
                    
                    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                        x_coords.append(gx)
                        y_coords.append(gy)

            distance_traveled = np.linalg.norm(position - last_mapped_pos)
            if distance_traveled > 0.03 and len(x_coords) > 0:
                last_mapped_pos = np.copy(position)
                for gx, gy in zip(x_coords, y_coords):  
                    grid_map[gx, gy] = 1

    # Grid Map Plotting
    ax.clear()  
    ax.imshow(grid_map, cmap='gray', interpolation='nearest')  
    ax.set_title('LiDAR Map')  
    ax.set_xlim(0, GRID_SIZE)  
    ax.set_ylim(GRID_SIZE, 0)  
    plt.draw()  
    plt.pause(0.001)  

    # Motion Map Plotting
    curr_pos_x, curr_pos_y = transform_to_pos(position)
    draw_line(motion_map, prev_pos_x, prev_pos_y, curr_pos_x, curr_pos_y, value=1)
    ex.clear()  
    ex.imshow(motion_map, cmap='pink', interpolation='nearest')  
    ex.set_title('Motion Map')  
    ex.set_xlim(0, GRID_SIZE)  
    ex.set_ylim(GRID_SIZE, 0)  
    plt.draw()  
    plt.pause(0.001)          

plt.ioff()
plt.show()