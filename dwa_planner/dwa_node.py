import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def angle_diff(a, b):
    """Smallest absolute difference between two angles (0..pi)."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


class DwaNode(Node):
    def __init__(self):
        super().__init__('dwa_node')

        self.declare_parameter('goal_x', 1.5)
        self.declare_parameter('goal_y', 0.0)
        self.approach_dist = 0.8

        self.max_v = 0.22
        self.min_v = 0.0
        self.max_w = 1.5
        self.acc_v = 0.5
        self.acc_w = 2.0

        self.dt = 0.1
        self.horizon = 3.0
        self.v_samples = 8
        self.w_samples = 15

        self.robot_radius = 0.12
        self.influence_dist = 0.5
        self.goal_tolerance = 0.2

        self.w_goal = 2.0
        self.w_heading = 0.5
        self.w_clear = 2.5
        self.w_vel = 0.5
        self.w_smooth = 0.1

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.scan = None
        self.goal_reached = False

        self.stuck_since = None
        self.recovery_until = 0.0
        self.recovery_w = 0.0

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb,
                                 qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dwa_markers', 10) 

        self.create_timer(self.dt, self.loop)
        self.get_logger().info('DWA node started')
        self.get_logger().info(f'Goal = ({self.get_parameter("goal_x").value}, 'f'{self.get_parameter("goal_y").value})')
    

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

    def scan_cb(self, msg):
        self.scan = msg

    def publish_cmd(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def dynamic_window(self):
        v_low = max(self.min_v, self.v - self.acc_v * self.dt)
        v_high = min(self.max_v, self.v + self.acc_v * self.dt)
        w_low = max(-self.max_w, self.w - self.acc_w * self.dt)
        w_high = min(self.max_w, self.w + self.acc_w * self.dt)
        return v_low, v_high, w_low, w_high

    def predict(self, v, w):
        x, y, th = self.x, self.y, self.yaw
        path = []
        for _ in range(int(self.horizon / self.dt)):
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += w * self.dt
            path.append((x, y, th))
        return np.array(path)  

    def obstacle_points(self):
        """Laser beams -> (x, y) points in the odom frame."""
        s = self.scan
        n = len(s.ranges)
        r = np.array(s.ranges[::4]) 
        ang = s.angle_min + np.arange(0, n, 4) * s.angle_increment
        ok = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
        r, ang = r[ok], ang[ok]
        ox = self.x + r * np.cos(self.yaw + ang)
        oy = self.y + r * np.sin(self.yaw + ang)
        return np.stack([ox, oy], axis=1)
    
    def side_clearance(self):
        """Average free distance on the left and right sides of the robot."""
        s = self.scan
        r = np.array(s.ranges, dtype=float)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        r = np.where(np.isfinite(r), r, s.range_max)
        
        left_mask = (ang > 0.3) & (ang < 1.5)
        right_mask = (ang < -0.3) & (ang > -1.5)
        
        left = float(r[left_mask].mean()) if left_mask.any() else s.range_max
        right = float(r[right_mask].mean()) if right_mask.any() else s.range_max
        return left, right
    
    def evaluate(self, path, v, w, obstacles, gx, gy):
        """Return the cost of one trajectory (math.inf = unusable)."""
       
        min_d = math.inf
        if len(obstacles) > 0:
            diff = path[:, None, :2] - obstacles[None, :, :]
            min_d = float(np.sqrt((diff ** 2).sum(axis=2)).min())

       
        free = min_d - self.robot_radius
        
        if v > math.sqrt(2.0 * self.acc_v * max(free, 0.0)) + 1e-6:
            return math.inf, min_d
        near_goal = min(1.0, math.hypot(gx - self.x, gy - self.y) / self.approach_dist)
        clear_cost = near_goal * max(0.0, (self.influence_dist - min_d) / self.influence_dist)

      
        ex, ey, eth = path[-1]
        goal_cost = math.hypot(gx - ex, gy - ey)

      
        bearing = math.atan2(gy - ey, gx - ex)
        near = min(1.0, math.hypot(gx - self.x, gy - self.y) / 0.6)
        heading_cost = near * angle_diff(eth, bearing)

        
        vel_cost = (self.max_v - v) / self.max_v
        smooth_cost = abs(w - self.w)

        total = (self.w_goal * goal_cost + self.w_heading * heading_cost +
                 self.w_clear * clear_cost + self.w_vel * vel_cost +
                 self.w_smooth * smooth_cost)
        return total, min_d

    def publish_markers(self, candidates, best_path, gx, gy):
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()

        wipe = Marker()
        wipe.action = Marker.DELETEALL     
        arr.markers.append(wipe)

        def make(mid, ns, mtype):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = now
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            return m

        def line(mid, ns, path, rgba, width):
            m = make(mid, ns, Marker.LINE_STRIP)
            m.scale.x = width
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.05) for p in path]
            return m

        for i, (path, rejected) in enumerate(candidates):
            if rejected:
                arr.markers.append(line(i, 'rejected', path, (1.0, 0.2, 0.2, 0.25), 0.005))
            else:
                arr.markers.append(line(i, 'candidates', path, (0.6, 0.6, 0.6, 0.5), 0.005))

        if best_path is not None:
            arr.markers.append(line(0, 'best', best_path, (0.0, 1.0, 0.0, 1.0), 0.03))

            goal = make(0, 'goal', Marker.SPHERE)
            goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
            goal.scale.x = goal.scale.y = goal.scale.z = 0.2
            goal.color.r, goal.color.a = 1.0, 1.0
            arr.markers.append(goal)
    
        self.marker_pub.publish(arr)

   
    def loop(self):
        if self.scan is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9

        gx = self.get_parameter('goal_x').value
        gy = self.get_parameter('goal_y').value
        dist_goal = math.hypot(gx - self.x, gy - self.y)

        if dist_goal < self.goal_tolerance:
            self.publish_cmd(0.0, 0.0)
            if not self.goal_reached:
                self.get_logger().info(f'GOAL REACHED at ({self.x:.2f}, {self.y:.2f})')
                self.goal_reached = True
            return
        self.goal_reached = False

  
        if now < self.recovery_until:
            self.publish_cmd(0.0, self.recovery_w)
            return

        obstacles = self.obstacle_points()
        v_lo, v_hi, w_lo, w_hi = self.dynamic_window()

        best_cost, best_v, best_w, best_path = math.inf, 0.0, 0.0, None
        candidates = []
        rejected = 0
        for i in range(self.v_samples):
            v = v_lo + (v_hi - v_lo) * i / (self.v_samples - 1)
            for j in range(self.w_samples):
                w = w_lo + (w_hi - w_lo) * j / (self.w_samples - 1)
                path = self.predict(v, w)
                cost, _ = self.evaluate(path, v, w, obstacles, gx, gy)
                candidates.append((path, math.isinf(cost)))
                if math.isinf(cost):
                    rejected += 1
                elif cost < best_cost:
                    best_cost, best_v, best_w, best_path = cost, v, w, path

        self.publish_markers(candidates, best_path, gx, gy)

        if math.isinf(best_cost) or (best_v < 0.03 and rejected > 0.5 * self.v_samples * self.w_samples):
            if self.stuck_since is None:
                self.stuck_since = now
            elif now - self.stuck_since > 1.5:
                left, right = self.side_clearance()
                direction = 1.0 if left > right else -1.0
                self.recovery_w = direction * 0.8
                self.recovery_until = now + 1.5
                self.stuck_since = None
                self.get_logger().warn(
                    f'STUCK -> rotating {"left" if direction > 0 else "right"} '
                    f'(free space left={left:.2f} right={right:.2f})')
                return
        else:
            self.stuck_since = None

        if math.isinf(best_cost):
            self.get_logger().warn('All trajectories collide -> stopping',
                                   throttle_duration_sec=1.0)
            self.publish_cmd(0.0, 0.0)
            return

        self.publish_cmd(best_v, best_w)
        closest = float(np.hypot(obstacles[:, 0] - self.x, obstacles[:, 1] - self.y).min()) \
            if len(obstacles) else float('inf')
        bearing = math.degrees(math.atan2(gy - self.y, gx - self.x))
        self.get_logger().info(
            f'pose=({self.x:.2f},{self.y:.2f}) yaw={math.degrees(self.yaw):.0f} '
            f'goal_bearing={bearing:.0f} goal_dist={dist_goal:.2f} '
            f'closest_obs={closest:.2f} cmd=(v={best_v:.2f}, w={best_w:.2f}) '
            f'rejected={rejected}/{self.v_samples * self.w_samples}',
            throttle_duration_sec=0.5)

def main():
    rclpy.init()
    node = DwaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)   
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()