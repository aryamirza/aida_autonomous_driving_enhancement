import rclpy
from rclpy.node import Node
import json
import os
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
import tf2_ros
import math

class AidaMemoryNode(Node):
    def __init__(self):
        super().__init__('aida_memory_node')

        self.map_file = os.path.expanduser('~/.aida/aida_memory_map.json')
        self.tmp_file = os.path.expanduser('~/.aida/aida_memory_map.tmp')

        self.map_data = {}
        self.tracked_cells = {}
        self.map_is_dirty = False
        self.current_odom = None

        self.load_map()

        # Subscriptions
        self.create_subscription(LaserScan, '/scan_raw', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Empty, '/aida/hazard_trigger', self.trigger_cb, 10)
        self.create_subscription(String, '/aida/anomaly_labels', self.anomaly_cb, 10)

        # Publisher
        self.warning_pub = self.create_publisher(String, '/aida/hazard_warning', 10)

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timers
        self.create_timer(5.0, self.save_map)
        self.create_timer(1.0 / 30.0, self.anticipation_loop)

        rclpy.get_default_context().on_shutdown(self.save_map)

    def load_map(self):
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        if os.path.exists(self.map_file):
            try:
                with open(self.map_file, 'r') as f:
                    raw_data = json.load(f)
                # Graceful migration: if data is old format (float), migrate it to new structure
                migrated = 0
                for k, v in raw_data.items():
                    if isinstance(v, (float, int)):
                        self.map_data[k] = {"confidence": float(v), "labels": ["unknown"]}
                        migrated += 1
                    elif isinstance(v, dict) and "confidence" in v and "labels" in v:
                        self.map_data[k] = v
                    else:
                        # Malformed data, skip
                        pass
                if migrated > 0:
                    self.map_is_dirty = True
                    self.get_logger().info(f"Migrated {migrated} old format cells.")
                self.get_logger().info(f"Loaded {len(self.map_data)} cells from map.")
            except Exception as e:
                self.get_logger().error(f"Failed to load map: {e}")

    def save_map(self):
        if self.map_is_dirty:
            try:
                with open(self.tmp_file, 'w') as f:
                    json.dump(self.map_data, f)
                os.replace(self.tmp_file, self.map_file)
                self.map_is_dirty = False
            except Exception as e:
                self.get_logger().error(f"Failed to save map: {e}")

    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_odom = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': yaw,
            'vx': msg.twist.twist.linear.x,
            'wz': msg.twist.twist.angular.z
        }

    def scan_cb(self, msg: LaserScan):
        if not self.current_odom:
            return

        try:
            trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            return

        q = trans.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        T = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])

        ranges = np.array(msg.ranges)
        valid_mask = (ranges > 0.0) & (ranges < float('inf'))
        valid_ranges = ranges[valid_mask]
        indices = np.where(valid_mask)[0]
        angles = msg.angle_min + indices * msg.angle_increment

        px = valid_ranges * np.cos(angles)
        py = valid_ranges * np.sin(angles)
        pz = np.zeros_like(px)

        points_local = np.vstack((px, py, pz))
        points_global = R @ points_local + T[:, np.newaxis]

        gx = points_global[0, :]
        gy = points_global[1, :]

        # Track boundaries
        valid_bounds = (gx >= 0.0) & (gx <= 1.184) & (gy >= 0.0) & (gy <= 0.781)
        gx = gx[valid_bounds]
        gy = gy[valid_bounds]

        gx_rounded = np.round(gx, 2) + 0.0
        gy_rounded = np.round(gy, 2) + 0.0

        for x_val, y_val in zip(gx_rounded, gy_rounded):
            key = f"{x_val:.2f}_{y_val:.2f}"
            if key not in self.map_data:
                self.map_data[key] = {"confidence": 0.40, "labels": ["unknown"]}
            else:
                self.map_data[key]["confidence"] = min(0.95, self.map_data[key]["confidence"] + 0.05)
                # Ensure no garbage collection here since it only goes up
            self.map_is_dirty = True

    def trigger_cb(self, msg: Empty):
        if not self.current_odom:
            return

        odom_x, odom_y = self.current_odom['x'], self.current_odom['y']
        key = f"{round(odom_x, 2) + 0.0:.2f}_{round(odom_y, 2) + 0.0:.2f}"
        if key not in self.map_data:
            self.map_data[key] = {"confidence": min(0.95, 0.40 + 0.15), "labels": ["unknown"]}
        else:
            self.map_data[key]["confidence"] = min(0.95, self.map_data[key]["confidence"] + 0.15)

        cos_y = math.cos(self.current_odom['yaw'])
        sin_y = math.sin(self.current_odom['yaw'])

        for k in list(self.tracked_cells.keys()):
            kx, ky = map(float, k.split('_'))
            k_lx = (kx - odom_x) * cos_y + (ky - odom_y) * sin_y
            k_ly = -(kx - odom_x) * sin_y + (ky - odom_y) * cos_y

            if abs(k_ly) <= 0.095 and -0.10 <= k_lx <= 0.20:
                self.tracked_cells[k]["boosted"] = True
                if k in self.map_data:
                    self.map_data[k]["confidence"] = min(0.95, self.map_data[k]["confidence"] + 0.15)

        self.map_is_dirty = True

    def anomaly_cb(self, msg: String):
        try:
            labels_data = json.loads(msg.data)
            for item in labels_data:
                x_val = item.get("x")
                y_val = item.get("y")
                label = item.get("label")

                if x_val is None or y_val is None or not label:
                    continue

                # Bounds check
                if not (0.0 <= x_val <= 1.184 and 0.0 <= y_val <= 0.781):
                    continue

                key = f"{round(x_val, 2) + 0.0:.2f}_{round(y_val, 2) + 0.0:.2f}"
                if key not in self.map_data:
                    self.map_data[key] = {"confidence": 0.40, "labels": ["unknown", label] if label != "unknown" else ["unknown"]}
                else:
                    self.map_data[key]["confidence"] = min(0.95, self.map_data[key]["confidence"] + 0.05)
                    if label not in self.map_data[key]["labels"]:
                        self.map_data[key]["labels"].append(label)
                self.map_is_dirty = True
        except Exception as e:
            self.get_logger().error(f"Failed to process anomaly labels: {e}")

    def anticipation_loop(self):
        if not self.current_odom or not self.map_data:
            return

        odom_x = self.current_odom['x']
        odom_y = self.current_odom['y']
        odom_yaw = self.current_odom['yaw']
        vx = self.current_odom['vx']
        wz = self.current_odom['wz']

        keys = list(self.map_data.keys())
        coords = np.array([list(map(float, k.split('_'))) for k in keys]).T
        gx = coords[0, :]
        gy = coords[1, :]

        dx = gx - odom_x
        dy = gy - odom_y

        cos_y = np.cos(odom_yaw)
        sin_y = np.sin(odom_yaw)
        lx = dx * cos_y + dy * sin_y
        ly = -dx * sin_y + dy * cos_y

        # Silent Pass Tracked Cells Mechanics
        in_footprint_y = np.abs(ly) <= 0.095
        in_front = (lx > -0.20) & (lx <= 0.30)
        entering = in_footprint_y & in_front
        for i in np.where(entering)[0]:
            k = keys[i]
            if k not in self.tracked_cells:
                self.tracked_cells[k] = {"boosted": False}

        for k in list(self.tracked_cells.keys()):
            kx, ky = map(float, k.split('_'))
            k_lx = (kx - odom_x) * cos_y + (ky - odom_y) * sin_y
            if k_lx <= -0.20:
                if not self.tracked_cells[k]["boosted"]:
                    if k in self.map_data:
                        self.map_data[k]["confidence"] -= 0.20
                        self.map_is_dirty = True
                        if self.map_data[k]["confidence"] <= 0.0:
                            del self.map_data[k]
                del self.tracked_cells[k]

        # Anticipation Sweep
        hazard_mask = np.array([self.map_data.get(k, {}).get("confidence", 0.0) >= 0.80 for k in keys])
        if not np.any(hazard_mask):
            return

        confident_indices = np.where(hazard_mask)[0]
        hlx = lx[confident_indices]
        hly = ly[confident_indices]

        D = abs(vx) * 0.8
        if D < 0.01:
            return

        if abs(wz) < 0.02:
            in_path = (np.abs(hly) <= 0.095) & (hlx >= 0.0) & (hlx <= D)
            if np.any(in_path):
                distances = hlx[in_path]
                best_sub_idx = np.argmin(distances)
                min_dist = distances[best_sub_idx]

                in_path_indices = np.where(in_path)[0]
                original_idx = confident_indices[in_path_indices[best_sub_idx]]
                best_key = keys[original_idx]

                # Fetch y_offset from local y coordinate
                y_offset = hly[in_path_indices[best_sub_idx]]

                ttc = min_dist / max(abs(vx), 0.01)
                self.publish_warning(ttc, self.map_data[best_key], y_offset)
        else:
            R = vx / wz
            dist_to_center = np.sqrt(hlx**2 + (hly - R)**2)
            in_path_width = np.abs(dist_to_center - abs(R)) <= 0.095
            in_front = hlx > 0.0

            ratio = np.clip(hlx / dist_to_center, -1.0, 1.0)
            arc_len = abs(R) * np.arcsin(ratio)

            in_range = arc_len <= D
            in_path = in_path_width & in_front & in_range

            if np.any(in_path):
                distances = arc_len[in_path]
                best_sub_idx = np.argmin(distances)
                min_dist = distances[best_sub_idx]

                in_path_indices = np.where(in_path)[0]
                original_idx = confident_indices[in_path_indices[best_sub_idx]]
                best_key = keys[original_idx]

                # Fetch y_offset from local y coordinate
                y_offset = hly[in_path_indices[best_sub_idx]]

                ttc = min_dist / max(abs(vx), 0.01)
                self.publish_warning(ttc, self.map_data[best_key], y_offset)

    def publish_warning(self, ttc, cell_data, y_offset):
        msg = String()
        labels = cell_data.get("labels", [])

        # Select first non-unknown label, or "unknown"
        selected_label = "unknown"
        for lbl in labels:
            if lbl != "unknown":
                selected_label = lbl
                break

        msg.data = json.dumps({
            "ttc": float(ttc),
            "label": selected_label,
            "y_offset": float(y_offset)
        })
        self.warning_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AidaMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
