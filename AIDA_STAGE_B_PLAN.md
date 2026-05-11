# AIDA Stage B (Impact Verification) Implementation Plan

## 1. Overview
The goal of Stage B is to process high-frequency (100Hz) Z-axis linear acceleration data from the RRC Lite STM32 IMU to isolate sharp mechanical impacts (potholes/speed bumps) from continuous chassis and engine noise. This is achieved using a Discrete Wavelet Transform (DWT) and specific thresholding logic within a ROS 2 Node.

The hardware constraints are locked to the MentorPi A1 platform (Ackermann steering, non-Mecanum, Raspberry Pi 5):
* **Wheelbase:** 0.213m
* **Track Width:** 0.145m
* **Wheel Width:** 0.025m
* **Wheel Radius:** 0.0325m

## 2. Workspace Setup
The workspace will be organized under `aida_ws/src/` as a ROS 2 Python package named `aida_core`.

**Directory Structure:**
```text
aida_ws/
├── src/
│   └── aida_core/
│       ├── aida_core/
│       │   ├── __init__.py
│       │   └── impact_verifier_node.py       # Core logic for Stage B
│       ├── config/
│       │   └── aida_kinematics.yaml          # All kinematic constraints and threshold configs
│       ├── package.xml                       # ROS 2 package dependencies (rclpy, std_msgs, sensor_msgs, etc.)
│       ├── setup.py                          # ROS 2 python setup file
│       └── setup.cfg
```

## 3. Data Acquisition
* **Topic:** Subscribe to `/imu/data` (`sensor_msgs/msg/Imu`).
* **Frequency:** 100Hz publication from the RRC Lite STM32 IMU.
* **Buffering:** We will maintain a rolling window buffer of `Z-axis` linear acceleration values. The window size will be exactly **64 samples** ($2^6$), which perfectly accommodates PyWavelets DWT processing with optimal efficiency on the Raspberry Pi 5.

## 4. Signal Processing
* **Wavelet Choice:** **Daubechies 4 (`db4`)**. `db4` has excellent properties for identifying sharp, transient signals (like sudden mechanical shocks from hitting a pothole) while filtering out periodic harmonic vibrations (like continuous rolling/engine noise).
* **Processing Flow:**
  1. Once the 64-sample buffer is full, apply the DWT using PyWavelets to decompose the Z-axis acceleration signal into approximation (low-frequency) and detail (high-frequency) coefficients.
  2. Isolate the detail coefficients that correspond to the sharp transients.
* **Thresholding Logic (Rough Road vs. Pothole):**
  * **Initial Testing:** A fixed absolute threshold applied to the detail coefficients. The threshold value will be dynamically loaded from `aida_kinematics.yaml`. If the maximum magnitude in the detail coefficients exceeds this threshold, an impact is declared.
  * **Stretch Goal (Dynamic):** Implement a dynamic threshold based on $N \times \sigma$ standard deviations above a rolling mean of the detail coefficients. This adaptively distinguishes a true pothole spike from a generally rough road surface (where $\sigma$ and mean naturally elevate but no singular spike occurs).

## 5. ROS 2 Integration
* **Node Structure:** `impact_verifier_node.py` will inherit from `rclpy.node.Node`.
  * **Subscriber:** `/imu/data` (QoS profile optimized for high frequency).
  * **Publisher:** `/aida/hazard_trigger` publishing `std_msgs/msg/Header` containing the timestamp of the impact verification.
* **Parameters (`aida_kinematics.yaml`):**
  * `wheelbase: 0.213`
  * `track_width: 0.145`
  * `wheel_width: 0.025`
  * `wheel_radius: 0.0325`
  * `impact_threshold_fixed: 2.5` (example value, to be tuned)
  * `impact_threshold_dynamic_n_sigma: 3.0` (stretch goal parameter)
  * `window_size: 64`
* **Lifecycle:** Node startup will declare and read parameters, initialize the PyWavelets rolling buffer, and begin the subscription loop. Any verified impact will trigger an `RCLCPP_INFO` log and publish the `Header` to `/aida/hazard_trigger`.
