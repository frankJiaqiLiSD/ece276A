## Project 1: Orientation Tracking
This project implements an orientation tracking algorithm to estimate the 3D orientation of a rigid body using data from an Inertial Measurement Unit (IMU). The goal is to fuse measurements from a gyroscope and an accelerometer to compute a stable and accurate orientation estimate over time, comparing it against ground truth data (e.g., from a Vicon motion capture system).

The core of the project involves formulating the problem as an optimization task and solving it using Gradient Descent on the manifold of unit quaternions (or rotation matrices).