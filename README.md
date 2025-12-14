# Computer Vision Aimbot

Developed a high-performance computer vision application utilizing the YOLO (You Only Look Once) architecture for real-time object detection and classification within dynamic environments.

To address initial latency constraints, the system was optimized using multithreading and parallel processing, significantly reducing inference time. The project features advanced post-processing logic, including:
- Heuristic Filtering: Implementation of distance-based prioritization and skeletal segmentation to eliminate false positives (e.g., distinguishing targets from background noise).
- Motion Control Algorithms: Design of mouse smoothing and input damping functions to simulate organic human reaction curves and stabilize vector tracking.

The software demonstrates high precision and responsiveness, successfully operating under strict timing constraints.
