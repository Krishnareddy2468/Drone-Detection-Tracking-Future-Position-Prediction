# Drone Detection, Tracking and Trajectory Prediction
## System Architecture and Flowcharts


## Overview

A real-time pipeline that detects drones in a camera feed, tracks each one across frames, and forecasts its path 1 to 2 seconds ahead. It runs at about 48 fps on a laptop.

- **Detect**: find drone-sized objects in every frame
- **Track**: hold a stable identity per drone
- **Predict**: forecast where each drone is heading

**Scope.** The diagrams show the full fielded design (multi-camera EO/IR, edge hardware, fusion). The prototype implements the detect-track-predict core on a single video. Solid arrows are data flow; dashed arrows are feedback.


## Diagram 1: System Context

```mermaid
flowchart LR
    subgraph SENSORS["Inputs"]
        EO["EO Camera<br/>(visible light)"]
        IR["IR / Thermal Camera"]
        RF["Radar / RF<br/>(optional cue)"]
    end

    subgraph PIPELINE["Perception Pipeline"]
        CORE["Detect → Track → Classify<br/>Localise → Predict → Intent"]
    end

    subgraph OUTPUTS["Outputs"]
        OP["Operator Dashboard<br/>(boxes, IDs, paths)"]
        EFF["Effector<br/>(jammer / PTZ slew)"]
        LOG["Audit Log"]
    end

    EO -->|frames| CORE
    IR -->|thermal frames| CORE
    RF -.->|bearing cue| CORE
    CORE -->|tracks + alerts| OP
    CORE -->|target state| EFF
    CORE --> LOG
    EFF -.->|slew command| EO
```

Sensors feed the pipeline; it outputs tracks, alerts, and cueing. The prototype uses one EO video; the rest is the production system around it.


## Diagram 2: End-to-End Pipeline

```mermaid
flowchart TD
    A["1. INPUT<br/>Multi-camera EO/IR streams<br/>pose + timestamp"]
    B["2. PREPROCESS<br/>Denoise · CLAHE · Stabilise<br/>Normalise · Tile small targets"]
    C["3. DETECT<br/>YOLO per camera<br/>boxes + confidence"]
    D["4. FUSE<br/>Multi-camera track-level fusion<br/>triangulate to a common frame"]
    E["5. TRACK<br/>ByteTrack assigns IDs<br/>Kalman gate · lifecycle"]
    K["6. MOTION FEATURES<br/>position · velocity · acceleration<br/>heading · per-track state history"]
    F["7. CLASSIFY<br/>Per-track airframe type<br/>Quad · Hex · Fixed-wing · Unknown"]
    G["8. LOCALISE<br/>Image to geodetic<br/>lat · lon · alt · speed · heading"]
    H["9. PREDICT<br/>Kalman + LSTM / Transformer<br/>future path + uncertainty"]
    I["10. BEHAVIOUR<br/>Hovering · loitering<br/>approaching · anomalous"]
    J["11. OUTPUT<br/>Tracks · classes · alerts · API"]

    A --> B --> C --> D --> E --> K --> F --> G --> H --> I --> J
    H -.->|next-frame position hint| E
```

The eleven stages from input to output. The dashed arrow feeds the predictor's next-frame estimate back to the tracker. The prototype implements detection (3), tracking (5), motion-feature extraction (6) and prediction (9) in a single-camera, image-coordinate form; fusion (4), classification (7), localisation (8) and behaviour (10) are production stages, detailed under Production-Grade Extensions.


## Diagram 3: Detection Stage

```mermaid
flowchart TD
    FRAME["Full-Resolution Frame"]

    FRAME -->|production path| TILE["Sliced Inference<br/>Tiles at native resolution<br/>Small targets stay visible"]
    FRAME -->|prototype path| WHOLE["Whole-frame resize<br/>to detector input size"]

    TILE --> YOLO["YOLOv8 / YOLO11<br/>Backbone + FPN/PAN neck<br/>P2 head for tiny objects"]
    WHOLE --> YOLO

    YOLO --> NMS["Non-Maximum Suppression<br/>Merge tile results<br/>Recall-biased NMS"]

    IR["IR / Thermal Frame"] -.->|warm-target mask| FUSE
    NMS --> FUSE["EO + IR + temporal fusion"]
    FUSE --> OUT["Boxes + Confidence<br/>to Tracking"]
```

Sliced inference and a P2 head keep small targets detectable. NMS is biased toward recall, because a detection dropped here cannot be recovered downstream. The prototype runs whole-frame YOLO.


## Diagram 4: Tracking Stage (ByteTrack)

```mermaid
flowchart TD
    DET["Detections this frame"] --> SPLIT{Confidence split}

    SPLIT -->|"score ≥ 0.25"| HIGH["High-confidence set"]
    SPLIT -->|"0.10 to 0.25"| LOW["Low-confidence set"]

    PRED["Kalman predicted position<br/>for each active track"] --> P1

    HIGH --> P1["Pass 1: Match high-conf<br/>to existing tracks<br/>IoU + Hungarian algorithm"]
    P1 -->|unmatched tracks| P2["Pass 2: Match low-conf<br/>to unmatched tracks<br/>Recovers distant drones"]
    LOW --> P2

    P1 -->|matched| UPD["Update matched tracks"]
    P2 -->|matched| UPD
    P2 -->|new detection| NEW["Create tentative track"]

    UPD --> LC["Lifecycle update"]
    NEW --> LC

    REID["Appearance embedding<br/>(optional re-ID)"] -.->|extra cost term| P1
    CMC["Camera motion compensation"] -.->|de-jitter boxes| P1
```

Pass 2 matches the low-confidence boxes other trackers discard, which is where faint, distant drones appear.


## Diagram 5: Track Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Tentative : new unmatched detection
    Tentative --> Confirmed : matched for several consecutive frames
    Tentative --> Deleted : lost before confirmation
    Confirmed --> Coasting : detection missed this frame
    Coasting --> Confirmed : re-associated via prediction
    Coasting --> Deleted : lost for more than 30 frames
    Confirmed --> Deleted : track ended
    Deleted --> [*]
```

A track must persist several frames to be Confirmed, which filters one-frame false alarms. Coasting holds a track's identity for up to 30 frames through brief occlusion.


## Diagram 6: Prediction Stage

```mermaid
flowchart LR
    HIST["Track History<br/>last 10 positions"]

    HIST --> KF
    HIST --> LSTM

    subgraph KF["Physics Path (Kalman Filter)"]
        KF1["Predict next position<br/>using constant-acceleration<br/>motion model"]
        KF2["Roll forward H steps<br/>to get the full forecast path"]
        KF1 --> KF2
    end

    subgraph LSTM["Learned Path (LSTM Network)"]
        L1["Input: 10 frame-to-frame<br/>displacements (dx, dy)"]
        L2["LSTM · 64 hidden units<br/>(upgrade: temporal Transformer)<br/>Output: 15 future displacements"]
        L1 --> L2
    end

    KF2 -->|"cyan path"| OUT["Fused Future Path<br/>x(t+1) … x(t+H)<br/>+ uncertainty envelope"]
    L2 -->|"magenta path"| OUT
```

Two forecasts run in parallel: the Kalman filter (physics, stable) and the LSTM (learned, follows manoeuvres). Production upgrade: a temporal Transformer for swarms and longer horizons, with the filter as the deterministic anchor.


## Diagram 7: Input Pipeline and Sensor Synchronisation

```mermaid
flowchart LR
    EO["EO Camera"] --> SYNC
    IR["IR Camera"] --> SYNC
    POSE["Mount Pose + GNSS Clock"] --> SYNC

    SYNC["Hardware Sync<br/>PTP / GPS clock<br/>Hardware trigger"]

    SYNC --> CAL["Calibration<br/>Lens distortion correction<br/>EO to IR homography<br/>Align to common frame"]

    CAL --> PRE["Preprocessing<br/>Debayer → CLAHE<br/>Stabilize → Tile<br/>Resize → Normalize"]

    PRE --> TENSOR["Input Tensor<br/>(N × C × H × W)<br/>→ YOLO"]
```

Hardware-synchronised EO/IR, calibrated and registered, then preprocessed into the input tensor. The prototype runs resize and normalise only.


## Diagram 8: Deployment Architecture

```mermaid
flowchart TD
    subgraph FIELD["Field Unit"]
        CAM["PTZ EO/IR Camera<br/>(long-focal, steerable)"]
        EDGE["Edge Node<br/>Jetson-class GPU<br/>TensorRT-exported YOLO<br/>ByteTrack · Kalman · LSTM"]
        CAM -->|frames + pose| EDGE
    end

    EDGE -->|tracks + alerts via JSON/gRPC| BUS["Message Bus<br/>LAN / tactical radio"]
    BUS --> C2["C2 Dashboard<br/>Operator view"]
    BUS --> REC["Recording / Audit"]
    C2 -.->|slew-to-cue| CAM
    EDGE -.->|GPU saturated: Kalman-only fallback| EDGE
```

An edge node (Jetson-class, TensorRT) feeds a message bus to command-and-control. Degraded mode: if the GPU saturates, the Kalman filter runs alone rather than stalling.


## Latency Budget

End-to-end per-frame timing (Apple M3, YOLOv8n at 640 px, 432-frame average):

| Stage | Time |
|---|---|
| Capture + preprocess | ~1.0 ms |
| YOLO detect | ~6.4 ms |
| ByteTrack + Kalman | ~1.0 ms |
| LSTM predict | ~1.0 ms |
| Draw + encode | ~11.0 ms |
| **End-to-end** | **~20.9 ms (47.9 FPS)** |

The 30 fps budget is 33 ms, leaving about 12 ms of headroom for EO/IR fusion and tiling in production.


## Production-Grade Extensions

The stages below take the core pipeline from a working demo toward a fielded system. They are part of the architecture, not the prototype code.

### Diagram 10: Multi-Camera Fusion

```mermaid
flowchart TD
    CA["Camera A (EO)<br/>detect + track"]
    CB["Camera B (EO)<br/>detect + track"]
    CT["Thermal camera<br/>detect + track"]

    CA --> ASSOC
    CB --> ASSOC
    CT --> ASSOC

    ASSOC["Track-to-track association<br/>match the same drone<br/>across cameras"]
    ASSOC --> TRI["Triangulation<br/>two or more views<br/>give a 3D position"]
    TRI --> CI["Covariance intersection<br/>weight each sensor<br/>by its uncertainty"]
    CI --> FT["Single fused track<br/>in a common 3D frame"]
```

Late, track-level fusion across separated cameras. Triangulation recovers 3D position; covariance intersection weights each sensor by its uncertainty.

### Diagram 11: Target Classification

```mermaid
flowchart LR
    TRK["Confirmed track<br/>sequence of crops"]
    TRK --> CLS["Per-frame classifier<br/>CNN on the crop"]
    CLS --> VOTE["Temporal vote<br/>accumulate over the track"]
    VOTE --> DEC{"Confident<br/>enough?"}
    DEC -->|yes| LAB["Class + confidence<br/>Quad · Hex · Fixed-wing · VTOL"]
    DEC -->|no| UNK["Unknown aerial object<br/>(open-set)"]
```

Classification runs per track, not per frame: a temporal vote builds confidence. An open-set unknown class flags unfamiliar airframes instead of mislabelling them.

### Diagram 12: World-Coordinate Estimation

```mermaid
flowchart LR
    PX["Target pixel (u, v)<br/>from the track"] --> RAY
    CAL["Camera calibration"] --> RAY
    RAY["Back-project to a ray<br/>in the camera frame"]
    INS["IMU / INS orientation"] --> ROT
    RAY --> ROT["Rotate into<br/>the world frame"]
    RNG["Range source<br/>stereo · size · terrain"] --> POS
    GPS["Platform GPS"] --> POS
    ROT --> POS["Place along the ray<br/>at the estimated range"]
    POS --> GEO["Geodetic state<br/>lat · lon · alt<br/>speed · heading<br/>+ uncertainty"]
```

Pixel to geodetic state using calibration, GPS, INS and a range source. Range is the hard part for one camera; the output carries an uncertainty ellipse. This is the largest gap between prototype and deployment.

### Diagram 13: Behaviour and Intent Analysis

```mermaid
flowchart TD
    TRAJ["Track history +<br/>predicted path"]
    ASSET["Protected asset<br/>geometry / geofence"]

    TRAJ --> GEOM["Geometric tests (rules)"]
    ASSET --> GEOM
    GEOM --> B1["Hovering"]
    GEOM --> B2["Loitering"]
    GEOM --> B3["Approaching / leaving"]

    TRAJ --> ANOM["Learned anomaly model"]
    ANOM --> B4["Anomalous"]

    B1 --> ST["Behaviour state<br/>+ priority"]
    B2 --> ST
    B3 --> ST
    B4 --> ST
    ST --> OUT["To operator<br/>and alerting"]
```

Hovering, loitering and approach are deterministic geometric rules against the protected asset; anomalous behaviour uses a learned model of normal traffic.


## Summary Table

| Diagram | Shows | Prototype |
|---|---|---|
| 1: System context | Sensors in, outputs out | Pipeline only |
| 2: End-to-end pipeline | All 10 stages | Stages 3, 5, 8 |
| 3: Detection | YOLO, small-target strategy, NMS | Whole-frame YOLO |
| 4: Tracking | ByteTrack two-pass association | Implemented |
| 5: Track lifecycle | Tentative, Confirmed, Coasting | Implemented |
| 6: Prediction | Kalman + LSTM (upgrade: Transformer) | Implemented |
| 7: Input pipeline | Sensor sync, calibration, preprocess | Resize/normalise |
| 8: Deployment | Edge node, bus, degraded mode | Design only |
| 10: Multi-camera fusion | Track-level fusion, triangulation | Design only |
| 11: Classification | Per-track airframe type, unknown class | Design only |
| 12: World coordinates | Image to geodetic, with uncertainty | Design only |
| 13: Behaviour / intent | Hover, loiter, approach, anomaly | Design only |
