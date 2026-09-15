# Video Explanation

## Common Explanation

This demo shows our drone detection, tracking and trajectory prediction pipeline.

YOLO detects the object in each frame. ByteTrack gives the object a stable ID. Then the system stores the center points of the object and predicts its future path using Kalman filter and LSTM.

```text
Video -> YOLO Detection -> ByteTrack Tracking -> Kalman + LSTM Prediction -> Output Video
```

## What To Point Out

- Green box: detected object
- ID number: tracked object ID
- Cyan line: Kalman predicted path
- Magenta line: LSTM predicted path
- Rings: prediction uncertainty
- Behaviour label: hover, transit, approach, depart or loiter
- FPS: real-time speed

## 1. Drone3 Video

This video shows the full pipeline on a drone clip.

The detector finds the object, the tracker follows it with an ID, and the system predicts where it may move next. The cyan and magenta lines show the future trajectory prediction.

## 2. Drone 2 Video

This video has more movement and more detections.

It shows that the system can track multiple objects separately. Each object gets its own ID, and the trajectory is predicted for active tracks.

## 3. DroneVideo1 Video

This video is useful to explain stable tracking.

The system keeps following the object over many frames. It stores the center-point history and uses that history to predict the future path.

## 4. Birds Video

This is a comparison video.

Birds can look similar to drones, so this video shows why training data is important. The tracking and prediction still work, but better hard-negative training can reduce false detections.

## Important Note

The current trained model shows numeric class labels like `0` and `1`. Detection and tracking still work. Only the display label needs to be renamed later.

## Closing Line

Overall, the system detects the object, tracks it over time, stores its motion history, and predicts its future trajectory.
