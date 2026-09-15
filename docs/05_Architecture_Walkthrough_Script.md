# Architecture Walkthrough Script (3 to 4 minutes)

A screen-recording script for explaining the system architecture diagrams. Open
`01_System_Architecture_and_Flowcharts.pdf` and scroll to each diagram as you speak.
Each section has **SHOW** (what to put on screen) and **SAY** (what to narrate).
Target pace is about 150 words per minute.

---

## 0. Opening (about 15 seconds)

**SHOW:** Title page of the architecture PDF.

**SAY:** "This is the system architecture for a real-time drone detection, tracking,
and trajectory prediction pipeline, designed for counter-UAS use on edge hardware.
I will walk through the key diagrams: the overall flow, then detection, tracking,
and prediction, and finally the production extensions."

---

## 1. System Context (about 25 seconds)

**SHOW:** Diagram 1, System Context.

**SAY:** "At the top level, sensors come in on the left: an EO camera, a thermal
camera, and an optional radar or RF cue. The perception pipeline sits in the middle
and produces three outputs: tracks and alerts for the operator, target state for an
effector such as a jammer or a steered camera, and an audit log. The demo runs the
middle block on a single video; everything around it is the production system."

---

## 2. End-to-End Pipeline (about 45 seconds)

**SHOW:** Diagram 2, End-to-End Pipeline.

**SAY:** "This is the spine of the system, ten stages from input to output. Frames
are preprocessed, then YOLO detects drone-sized objects. Detections are fused across
cameras, then tracked so each drone keeps a stable identity. Each track is then
classified by airframe type and localised into real-world coordinates. The predictor
forecasts where each drone is going, a behaviour stage decides whether it is
hovering, loitering, or approaching, and the output goes to the dashboard and the
API. The dashed arrow is important: the predictor feeds its next-frame estimate back
to the tracker so it knows where to search. In the prototype I implement detection,
tracking, and prediction; the other stages are the production design."

---

## 3. Detection (about 30 seconds)

**SHOW:** Diagram 3, Detection Stage.

**SAY:** "The hardest part of this problem is that a drone at range can be only a few
pixels. So instead of shrinking the whole frame, the production path uses sliced
inference, running the detector on full-resolution tiles, plus a P2 head that is
dedicated to tiny objects. One design choice matters here: the non-maximum
suppression is biased toward recall, because a detection dropped at this stage can
never be recovered downstream. The prototype runs whole-frame YOLO."

---

## 4. Tracking (about 35 seconds)

**SHOW:** Diagram 4, Tracking Stage (ByteTrack).

**SAY:** "Tracking uses ByteTrack. The detections are split by confidence. Pass one
matches the confident boxes to existing tracks using overlap and the Kalman-predicted
position. Pass two is the key idea: it takes the low-confidence boxes that most
trackers throw away and uses them to rescue tracks, which is exactly where faint,
distant drones live. A track has to persist for several frames before it is
confirmed, and that confirmation gate is the main defence against false alarms like a
single-frame flicker or a bird crossing once."

---

## 5. Prediction, the core (about 45 seconds)

**SHOW:** Diagram 6, Prediction Stage.

**SAY:** "Prediction is the core of the system, and it runs two forecasts in
parallel. The physics path is a Kalman filter: it is fast, stable, and never predicts
anything physically impossible, but it lags on a sharp turn. The learned path is an
LSTM trained on real motion: it picks up the manoeuvres the filter cannot anticipate.
Both paths are drawn on the video, the Kalman path in cyan and the LSTM path in
magenta, so you can see them directly. The documented upgrade is a temporal
Transformer, which becomes the better choice once we forecast many interacting drones
in a swarm, with the filter still acting as the deterministic anchor."

---

## 6. Production Extensions and Latency (about 25 seconds)

**SHOW:** Scroll through Diagrams 10 to 13 and the Latency Budget table.

**SAY:** "Beyond the core, the architecture documents four production stages:
multi-camera fusion for triangulating 3D position, per-track classification with an
unknown class, conversion to real-world geodetic coordinates with uncertainty, and
behaviour analysis. And on timing, the whole pipeline runs in about 21 milliseconds a
frame, around 48 frames per second, which clears the 30 fps real-time budget with
headroom to spare."

---

## 7. Close (about 10 seconds)

**SHOW:** The Summary Table.

**SAY:** "So that is the full design: a modular detect, track, and predict pipeline,
built for real-time edge use, with a clear path from this prototype to a fielded
system. Next I will show it actually running."

---

### Quick reference: order of diagrams to show

1. Diagram 1, System Context
2. Diagram 2, End-to-End Pipeline (spend the most time here)
3. Diagram 3, Detection
4. Diagram 4, Tracking
5. Diagram 6, Prediction (the core)
6. Diagrams 10 to 13 + Latency table (brief)
7. Summary Table (close)
