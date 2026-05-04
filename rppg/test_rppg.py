import rppg
import time
import cv2
import socket
import json
from collections import deque

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model = rppg.Model()

# ── SQI smoothing ─────────────────────────────────────────
SQI_HISTORY_SIZE = 5
sqi_history = deque(maxlen=SQI_HISTORY_SIZE)

# ── Face box smoothing (reduces bounding box jitter) ──────
BOX_SMOOTH_SIZE = 8
box_history = deque(maxlen=BOX_SMOOTH_SIZE)

def smooth_box(box):
    """Average the last N bounding boxes to reduce detector jitter."""
    if box is None:
        return None
    box_history.append(box)
    if len(box_history) < 2:
        return box
    avg_y1 = int(sum(b[0][0] for b in box_history) / len(box_history))
    avg_y2 = int(sum(b[0][1] for b in box_history) / len(box_history))
    avg_x1 = int(sum(b[1][0] for b in box_history) / len(box_history))
    avg_x2 = int(sum(b[1][1] for b in box_history) / len(box_history))
    return ((avg_y1, avg_y2), (avg_x1, avg_x2))

def get_smoothed_sqi(raw_sqi):
    """Maintain a rolling average of SQI to reduce frame-to-frame noise."""
    sqi_history.append(raw_sqi)
    return sum(sqi_history) / len(sqi_history)

# ─────────────────────────────────────────────────────────

with model.video_capture(0):
    last_send = 0

    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        now = time.time()
        if now - last_send > 1.0:

            # FIX 1: Longer window (15s) → more stable SQI computation
            result = model.hr(start=-15)

            if result and result.get('hr'):
                raw_sqi = float(result.get('SQI', 0))

                # FIX 2: Smooth SQI over last 5 readings
                sqi = get_smoothed_sqi(raw_sqi)

                # FIX 3: Raise threshold — only send high-quality signal
                if sqi > 0.6:
                    hrv = result.get('hrv', {})
                    payload = {
                        "hr":  round(float(result['hr']), 1),
                        "sqi": round(sqi, 3),
                        "hrv": {
                            "rmssd":         float(hrv.get('rmssd', 0)),
                            "sdnn":          float(hrv.get('sdnn', 0)),
                            "ibi":           float(hrv.get('ibi', 0)),
                            "lf_hf":         float(hrv.get('LF/HF', 0)),
                            "breathingrate": float(hrv.get('breathingrate', 0))
                        }
                    }
                    sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))
                    print(payload)
                else:
                    print(f"[SKIP] SQI too low: raw={raw_sqi:.3f}  smoothed={sqi:.3f}")

            last_send = now

        # FIX 4: Smooth the bounding box to reduce face detector jitter
        smooth = smooth_box(box)
        if smooth is not None:
            y1, y2 = smooth[0]
            x1, x2 = smooth[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break