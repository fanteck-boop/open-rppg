import rppg
import time
import cv2
import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model = rppg.Model()

with model.video_capture(0):
    last_send = 0

    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        now = time.time()
        if now - last_send > 1.0:
            result = model.hr(start=-30)
            if result and result.get('hr'):
                sqi = float(result.get('SQI', 0))
                if sqi > 0.3:
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
            last_send = now

        if box is not None:
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break