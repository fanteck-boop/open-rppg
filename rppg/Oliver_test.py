import rppg
import time
import cv2
import os
from datetime import datetime

# ── Settings ──────────────────────────────────────────────
BASELINE_DURATION   = 120   # seconds
LOG_FOLDER          = r"C:\Users\trist\Desktop\LAN_Sessions"
SQI_THRESHOLD       = 0.5
IBI_MIN             = 300   # anything below = bad signal
# ──────────────────────────────────────────────────────────

os.makedirs(LOG_FOLDER, exist_ok=True)

# Auto-increment session number
session_number = 1
while os.path.exists(os.path.join(LOG_FOLDER, f"Session_{session_number:02d}_{datetime.now().strftime('%Y-%m-%d')}.txt")):
    session_number += 1

log_filename = f"Session_{session_number:02d}_{datetime.now().strftime('%Y-%m-%d')}.txt"
log_path     = os.path.join(LOG_FOLDER, log_filename)

# ── Baseline state ─────────────────────────────────────────
baseline_ibi       = 0
baseline_rmssd     = 0
baseline_lfhf      = 0
baseline_ready     = False
collecting         = True
baseline_start     = time.time()
baseline_ibi_sum   = 0
baseline_rmssd_sum = 0
baseline_lfhf_sum  = 0
baseline_samples   = 0

previous_label = ""
entry_count    = 0

# ── Helpers ────────────────────────────────────────────────
def get_arousal_label(score):
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    return "High"

def calculate_arousal(ibi, rmssd, lfhf):
    ibi_score   = max(0, min(1, (baseline_ibi - ibi) / baseline_ibi)) if baseline_ibi > 0 else 0
    rmssd_score = max(0, min(1, 1 - (rmssd / baseline_rmssd)))        if baseline_rmssd > 0 else 0
    lfhf_score  = max(0, min(1, lfhf / (baseline_lfhf * 3)))          if baseline_lfhf > 0 else 0
    return (ibi_score * 0.5) + (rmssd_score * 0.3) + (lfhf_score * 0.2)

def write_header(f):
    f.write("==============================================\n")
    f.write(f"  Session: Session {session_number:02d}\n")
    f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Participant: LAN test\n")
    f.write("==============================================\n\n")
    f.write(f"{'Time':<10} {'Event':<20} {'Arousal':<10} {'Score':<8} "
            f"{'HR':<8} {'IBI':<8} {'RMSSD':<8} {'LF/HF':<8} "
            f"{'Breathing':<12} {'SQI':<6}\n")
    f.write("-" * 100 + "\n")

def log_entry(f, event, label, score, hr, ibi, rmssd, lfhf, breathing, sqi):
    global entry_count
    elapsed  = time.time() - session_start
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    line = (f"{time_str:<10} {event:<20} {label:<10} {score:<8.3f} "
            f"{hr:<8.1f} {ibi:<8.1f} {rmssd:<8.1f} {lfhf:<8.2f} "
            f"{breathing:<12.3f} {sqi:<6.3f}\n")
    f.write(line)
    f.flush()
    entry_count += 1

# ── Start ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Session {session_number:02d} — LAN Arousal Monitor")
print(f"  Log: {log_path}")
print(f"{'='*50}\n")
print(f"[BASELINE] Sit still and relax for {BASELINE_DURATION} seconds...")
print(f"[BASELINE] Starting now...\n")

session_start = time.time()
model = rppg.Model()

# ✅ UTF-8 FIX
with open(log_path, "w", encoding="utf-8") as log_file:
    write_header(log_file)
    log_file.write(f"{'00:00:00':<10} {'BASELINE START':<20}\n")
    log_file.flush()

    last_send      = 0
    last_countdown = 0
    arousal_score  = 0
    arousal_label  = "Low"

    # ✅ FIXED VIDEO CAPTURE (no crash)
    cap = model.video_capture(1)
    cap.__enter__()

    try:
        for frame, box in model.preview:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            now   = time.time()

            # ── Baseline countdown ─────────────────────────
            if collecting:
                elapsed_baseline = now - baseline_start
                remaining        = BASELINE_DURATION - elapsed_baseline
                countdown        = int(remaining)

                if countdown != last_countdown:
                    print(f"[BASELINE] {countdown}s remaining — samples: {baseline_samples}", end="\r")
                    last_countdown = countdown

                if elapsed_baseline >= BASELINE_DURATION:
                    baseline_ibi   = baseline_ibi_sum   / baseline_samples if baseline_samples > 0 else 800
                    baseline_rmssd = baseline_rmssd_sum / baseline_samples if baseline_samples > 0 else 60
                    baseline_lfhf  = baseline_lfhf_sum  / baseline_samples if baseline_samples > 0 else 1

                    collecting     = False
                    baseline_ready = True

                    elapsed  = now - session_start
                    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

                    log_file.write(f"\n{time_str:<10} BASELINE END\n\n")
                    log_file.write("  -- Baseline Summary --\n")
                    log_file.write(f"  IBI:     {baseline_ibi:.1f} ms\n")
                    log_file.write(f"  RMSSD:   {baseline_rmssd:.1f} ms\n")
                    log_file.write(f"  LF/HF:   {baseline_lfhf:.2f}\n")
                    log_file.write(f"  Samples: {baseline_samples}\n\n")
                    log_file.flush()

                    print(f"\n\n[BASELINE] Complete!")
                    print(f"  IBI:     {baseline_ibi:.1f}")
                    print(f"  RMSSD:   {baseline_rmssd:.1f}")
                    print(f"  LF/HF:   {baseline_lfhf:.2f}")
                    print(f"\n[MONITORING] Started...\n")

            # ── Process data every second ──────────────────
            if now - last_send > 1.0:
                result = model.hr(start=-10)

                if result and result.get('hr'):
                    hrv       = result.get('hrv', {})
                    hr        = float(result['hr'])
                    sqi       = float(result.get('SQI', 0))
                    ibi       = float(hrv.get('ibi', 0))
                    rmssd     = float(hrv.get('rmssd', 0))
                    lfhf      = float(hrv.get('LF/HF', 0))
                    breathing = float(hrv.get('breathingrate', 0))

                    # ✅ CLEAN BASELINE DATA
                    if collecting and ibi > IBI_MIN and sqi > SQI_THRESHOLD:
                        baseline_ibi_sum   += ibi
                        baseline_rmssd_sum += rmssd
                        baseline_lfhf_sum  += lfhf
                        baseline_samples   += 1

                    if baseline_ready:
                        if sqi > SQI_THRESHOLD and ibi > IBI_MIN:
                            arousal_score = calculate_arousal(ibi, rmssd, lfhf)
                            arousal_label = get_arousal_label(arousal_score)

                            log_entry(log_file, "DATA", arousal_label, arousal_score,
                                      hr, ibi, rmssd, lfhf, breathing, sqi)

                            print(f"HR: {hr:.1f} | Arousal: {arousal_label} ({arousal_score:.3f}) | SQI: {sqi:.3f}")
                        else:
                            print(f"[WEAK SIGNAL] SQI: {sqi:.3f}")

                last_send = now

            # ── Draw face box ──────────────────────────────
            if box is not None:
                y1, y2 = box[0]
                x1, x2 = box[1]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Arousal Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ❗ DO NOT call cap.__exit__() (this causes crash)
        pass

    # ── End session ────────────────────────────────────────
    elapsed  = time.time() - session_start
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    log_file.write("\n" + "-" * 100 + "\n")
    log_file.write(f"Session ended: {datetime.now()}\n")
    log_file.write(f"Duration: {time_str}\n")
    log_file.write(f"Entries: {entry_count}\n")

cv2.destroyAllWindows()

print(f"\n[DONE] Session saved to: {log_path}")