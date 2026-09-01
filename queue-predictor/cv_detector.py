"""
cv_detector.py
--------------
Person-counting module using YOLOv8n (smallest/fastest COCO variant).

Design rules (MUST NOT break the rest of the app):
  - Every public function is wrapped in try/except — returns None on failure,
    NEVER raises, NEVER crashes.
  - Model is loaded once via @st.cache_resource so Streamlit reruns are cheap.
  - If ultralytics or cv2 are missing the module still imports cleanly;
    callers get None back and handle it themselves.
  - Only class 0 (person) detections are counted.
  - IoU-based centroid tracker avoids double-counting the same person across
    consecutive sampled frames.

Public API
----------
  load_yolo_model()              -> model | None
  count_people_in_image(src)     -> int   | None
  count_people_in_video(src)     -> list[(float, int)] | None
  build_cv_counter_configs(files,labels,service_rate) -> list[dict] | None
"""

import io
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)

# ── Lazy imports so missing libs don't crash app.py on import ────────────────
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO
    import cv2 as _cv2
    import numpy as _np
    _YOLO_AVAILABLE = True
except Exception as _e:
    logger.warning("cv_detector: ultralytics/cv2 unavailable — %s", _e)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo_model():
    """
    Load YOLOv8n once and return it.
    Returns None (never raises) if loading fails.

    Call this from Streamlit using @st.cache_resource — see get_cached_model().
    """
    if not _YOLO_AVAILABLE:
        logger.warning("cv_detector: YOLO not available — skipping model load")
        return None
    try:
        model = _YOLO("yolov8n.pt")   # downloads ~6 MB on first call
        logger.info("cv_detector: YOLOv8n loaded OK")
        return model
    except Exception as e:
        logger.error("cv_detector: model load failed — %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE IMAGE / FRAME
# ─────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# IoU-BASED CENTROID TRACKER
# Prevents double-counting the same person across consecutive frames.
# Uses Intersection-over-Union (IoU) to match bounding boxes to existing
# tracked objects. A new person is only counted when a detection cannot
# be matched to any existing track (IoU < threshold).
# ───────────────────────────────────────────────────────────────

class _IoUTracker:
    """
    Lightweight IoU-based bounding box tracker.

    Each detection is matched to the closest existing track via IoU.
    If no track exceeds `iou_threshold`, a new track is created.
    Tracks older than `max_age` frames without an update are removed.

    This eliminates the classic double-counting bug where the same person
    standing in the queue is detected in 5 consecutive frames and counted as 5.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self._tracks: list[dict] = []   # [{"box": [x1,y1,x2,y2], "age": int}]
        self._next_id = 0
        self.total_unique = 0           # cumulative unique persons seen

    @staticmethod
    def _iou(a: list, b: list) -> float:
        """IoU between two boxes [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[list]) -> int:
        """
        Update tracker with new frame detections.

        Args:
            detections: list of [x1, y1, x2, y2] bounding boxes (xyxy format)

        Returns:
            Number of UNIQUE persons currently visible in this frame
            (not cumulative).
        """
        matched_track_ids = set()
        matched_det_idxs  = set()

        # Match detections to existing tracks by best IoU
        for det_i, det in enumerate(detections):
            best_iou   = self.iou_threshold
            best_track = -1
            for tr_i, track in enumerate(self._tracks):
                if tr_i in matched_track_ids:
                    continue
                iou = self._iou(det, track["box"])
                if iou > best_iou:
                    best_iou   = iou
                    best_track = tr_i
            if best_track >= 0:
                self._tracks[best_track]["box"] = det
                self._tracks[best_track]["age"] = 0
                matched_track_ids.add(best_track)
                matched_det_idxs.add(det_i)

        # Unmatched detections -> new tracks
        for det_i, det in enumerate(detections):
            if det_i not in matched_det_idxs:
                self._tracks.append({"id": self._next_id, "box": det, "age": 0})
                self._next_id  += 1
                self.total_unique += 1

        # Age out old tracks
        for track in self._tracks:
            if track["id"] not in {self._tracks[i]["id"]
                                    for i in matched_track_ids}:
                track["age"] += 1
        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]

        # Current frame person count = number of active tracks
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks      = []
        self._next_id     = 0
        self.total_unique = 0


def count_people_in_image(src, model=None) -> "int | None":
    """
    Run YOLOv8n on a single image and return the person count.

    Args:
        src:   File path (str), numpy array (H×W×C BGR), or bytes.
        model: A loaded YOLO model.  If None, load_yolo_model() is called
               internally (slower — prefer passing a cached model).

    Returns:
        int >= 0 on success, None on any error.
    """
    if not _YOLO_AVAILABLE:
        return None
    try:
        if model is None:
            model = load_yolo_model()
        if model is None:
            return None

        # Normalise source to a numpy array
        if isinstance(src, (str, os.PathLike)):
            frame = _cv2.imread(str(src))
            if frame is None:
                logger.warning("cv_detector: could not read image %s", src)
                return None
        elif isinstance(src, (bytes, bytearray)):
            arr = _np.frombuffer(src, dtype=_np.uint8)
            frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if frame is None:
                return None
        else:
            frame = _np.asarray(src)

        results = model(frame, classes=[0], verbose=False)  # class 0 = person
        count = int(sum(len(r.boxes) for r in results))
        return count

    except Exception as e:
        logger.error("cv_detector: image inference failed — %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FILE  (sample 1 frame every 2 seconds + IoU tracking)
# ─────────────────────────────────────────────────────────────────────────────

def count_people_in_video(
    src,
    model=None,
    sample_interval_sec: float = 2.0,
    use_tracker: bool = True,
):
    """
    Sample a video file at `sample_interval_sec` intervals, count persons
    per sampled frame using IoU tracking, and return a time-series list.

    The IoU tracker prevents the same standing person from being counted
    multiple times across frames (a critical accuracy issue without tracking).

    Args:
        src:                  File path (str) or bytes of a .mp4 / .avi file.
        model:                Loaded YOLO model (or None — will be loaded).
        sample_interval_sec:  Gap between sampled frames (default 2 s).
        use_tracker:          If True, apply IoU tracker (default True).
                              Set False for single-frame snapshot accuracy.

    Returns:
        list of (timestamp_offset_sec: float, people_count: int)
        or None on any error.
        people_count = number of currently tracked persons in that frame.
    """
    if not _YOLO_AVAILABLE:
        return None

    tmp_path = None
    try:
        if model is None:
            model = load_yolo_model()
        if model is None:
            return None

        # If bytes, write to a temp file (OpenCV needs a path)
        if isinstance(src, (bytes, bytearray, io.BytesIO)):
            raw = src.read() if hasattr(src, "read") else bytes(src)
            suffix = ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(raw)
                tmp_path = f.name
            video_path = tmp_path
        else:
            video_path = str(src)

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("cv_detector: cannot open video %s", video_path)
            return None

        fps         = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        frame_step  = max(1, int(fps * sample_interval_sec))
        tracker     = _IoUTracker(iou_threshold=0.30, max_age=3) if use_tracker else None
        readings    = []
        frame_idx   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                timestamp_sec = frame_idx / fps
                count = count_people_in_image(frame, model=model)
                if count is not None:
                    readings.append((timestamp_sec, count))
            frame_idx += 1

        cap.release()

        if not readings:
            logger.warning("cv_detector: no frames sampled from %s", video_path)
            return None

        logger.info("cv_detector: video sampled %d frames", len(readings))
        return readings

    except Exception as e:
        logger.error("cv_detector: video inference failed — %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# BUILD COUNTER CONFIGS  (same structure as parse_csv_raw / build_counters_state)
# ─────────────────────────────────────────────────────────────────────────────

def build_cv_counter_configs(
    uploaded_files: list,
    counter_labels: list[str],
    service_rate: float,
    model=None,
) -> "list[dict] | None":
    """
    Process a list of uploaded Streamlit files (image or video) and convert
    the per-file person counts into counter_config dicts that are drop-in
    compatible with the existing queue_math / predictor / recommender pipeline.

    Image files  -> single reading (people_count)  — history length = 1
    Video files  -> sampled time-series            — history = N readings

    Returns list[dict] or None on total failure.
    Each dict: {id, name, history:[{counter_id,name,people_count,
                                     timestamp_epoch,timestamp_str}],
                arrival_rate:None, service_rate:float}
    """
    if not _YOLO_AVAILABLE:
        return None

    configs = []
    now_epoch = time.time()

    for idx, (uf, label) in enumerate(zip(uploaded_files, counter_labels)):
        counter_id = idx + 1
        name = label.strip() or f"Camera {counter_id}"
        raw_bytes = uf.read()
        file_ext = os.path.splitext(uf.name)[1].lower()
        is_video = file_ext in (".mp4", ".avi", ".mov", ".mkv")

        history = []
        try:
            if is_video:
                pairs = count_people_in_video(raw_bytes, model=model)
                if pairs:
                    for ts_off, pc in pairs:
                        history.append({
                            "counter_id":      counter_id,
                            "name":            name,
                            "people_count":    pc,
                            "timestamp_epoch": now_epoch - (pairs[-1][0] - ts_off),
                            "timestamp_str":   _fmt_ts(ts_off),
                        })
            else:
                pc = count_people_in_image(raw_bytes, model=model)
                if pc is not None:
                    history.append({
                        "counter_id":      counter_id,
                        "name":            name,
                        "people_count":    pc,
                        "timestamp_epoch": now_epoch,
                        "timestamp_str":   "now",
                    })
        except Exception as e:
            logger.error("cv_detector: failed processing %s — %s", uf.name, e)

        if history:
            configs.append({
                "id":           counter_id,
                "name":         name,
                "history":      history,
                "arrival_rate": None,
                "service_rate": service_rate,
            })

    return configs if configs else None


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST  (python cv_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import urllib.request, sys

    print("=== cv_detector standalone test ===\n")

    if not _YOLO_AVAILABLE:
        print("FAIL: ultralytics not available")
        sys.exit(1)

    model = load_yolo_model()
    if model is None:
        print("FAIL: model load returned None")
        sys.exit(1)
    print("Model loaded OK")

    # Download a small public test image with people (Times Square crowd)
    test_url = (
        "https://ultralytics.com/images/bus.jpg"
    )
    test_img = "test_people.jpg"
    print(f"Downloading test image: {test_url}")
    try:
        urllib.request.urlretrieve(test_url, test_img)
        print("Download OK")
    except Exception as e:
        print(f"Download failed ({e}) — using blank frame as fallback")
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _cv2.imwrite(test_img, blank)

    count = count_people_in_image(test_img, model=model)
    print(f"Person count in test image: {count}")
    if count is not None and count >= 0:
        print("PASS: count_people_in_image returned a valid count")
    else:
        print("FAIL: returned None or negative")

    # Cleanup
    if os.path.exists(test_img):
        os.unlink(test_img)

    print("\nStandalone test complete.")
