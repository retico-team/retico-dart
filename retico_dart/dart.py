"""
DART Module
==============

This module implements the DART (Detect Anything in Real Time) model for
real-time, open-vocabulary object detection.

Live overlay (optional, off by default)
---------------------------------------
DART needs ~200 ms per frame at imgsz=1008, so a screen that only shows
frames DART has finished looks choppy, and its boxes trail behind anything
that moves. ``DartModule(..., overlay=True)`` fixes that:
"""

# --- standard library ---
import threading
import time
from collections import deque

# --- OpenCV ---
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None

# --- retico ---
import retico_core
from retico_core import abstract, UpdateMessage, UpdateType
from retico_vision.vision import (
    Convert_DetectedObjectsIU_ImageIU,
    DetectedObjectsIU,
    ImageIU,
)

# --- DART / SAM3 ---
from sam3.coco_classes import COCO_CLASSES
from sam3.efficient_backbone import build_efficientsam3_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)


def _as_pil(image):
    return image if hasattr(image, "convert") else Image.fromarray(image)


class LiveOverlay:
    """Draws DART's latest boxes on live camera frames and keeps them glued
    to moving objects.

    A helper class. DartModule uses it when
    ``overlay=True``. All methods must be called from ONE thread.

    How the tracking works: it remembers the last second or so of frames
    (small grayscale copies). When a detection arrives for frame F, it picks
    corner points inside each box on F and follows them with optical flow
    through every frame captured since, so the boxes "catch up" to the
    current frame. Between DART results it keeps following the video the
    same way. A brand-new object still needs one DART inference to appear.
    """

    def __init__(
        self,
        num_obj_to_display=15,
        max_box_age=1.0,
        show_scores=False,
        box_color="red",
        box_width=2,
        track=True,
        track_width=320,
        max_track_points=40,
        motion_threshold=0.5,
        history_frames=60,
        verbose=False,
    ):
        """
        Args:
            num_obj_to_display (int): max boxes to draw per frame.
            max_box_age (float): seconds after which old detections are
                dropped (so stale boxes don't linger if DART stalls).
                None = never drop.
            show_scores (bool): append the confidence to each label.
            box_color (str): outline/text color.
            box_width (int): outline width in pixels.
            track (bool): move boxes along with the video using optical
                flow (needs OpenCV). False = just draw the latest boxes.
            track_width (int): frames are shrunk to this width for tracking
                (keeps it to ~1-2 ms per frame).
            max_track_points (int): corner points followed per box.
            motion_threshold (float): a point counts as "moving" if it moves
                more than this many pixels (in the shrunk image) in one
                frame. Static background inside a box is ignored this way.
            history_frames (int): recent frames remembered so a late
                detection can be caught up (60 is ~2 s at 30 fps).
            verbose (bool): print how old each detection was and how long
                the catch-up took.
        """
        self.num_obj_to_display = num_obj_to_display
        self.max_box_age = max_box_age
        self.show_scores = show_scores
        self.box_color = box_color
        self.box_width = box_width
        self.track = track
        self.track_width = track_width
        self.max_track_points = max_track_points
        self.motion_threshold = motion_threshold
        self.verbose = verbose

        if self.track and cv2 is None:
            print(
                "[LiveOverlay] OpenCV (cv2) not found -> tracking disabled. "
                "Install it with: pip install opencv-python"
            )
            self.track = False

        # (N, 4) float array in full-resolution pixels; moved by the tracker.
        self._boxes = np.empty((0, 4), dtype=np.float32)
        self._labels = []
        self._scores = []
        self._boxes_time = 0.0

        self._hist = deque(maxlen=history_frames)  # (frame key, small gray)
        self._pts = []  # per box: (K, 1, 2) float32 points, or None
        self._scale = 1.0  # small image size / full image size

    # ---- public API --------------------------------------------------
    def add_frame(self, image, key=None):
        """Call for every live frame, in order. Moves the current boxes to
        this frame and remembers the frame (``key`` identifies it so a later
        detection can say which frame it was computed from)."""
        if not self.track:
            return
        try:
            gray = self._small_gray(_as_pil(image))
            if self._hist and self._hist[-1][1].shape != gray.shape:
                # Camera resolution changed: old frames are useless.
                self._hist.clear()
                self._pts = []
            if self._hist:
                self._advance(self._hist[-1][1], gray)
            self._hist.append((key, gray))
        except Exception as e:  # never let tracking kill the video
            self._disable_tracking(e)

    def set_detections(
        self, boxes, labels=None, scores=None, source_key=None, source_time=None
    ):
        """A new DART result. ``source_key`` is the key given to add_frame()
        for the frame DART looked at; ``source_time`` is when that frame was
        captured (only used for the verbose printout)."""
        # Always copy: the tracker edits these in place.
        self._boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        self._labels = list(labels or [])
        self._scores = list(scores or [])
        self._boxes_time = time.time()
        self._pts = []

        if self.track and len(self._boxes) and self._hist:
            try:
                self._catch_up(source_key, source_time)
            except Exception as e:
                self._disable_tracking(e)

    def draw(self, image):
        """Returns a COPY of ``image`` with the current boxes drawn on it."""
        image = _as_pil(image).copy()

        fresh = len(self._boxes) > 0 and (
            self.max_box_age is None
            or (time.time() - self._boxes_time) <= self.max_box_age
        )
        if fresh:
            draw = ImageDraw.Draw(image)
            for i, box in enumerate(self._boxes[: self.num_obj_to_display]):
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle(
                    [x1, y1, x2, y2], outline=self.box_color, width=self.box_width
                )
                label = self._labels[i] if i < len(self._labels) else f"Object {i+1}"
                if self.show_scores and i < len(self._scores):
                    label = f"{label} {self._scores[i]:.2f}"
                draw.text((x1, y1), label, fill=self.box_color)
        return image

    # ---- tracking internals -----------------------------------------
    def _catch_up(self, source_key, source_time):
        """The boxes describe frame F (the one DART looked at). Seed points
        on F, then replay the optical flow through every frame since."""
        t0 = time.perf_counter()
        hist = list(self._hist)

        start = None
        if source_key is not None:
            for idx in range(len(hist) - 1, -1, -1):
                if hist[idx][0] == source_key:
                    start = idx
                    break
        found = start is not None
        if not found:
            # Source frame unknown / too old: assume the newest frame.
            start = len(hist) - 1

        self._pts = [self._seed_points(hist[start][1], b) for b in self._boxes]
        for k in range(start + 1, len(hist)):
            self._advance(hist[k - 1][1], hist[k][1])

        if self.verbose:
            age_ms = (
                (time.time() - source_time) * 1000
                if source_time is not None
                else float("nan")
            )
            n_pts = sum(len(p) for p in self._pts if p is not None)
            print(
                f"[overlay] detection {age_ms:5.0f} ms old "
                f"({len(hist) - 1 - start} frames behind"
                f"{'' if found else ', source frame not found'}), "
                f"caught up in {(time.perf_counter() - t0) * 1000:4.1f} ms, "
                f"{len(self._boxes)} boxes, {n_pts} points"
            )

    def _small_gray(self, image):
        w, h = image.size
        self._scale = min(1.0, self.track_width / float(w))
        gray = np.asarray(image.convert("L"))
        if self._scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, round(w * self._scale)), max(1, round(h * self._scale))),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(gray)

    def _seed_points(self, gray, box):
        """Pick corner points to follow inside a box (shrunk by 10% per side
        so most of them land on the object, not the background)."""
        h, w = gray.shape
        x1, y1, x2, y2 = (box * self._scale).tolist()
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        x1, y1 = int(max(0, x1 + mx)), int(max(0, y1 + my))
        x2, y2 = int(min(w, x2 - mx)), int(min(h, y2 - my))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        pts = cv2.goodFeaturesToTrack(
            gray[y1:y2, x1:x2],
            maxCorners=self.max_track_points,
            qualityLevel=0.01,
            minDistance=3,
        )
        if pts is None:
            return None
        pts = pts.reshape(-1, 2).astype(np.float32) + np.array(
            [x1, y1], dtype=np.float32
        )
        return pts.reshape(-1, 1, 2)

    def _advance(self, prev_gray, gray):
        """Follow every box's points from prev_gray to gray and shift the
        boxes by how far the *moving* points travelled."""
        owners = [i for i, p in enumerate(self._pts) if p is not None and len(p)]
        if not owners:
            return

        all_pts = np.concatenate([self._pts[i] for i in owners], axis=0)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            all_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        ok_all = status.reshape(-1).astype(bool)

        offset = 0
        inv_scale = 1.0 / self._scale
        for i in owners:
            n = len(self._pts[i])
            ok = ok_all[offset : offset + n]
            old = self._pts[i][:, 0, :][ok]
            new = new_pts[offset : offset + n, 0, :][ok]
            offset += n

            if len(new) < 3:
                # Lost it. The box stays put until the next DART result.
                self._pts[i] = None
                continue

            d = new - old
            moving = np.hypot(d[:, 0], d[:, 1]) > self.motion_threshold
            # Only trust the shift if enough points actually moved. A box
            # around a moving object often contains static background
            # points; ignoring the non-movers keeps the box on the object.
            if moving.sum() >= max(3, 0.25 * len(d)):
                dx, dy = np.median(d[moving], axis=0) * inv_scale
                self._boxes[i] += np.array([dx, dy, dx, dy], dtype=np.float32)

            self._pts[i] = new.reshape(-1, 1, 2).astype(np.float32)

    def _disable_tracking(self, err):
        print(
            f"[LiveOverlay] tracking failed ({err!r}); falling back to plain overlay."
        )
        self.track = False
        self._pts = []
        self._hist.clear()


class DartModule(abstract.AbstractModule):

    @staticmethod
    def name():
        return "DART Module"

    @staticmethod
    def description():
        return "A module implementing the DART model."

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return DetectedObjectsIU

    def __init__(
        self,
        classes=None,
        coco=False,
        checkpoint=None,
        track=False,
        compile_mode=None,
        trt_backbone_engine=None,
        trt_enc_dec_engine=None,
        trt_max_classes=4,
        text_cache=None,
        efficient_backbone=None,
        efficient_model=None,
        imgsz=1008,
        confidence_threshold=0.3,
        nms_threshold=0.7,
        per_class_nms=True,
        detection_only=True,
        warmup=0,
        split_backbone=False,
        split_block=20,
        cuda_graphs=False,
        skip_blocks=None,
        mask_blocks=None,
        class_agnostic_nms=None,
        track_thresh=0.5,
        match_thresh=0.5,
        max_time_lost=30,
        presence_threshold=0.05,
        shared_encoder=False,
        generic_prompt="object",
        single_pass=False,
        class_method="cosine",
        prototype_path=None,
        load_from_hf=True,  # REMOVE LATER
        overlay=False,
        overlay_options=None,
        **kwargs,
    ):
        """
        overlay (bool): if True, DartModule also sends every live camera
            frame with the latest boxes drawn on it (ImageIU), tracked along
            with the video. Connect it straight to a ScreenModule. Default
            False: only DetectedObjectsIU is sent, as before.
        overlay_options (dict): settings for the overlay, e.g.
            {"show_scores": True, "max_box_age": 0.5, "track": False,
             "verbose": True}. See LiveOverlay for all options.
        (All other arguments are DART's own settings.)
        """
        super().__init__(**kwargs)

        if not classes and not coco:
            raise ValueError("Classes must be specified or COCO must be enabled.")
        if single_pass and shared_encoder:
            raise ValueError("single_pass and shared_encoder are mutually exclusive.")
        if class_method not in ("cosine", "attention", "prototype"):
            raise ValueError(
                "class_method must be one of 'cosine', 'attention', 'prototype'."
            )
        if class_method == "prototype" and prototype_path is None:
            raise ValueError(
                "prototype_path is required when class_method='prototype'."
            )
        if trt_enc_dec_engine and not detection_only:
            raise ValueError(
                "trt_enc_dec_engine requires detection_only=True -- the "
                "TRT enc-dec engine doesn't produce hidden states for mask "
                "generation, only boxes+scores."
            )

        self.classes = classes
        self.coco = coco
        self.checkpoint = checkpoint
        self.track = track
        self.compile_mode = compile_mode
        self.trt_backbone_engine = trt_backbone_engine
        self.trt_enc_dec_engine = trt_enc_dec_engine
        self.trt_max_classes = trt_max_classes
        self.text_cache = text_cache
        self.efficient_backbone = efficient_backbone
        self.efficient_model = efficient_model
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.per_class_nms = per_class_nms
        self.detection_only = detection_only
        self.warmup = warmup
        self.split_backbone = split_backbone
        self.split_block = split_block
        self.cuda_graphs = cuda_graphs
        self.skip_blocks = skip_blocks
        self.mask_blocks = mask_blocks
        self.class_agnostic_nms = class_agnostic_nms
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.presence_threshold = presence_threshold
        self.shared_encoder = shared_encoder
        self.generic_prompt = generic_prompt
        self.single_pass = single_pass
        self.class_method = class_method
        self.prototype_path = prototype_path
        self.load_from_hf = load_from_hf  # REMOVE LATER
        self.dart_model = None
        self.queue = deque(maxlen=1)
        self._detector_thread_active = False

        # Optional live overlay. The detector thread hands each result to
        # retico's frame thread through a one-slot mailbox, and only the
        # frame thread touches the overlay, so no locks are needed.
        self.overlay = overlay
        self._overlay = LiveOverlay(**(overlay_options or {})) if overlay else None
        self._overlay_results = deque(maxlen=1)
        self._overlay_prev_iu = None
        self._overlay_iu_counter = 0

    def setup(self):
        """
        Setup the DART model with the specified parameters.
        This method initializes the DART model and prepares it for processing input images.
        """
        # Initialize the DART model here
        self.dart_model = self.initialize_dart_model()

    def initialize_dart_model(self):
        """
        Initialize the DART model with the specified parameters.
        This method should load the model weights, set up the inference engine, and prepare the model for processing.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.efficient_backbone:
            dart_model = build_efficientsam3_model(
                backbone=self.efficient_backbone,
                model=self.efficient_model,
                checkpoint=self.checkpoint,
                device=device,
                enable_segmentation=not self.detection_only,
                eval_mode=True,
            )
        pruning_config = (
            load_pruned_config(self.checkpoint) if self.checkpoint else None
        )
        if pruning_config is not None:
            dart_model = build_pruned_sam3_image_model(
                checkpoint_path=self.checkpoint,
                pruning_config=pruning_config,
                device=device,
                eval_mode=True,
                enable_segmentation=not self.detection_only,
                skip_blocks=self.skip_blocks,
            )
        else:
            dart_model = build_sam3_image_model(
                device=device,
                checkpoint_path=self.checkpoint,
                load_from_HF=self.load_from_hf,
                eval_mode=True,
                enable_segmentation=not self.detection_only,
                skip_blocks=self.skip_blocks,
                mask_blocks=self.mask_blocks,
            )

        if self.imgsz != 1008:
            pos_enc = dart_model.backbone.vision_backbone.position_encoding
            pos_enc.precompute_for_resolution(self.imgsz)

        predictor = Sam3MultiClassPredictorFast(
            model=dart_model,
            device=device,
            resolution=self.imgsz,
            compile_mode=self.compile_mode,
            use_fp16=True,
            presence_threshold=self.presence_threshold,
            shared_encoder=self.shared_encoder,
            generic_prompt=self.generic_prompt,
            single_pass=self.single_pass,
            class_method=self.class_method,
            prototype_path=self.prototype_path,
            detection_only=self.detection_only,
            trt_engine_path=self.trt_backbone_engine,
            trt_enc_dec_engine_path=self.trt_enc_dec_engine,
            trt_max_classes=self.trt_max_classes,
        )

        class_names = COCO_CLASSES if self.coco else self.classes
        if self.text_cache:
            predictor.set_classes(class_names, text_cache=self.text_cache)
        else:
            predictor.set_classes(class_names)

        if self.warmup > 0:
            dummy = Image.new("RGB", (self.imgsz, self.imgsz))
            for _ in range(self.warmup):
                state = predictor.set_image(dummy)
                predictor.predict(
                    state,
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold,
                )
            if device == "cuda":
                torch.cuda.synchronize()

        return predictor

    def set_classes(self, classes):
        """Update the open-vocabulary class list at runtime. Recomputing
        text embeddings is cheap (milliseconds); the TRT engines don't
        need to be rebuilt UNLESS len(classes) > self.trt_max_classes."""
        if len(classes) > self.trt_max_classes:
            raise ValueError(
                f"{len(classes)} classes exceeds trt_max_classes="
                f"{self.trt_max_classes}. Rebuild the enc-dec engine with "
                f"a higher --max-classes, or raise trt_max_classes and "
                f"drop TRT (pure PyTorch) if that's acceptable for your "
                f"latency budget."
            )
        self.classes = classes
        if self.dart_model is not None:
            self.dart_model.set_classes(classes)

    def process_update(self, update_message):
        """Just hand the newest frame off to the detector thread. This
        never blocks on DART inference, so retico's own frame-reading
        loop is never held up waiting for a prediction to finish."""
        for iu, ut in update_message:
            if ut != UpdateType.ADD:
                continue
            if not isinstance(iu, ImageIU):
                continue
            self.queue.append(iu)
            if self._overlay is not None:
                self._send_overlay_frame(iu)

    def _send_overlay_frame(self, input_iu):
        """overlay=True: forward this live frame with the latest boxes drawn
        on it. Runs on retico's frame thread (never the detector thread)."""
        ov = self._overlay
        # 1) move the current boxes along to this frame and remember it
        ov.add_frame(input_iu.image, input_iu.iuid)
        # 2) if DART finished a frame since last time, take its boxes and
        #    catch them up from the frame DART saw to this one
        while self._overlay_results:
            ov.set_detections(*self._overlay_results.popleft())
        # 3) draw on a copy (the detector thread may be reading the original)
        image = ov.draw(input_iu.image)

        output_iu = ImageIU(
            creator=self,
            iuid=f"{hash(self)}:img:{self._overlay_iu_counter}",
            previous_iu=self._overlay_prev_iu,
            grounded_in=input_iu,
        )
        output_iu.set_image(image, input_iu.nframes or 1, input_iu.rate or 0)
        self._overlay_iu_counter += 1
        self._overlay_prev_iu = output_iu
        self.append(UpdateMessage.from_iu(output_iu, UpdateType.ADD))

    def _detector_thread(self):
        """Runs in the background, independent of retico's update loop.
        Always pulls whatever frame is currently queued -- since the
        queue is a maxlen=1 deque, that's always the newest one."""
        while self._detector_thread_active:
            if not self.queue:
                time.sleep(0.01)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image

            frame_age = 1000 * (
                time.time() - input_iu.created_at
            )  # ms since the webcam made this frame

            t0 = time.perf_counter()
            state = self.dart_model.set_image(image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            results = self.dart_model.predict(
                state,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                per_class_nms=self.per_class_nms,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            print(
                f"frame_age {frame_age:.0f} ms | set_image {1000*(t1-t0):.0f} ms | "
                f"predict {1000*(t2-t1):.0f} ms | total {1000*(t2-t0):.0f} ms"
            )
            print(
                results["class_names"],
                results["scores"].tolist() if len(results["scores"]) else [],
            )

            boxes = (
                results["boxes"].cpu().numpy()
                if len(results["scores"])
                else np.empty((0, 4))
            )

            output_iu = self.create_iu(input_iu)
            output_iu.set_detected_objects(image, boxes, "bb")
            output_iu.labels = list(results["class_names"])
            output_iu.confidences = results["scores"].cpu().tolist()
            output_iu.class_ids = results["class_ids"].cpu().tolist()
            # Tells LabeledConvert these boxes are already drawn on the live
            # frames (so it must not draw them again).
            output_iu.overlay_drawn = self._overlay is not None

            if self._overlay is not None:
                # mailbox to the frame thread (see _send_overlay_frame)
                self._overlay_results.append(
                    (
                        boxes,
                        output_iu.labels,
                        output_iu.confidences,
                        input_iu.iuid,
                        input_iu.created_at,
                    )
                )

            um = UpdateMessage.from_iu(output_iu, UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        """Called by retico right before the network starts running --
        this is where the detector thread gets started."""
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread, daemon=True).start()

    def shutdown(self):
        self._detector_thread_active = False
        self.dart_model = None


class LabeledConvert(Convert_DetectedObjectsIU_ImageIU):
    """Same as Convert_DetectedObjectsIU_ImageIU, except it draws the real
    detected class name (iu.labels[i]) instead of the hardcoded 'Object N'
    placeholder, when labels are available.

    Overlay-aware, so the SAME runner works whether DartModule uses
    overlay=True or overlay=False:
      * overlay=False: draws DART's detections on the frame DART looked at.
      * overlay=True:  DART already sends live frames with the boxes drawn, so
        those are passed straight through and the (flagged) detections are
        not drawn a second time.
    """

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU, ImageIU]

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            elif isinstance(iu, ImageIU):
                # Live frame from DartModule(overlay=True): boxes are already
                # drawn on it, so just pass it on.
                output_iu = self.create_iu(iu)
                output_iu.set_image(iu.image, iu.nframes or 1, iu.rate or 0)
                self.append(
                    retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                )
            elif isinstance(iu, DetectedObjectsIU):
                if getattr(iu, "overlay_drawn", False):
                    continue  # already drawn on the live frames by the overlay
                image: Image = iu.image

                output_iu = self.create_iu(iu)

                obj_type = iu.object_type
                num_objs = min(self.num_obj_to_display, iu.num_objects)
                labels = getattr(iu, "labels", None)

                if obj_type == "bb":
                    valid_boxes = iu.payload
                    for i in range(num_objs):
                        box = valid_boxes[i]
                        if box is not None:
                            x1, y1, x2, y2 = box
                            img_draw = ImageDraw.Draw(image)
                            img_draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                            label = (
                                labels[i]
                                if labels and i < len(labels)
                                else f"Object {i+1}"
                            )
                            img_draw.text((x1, y1), label, fill="red")
                elif obj_type == "seg":
                    valid_segs = iu.payload
                    for i in range(num_objs):
                        seg_mask = valid_segs[i]
                        if seg_mask is not None:
                            seg_mask_image = Image.fromarray(
                                seg_mask.astype("uint8") * 255
                            )
                            image = Image.composite(
                                image,
                                Image.new("RGB", image.size, (255, 0, 0)),
                                seg_mask_image,
                            )
                            img_draw = ImageDraw.Draw(image)
                            label = (
                                labels[i]
                                if labels and i < len(labels)
                                else f"Segmented Object {i+1}"
                            )
                            img_draw.text((10, 10 + i * 20), label, fill="red")
                else:
                    print("Object type is invalid. Can't retrieve segmented object.")
                    exit()

                output_iu.image = image
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
                self.append(um)
