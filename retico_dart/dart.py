"""
DART Module
==============

This module implements the DART (Dynamic Affect Representation and Transformation) model for affective computing.
It provides a framework for representing and transforming affective states in a dynamic manner.
"""

from retico_core import abstract, UpdateType, UpdateMessage
from retico_vision.vision import ImageIU, DetectedObjectsIU
from PIL import Image
import torch
from sam3.model_builder import (
    build_sam3_image_model,
    build_pruned_sam3_image_model,
    load_pruned_config,
)
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.efficient_backbone import build_efficientsam3_model
from sam3.coco_classes import COCO_CLASSES
import numpy as np
from collections import deque
import threading
import time
from PIL import Image, ImageDraw
import retico_core


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
        imgsz=512,
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
        **kwargs,
    ):
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

            state = self.dart_model.set_image(image)
            results = self.dart_model.predict(
                state,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                per_class_nms=self.per_class_nms,
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


from retico_vision.vision import Convert_DetectedObjectsIU_ImageIU


class LabeledConvert(Convert_DetectedObjectsIU_ImageIU):
    """Same as Convert_DetectedObjectsIU_ImageIU, except it draws the real
    detected class name (iu.labels[i]) instead of the hardcoded 'Object N'
    placeholder, when labels are available."""

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            elif isinstance(iu, DetectedObjectsIU):
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
