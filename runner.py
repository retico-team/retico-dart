from retico_core import *
from retico_vision.vision import WebcamModule
from retico_vision.vision import (
    WebcamModule,
    Convert_DetectedObjectsIU_ImageIU,
    ScreenModule,
)
from retico_dart.dart import DartModule

m1 = WebcamModule(pil=True)

m2 = DartModule(
    coco=True,
    load_from_hf=False,
    checkpoint=None,  # None = auto-download from HF
    confidence_threshold=0.3,
    nms_threshold=0.7,
    detection_only=True,  # boxes only, no masks -- matches
    # Convert_DetectedObjectsIU_ImageIU's
    # object_type='bb' drawing path
    fast=True,
    # Leave trt_backbone_engine / trt_enc_dec_engine unset for a first
    # run -- this falls back to eager PyTorch (slower, but nothing to
    # build first). Add them once you've built the TRT engines per the
    # README's Quick Start steps.
)

m3 = Convert_DetectedObjectsIU_ImageIU(num_obj_to_display=10)

m4 = ScreenModule(fps=30)

m1.subscribe(m2)
m2.subscribe(m3)
m3.subscribe(m4)

network.run(m1)

print("Press any key to exit")
input()

network.stop(m1)
