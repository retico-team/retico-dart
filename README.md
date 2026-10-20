# retico_dart
A ReTico module for real-time, open-vocabulary object detection with DART (Detect Anything in Real Time), built on SAM 3. You give it a list of class names (or use the COCO classes) and it produces bounding boxes, labels and confidences for every webcam frame it processes.

DART needs ~200 ms per frame at `imgsz=1008`, so it runs inference on a background thread and always works on the newest frame. An optional live overlay draws the latest boxes on every camera frame and uses optical flow to keep them on moving objects between DART results.

## Installation and Requirements

To use the DART module you first need to install the retico-dart package:
* Install retico-dart:
```pip install git+https://github.com/retico-team/retico-dart.git```

DART depends on the `sam3` package (SAM 3 with the DART multi-class predictor). If it isn't pulled in automatically, install it manually:
* Install sam3:
```pip install git+<URL-of-the-DART-sam3-repo>```

For the live overlay's motion tracking you also need OpenCV. Without it the overlay still works, but boxes won't follow moving objects:
* Install OpenCV:
```pip install opencv-python```

  or install retico-dart with the overlay extra:
```pip install "retico-dart[overlay] @ git+https://github.com/retico-team/retico-dart.git"```

### Hardware
* A CUDA GPU is strongly recommended. DART falls back to CPU, but it will be far too slow for real time.
* TensorRT is only needed if you pass `trt_backbone_engine` / `trt_enc_dec_engine`.

## Modules

### `DartModule`
Subscribes to `ImageIU` (e.g. from `WebcamModule`) and runs DART on the newest frame in a background thread, producing a `DetectedObjectsIU` with bounding boxes (`object_type="bb"`). Frames that arrive while DART is busy are skipped, so the module never falls behind the camera.

Each `DetectedObjectsIU` also carries:
* `labels` (list[str]): the detected class name for each box
* `confidences` (list[float]): the score for each box
* `class_ids` (list[int]): the class index for each box
* `overlay_drawn` (bool): `True` if the boxes are already drawn on the live overlay frames

When `overlay=True`, the module **also** outputs an `ImageIU` for every incoming camera frame, with the latest boxes drawn on it.

#### Arguments:

**Classes**
* `classes` (list[str]): the class names to detect; required unless `coco=True`
* `coco` (bool): if `True`, detect the 80 COCO classes instead of `classes`; defaults to `False`
* `generic_prompt` (str): the generic prompt used by the shared encoder / single-pass modes; defaults to `"object"`
* `class_method` (str): how detections are assigned to classes; one of `"cosine"`, `"attention"`, `"prototype"`; defaults to `"cosine"`
* `prototype_path` (str): path to class prototypes; required when `class_method="prototype"`
* `text_cache` (str): path to cached text embeddings for the classes, to skip encoding them at startup; defaults to `None`

**Model loading**
* `checkpoint` (str): path to a model checkpoint; if the checkpoint has a pruning config, the pruned model is built; defaults to `None`
* `load_from_hf` (bool): download the SAM 3 weights from Hugging Face; defaults to `True`
* `efficient_backbone` (str): name of an EfficientSAM3 backbone to use; defaults to `None`
* `efficient_model` (str): EfficientSAM3 model variant (used with `efficient_backbone`); defaults to `None`
* `detection_only` (bool): skip the segmentation head and only predict boxes (faster); defaults to `True`
* `skip_blocks` (list[int]): backbone blocks to skip; defaults to `None`
* `mask_blocks` (list[int]): backbone blocks to mask; defaults to `None`

**Inference**
* `imgsz` (int): input resolution; defaults to `1008`
* `confidence_threshold` (float): minimum score for a detection to be kept; defaults to `0.3`
* `presence_threshold` (float): minimum class-presence score; defaults to `0.05`
* `nms_threshold` (float): IoU threshold for non-maximum suppression; defaults to `0.7`
* `per_class_nms` (bool): run NMS separately for each class; defaults to `True`
* `shared_encoder` (bool): encode the image once and share it across classes; cannot be used with `single_pass`; defaults to `False`
* `single_pass` (bool): detect all classes in one pass; cannot be used with `shared_encoder`; defaults to `False`
* `warmup` (int): number of dummy inferences to run at startup so the first real frames aren't slow; defaults to `0`

**Speed-ups**
* `compile_mode` (str): `torch.compile` mode for the predictor; defaults to `None` (no compile)
* `trt_backbone_engine` (str): path to a TensorRT engine for the backbone; defaults to `None`
* `trt_enc_dec_engine` (str): path to a TensorRT engine for the encoder-decoder; requires `detection_only=True`; defaults to `None`
* `trt_max_classes` (int): the max number of classes the TensorRT enc-dec engine was built for; defaults to `4`

**Live overlay**
* `overlay` (bool): if `True`, also send every live camera frame as an `ImageIU` with the latest boxes drawn on it, tracked along with the video; connect it straight to a screen module (or through `LabeledConvert`); defaults to `False`
* `overlay_options` (dict): settings for the overlay, e.g. `{"show_scores": True, "max_box_age": 0.5}`; see `LiveOverlay` below; defaults to `None`

#### Changing classes at runtime
Call `dart.set_classes(["person", "bottle"])` while the network is running. Recomputing the text embeddings only takes milliseconds. The number of classes can't exceed `trt_max_classes`.

### `LiveOverlay`
A helper used by `DartModule` when `overlay=True`; you configure it through `overlay_options`. It remembers the last ~2 seconds of frames, so when a DART result arrives for an older frame it follows the objects with optical flow through every frame since and the boxes "catch up" to the present. Between DART results it keeps moving the boxes the same way. A brand-new object still needs one DART inference to appear.
This might be necessary if a big delay is causing the camera to fail and/or be unable to properly provide labels.

### `LabeledConvert`
Subscribes to `DetectedObjectsIU` and `ImageIU` and outputs an `ImageIU` ready for a screen. It works like `Convert_DetectedObjectsIU_ImageIU` from `retico_vision`, but draws the real class name for each box instead of `Object N`.

The same network works with `overlay=True` or `overlay=False`:
* `overlay=False`: draws DART's boxes on the frame DART looked at
* `overlay=True`: passes the live overlay frames straight through and doesn't draw the detections a second time

### Example
**Note:** The example runner uses `WebcamModule` and `ScreenModule` from `retico_vision`. Set `overlay=False` to only see the frames DART finished.

```python
from retico_core import *
from retico_vision.vision import WebcamModule, ScreenModule
from retico_dart.dart import DartModule, LabeledConvert

m1 = WebcamModule(pil=True)

m2 = DartModule(
    classes=["person", "bottle", "cup", "book"],
    load_from_hf=True,
    checkpoint=None,
    imgsz=1008,
    confidence_threshold=0.3,
    nms_threshold=0.7,
    detection_only=True,
    overlay=True,
)

m3 = LabeledConvert(num_obj_to_display=10)
m4 = ScreenModule(fps=30)

m1.subscribe(m2)
m2.subscribe(m3)
m3.subscribe(m4)

network.run(m1)
input()
network.stop(m1)
```

### Example 2
```python
from retico_core import *
from retico_vision.vision import WebcamModule, ScreenModule
from retico_dart.dart import DartModule, LabeledConvert

m1 = WebcamModule(pil=True)

m2 = DartModule(
    classes=["person", "bottle", "cup", "book"],
    load_from_hf=True,
    checkpoint=None,
    imgsz=1008,
    confidence_threshold=0.3,
    nms_threshold=0.7,
    detection_only=True,
)

m3 = LabeledConvert(num_obj_to_display=10)
m4 = ScreenModule(fps=30)

m1.subscribe(m2)
m2.subscribe(m3)
m3.subscribe(m4)

network.run(m1)
input()
network.stop(m1)
```

To detect the COCO classes instead, use `DartModule(coco=True)`.

Citation
````
@article{carion2025sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Carion, Nicolas and others},
  journal={arXiv preprint arXiv:2511.16719},
  year={2025}
}
````
``````
