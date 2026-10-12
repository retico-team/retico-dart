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
    overlay=True,  # the only line that changes: True or False
)

m3 = LabeledConvert(num_obj_to_display=10)
m4 = ScreenModule(fps=30)

m1.subscribe(m2)
m2.subscribe(m3)
m3.subscribe(m4)

network.run(m1)
print("Press Enter to exit")
input()
network.stop(m1)
