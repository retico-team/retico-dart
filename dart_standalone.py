"""
Standalone DART webcam test -- no retico involved.

Opens the webcam directly with OpenCV, runs DART on each frame, and draws
boxes + labels with OpenCV's own drawing functions. Useful for isolating
whether a problem is in DART/sam3 itself versus in the retico pipeline
around it.

Press 'q' in the display window to quit.
"""

import cv2
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

CLASSES = ["person", "car", "bicycle", "dog"]
CHECKPOINT = None  # None = auto-download from HF (requires access)
LOAD_FROM_HF = True  # set False to use random weights (no HF needed)
IMGSZ = 1008
CONFIDENCE = 0.3
NMS = 0.7

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Building model...")
model = build_sam3_image_model(
    device=device,
    checkpoint_path=CHECKPOINT,
    load_from_HF=LOAD_FROM_HF,
    eval_mode=True,
    enable_segmentation=False,  # detection_only, no masks
)

predictor = Sam3MultiClassPredictorFast(
    model,
    device=device,
    resolution=IMGSZ,
    use_fp16=(device == "cuda"),
    detection_only=True,
)

print("Setting classes...")
predictor.set_classes(CLASSES)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Running. Press 'q' to quit.")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    state = predictor.set_image(frame_rgb)
    results = predictor.predict(
        state,
        confidence_threshold=CONFIDENCE,
        nms_threshold=NMS,
    )

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().tolist()
    class_names = results["class_names"]

    for box, score, name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame_bgr,
            f"{name} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imshow("DART standalone", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
