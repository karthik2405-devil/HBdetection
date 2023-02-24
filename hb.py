import cv2
import numpy as np

# Load the image
img = cv2.imread('D:/Desktop/webdev/HBdetection/body.jfif')

# Load OpenPose model
net = cv2.dnn.readNetFromTensorflow('path/to/openpose.pb')

# Define body part indices
BODY_PARTS = {
    "Head": (0, 1),
    "Neck": (1, 2),
    "Right Shoulder": (2, 3),
    "Right Elbow": (3, 4),
    "Right Wrist": (4, 5),
    "Left Shoulder": (2, 6),
    "Left Elbow": (6, 7),
    "Left Wrist": (7, 8),
    "Mid Hip": (1, 11),
    "Right Hip": (11, 12),
    "Right Knee": (12, 13),
    "Right Ankle": (13, 14),
    "Left Hip": (11, 15),
    "Left Knee": (15, 16),
    "Left Ankle": (16, 17),
    "Chest": (1, 8),
    "Background": (0,)
}

# Perform body part detection
blob = cv2.dnn.blobFromImage(img, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(blob)
out = net.forward()

# Extract key points
points = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv2.minMaxLoc(heatMap)
    x = int(img.shape[1] * point[0] / out.shape[3])
    y = int(img.shape[0] * point[1] / out.shape[2])
    points.append((x, y) if conf > 0.1 else None)

# Label body parts
for part, (p1, p2) in BODY_PARTS.items():
    if points[p1] and points[p2]:
        cv2.line(img, points[p1], points[p2], (0, 255, 0), 3)
        cv2.circle(img, points[p1], 5, (0, 0, 255), -1)
        cv2.circle(img, points[p2], 5, (0, 0, 255), -1)
        cv2.putText(img, part, points[p1], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

# Display the image
cv2.imshow('Detected Body Parts', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
