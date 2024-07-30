import cv2
import numpy as np
#from yolov8 import YOLOv8
from spot_movement import Movement


# Load YOLO
net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")  # Adjust paths as necessary
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture from MJPEG stream
cap = cv2.VideoCapture('http://137.146.188.250:8080/?action=stream')  # Replace with your IP

classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Create Spot object
USERNAME = "user"
PASS = "vd87k7o35nrs"
ROBOT_IP = "137.146.188.170"

spot = Movement(USERNAME, PASS, ROBOT_IP)
print(":::: Authenticated Spot!")

authenticatedFlag = False
powerFlag = False
standing = False

if spot.auth():
    authenticatedFlag = True
    print(":::: authenticated during power on")
else:
    print("::: spot was not authenticated")
spot.toggle_power()
powerFlag = True
print(":::: powered on")
spot.self_right()
spot.self_right()
spot.self_right()
spot.self_right()
print(":::: self right")
spot.stand()
print(":::: standing")

spot.sit()
spot.battery_change_pose()
spot.battery_change_pose()
spot.battery_change_pose()
spot.battery_change_pose()
spot.toggle_estop()
spot.toggle_power()
powerFlag = False
spot.toggle_lease()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if class_id == 0:
                    spot.sit()

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
