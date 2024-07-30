import cv2
import numpy as np
import sys
#from yolov8 import YOLOv8
from spot_movement import Movement
import threading

authenticatedFlag = False
powerFlag = False
standing = False

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

def turnon():
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
    spot.self_right()
    spot.self_right()
    spot.self_right()
    spot.self_right()
    print(":::: self right")
    spot.stand()
    print(":::: standing")

def turnoff():
    print(":::: Spot is sitting...")
    spot.sit()
    print(":::: Spot is rolling over...")
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()
    spot.battery_change_pose()

    print(":::: Spot is powering off...")
    spot.toggle_estop()
    spot.toggle_power()
    powerFlag = False
    spot.toggle_lease()
    print(":::: Spot is off")

def move_spot(direction):
    if direction == "turn_left":
        spot.turn_left()
    elif direction == "turn_right":
        spot.turn_right()
    elif direction == "move_forward":
        spot.move_forward()
    elif direction == "move_backward":
        spot.move_backward()
    elif direction == "strafe_left":
        spot.strafe_left()
    elif direction == "strafe_right":
        spot.strafe_right()


if __name__ == "__main__":
    # Load YOLO object detection model
    print(":::: Loading YOLO model...")
    net = cv2.dnn.readNet("darknet/yolov7-tiny.weights", "darknet/cfg/yolov7-tiny.cfg")  # Adjust paths as necessary
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    print(":::: YOLO model loaded!")

    # Set up video capture from MJPEG stream
    print(":::: Setting up video capture...")
    cap = cv2.VideoCapture('http://137.146.188.250:8080/?action=stream')  # Replace with your IP
    print(":::: Video capture set up!")

    # Create Spot object
    print(":::: Creating Spot Object...")
    USERNAME = "user"
    PASS = "vd87k7o35nrs"
    ROBOT_IP = "137.146.188.170"

    spot = Movement(USERNAME, PASS, ROBOT_IP)
    print(":::: Spot Object Created!")

    print(":::: Would you like to turn on Spot? (y/n)")
    user_input = input()

    if user_input.lower() == 'y':
        print(":::: Spot is turning on...")
        turnon()
        print(":::: Spot is on")
    elif user_input.lower() == 'n':
        print(":::: Exiting...")
        sys.exit()
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

    print(":::: Starting object detection...")
    print(":::: Press 'q' to quit")

    # frame_counter = 0
    # frame_skip = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_counter += 1

        # if frame_counter % frame_skip != 0:
        #     continue

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
                        if center_x < ((width / 2) - (width * 0.2)):
                            threading.Thread(target=move_spot, args=("turn_left",)).start()
                        if center_x > ((width / 2) + (width * 0.2)):
                            threading.Thread(target=move_spot, args=("turn_right",)).start()
                    

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

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            threading.Thread(target=move_spot, args=("move_forward",)).start()
            print('w')
        elif key == ord('s'):
            threading.Thread(target=move_spot, args=("move_backward",)).start()
            print('s')
        elif key == ord('a'):
            threading.Thread(target=move_spot, args=("strafe_left",)).start()
            print('a')
        elif key == ord('d'):
            threading.Thread(target=move_spot, args=("strafe_right",)).start()
            print('d')

    cap.release()
    cv2.destroyAllWindows()
    turnoff()