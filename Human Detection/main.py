from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Mediapipe face detection and landmark solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Load the YOLO weights and configuration
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the COCO class labels
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])

MP_HAND_WEIGHT = 1
MP_FACE_WRIGHT = 1
YOLO_WEIGHT = 1

def main():
    ####
    # Code for capturing frames from video stream
    video_capture = cv2.VideoCapture(0)
    ###
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        
        # Create a copy of the frame to output
        out_frame = frame.copy()
        
        frame_count = (frame_count+1)%4
        
        # Detecting hands with mediapipe library
        _hands = detect_hands_mp(frame, out_frame)
            
        if _hands:
            print("Hands detected")
        
        else:
            # Detecting faces with mediapipe library
            _faces = detect_face_mp(frame, out_frame)
        
            if _faces:
                print("Faces detected")
            
            else:
                if frame_count == 3:
                    # Detecting humans using YOLO algorithm (opencv)
                    _humans = detect_human_yolo(frame, out_frame)
                    
                    if _humans:
                        print("Humans detected")

        # Display the resulting frame
        cv2.imshow('Video', out_frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()


def detect_hands_mp(frame: np.ndarray, out_frame: np.ndarray) -> Tuple[int, Tuple]:

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5) as hands:
        # Process the image with MediaPipe Hand model
        result = hands.process(image_rgb)
    
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(out_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Calculate bounding box coordinates for the hand
                x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
                y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
                x_min = min(x_coordinates)
                y_min = min(y_coordinates)
                
                # Calculate the position for the text
                text_x = int(x_min * frame.shape[1])
                text_y = int(y_min * frame.shape[0]) - 10
            
                # Add annotation to the frame
                cv2.putText(out_frame, 'Hand', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            return 1
        else:
            return 0
        

def detect_face_mp(frame: np.ndarray, out_frame: np.ndarray) -> Tuple[int, Tuple]:
  
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
        mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = face_detection.process(rgb_frame)

        # Process the detected faces
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Get the coordinates of the bounding box
                x, y, w, h = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
                             int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])

                # Verify if the bounding box coordinates are within the frame dimensions
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    # Draw a rectangle around the head
                    cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Add the annotation text above the head rectangle
                    cv2.putText(out_frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # Detect facial landmarks within the head region
                    face_landmarks = face_mesh.process(rgb_frame[y:y + h, x:x + w])

                    # Draw the facial landmarks within the head region
                    if face_landmarks.multi_face_landmarks:
                        for landmarks in face_landmarks.multi_face_landmarks:
                            # Draw the landmarks
                            for idx, landmark in enumerate(landmarks.landmark):
                                cx, cy = int(landmark.x * w), int(landmark.y * h)
                                cv2.circle(out_frame, (x + cx, y + cy), 2, (255, 0, 0), -1)
            return 1
        else:
            return 0


def detect_human_yolo(frame: np.ndarray, out_frame: np.ndarray) -> Tuple[int]:
  
    # Perform object detection with YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    detected = 0
    
    # Process the detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Class ID 0 represents humans in COCO dataset
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Apply non-maximum suppression to remove redundant detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes around detected humans
    for i in indices:
        # i = i[0]
        x, y, width, height = boxes[i]
        cv2.rectangle(out_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(out_frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        detected = 1
    
    return detected


if __name__ == "__main__":
    main()
