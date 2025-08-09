import cv2
import time
import torch  # pip install torch torchvision


class ObjectDetector:
    def __init__(self, activation_time=10, model_name="yolov5s"):
        """
        activation_time: seconds to keep detection active
        model_name: YOLOv5 model variant (yolov5s, yolov5m, yolov5l, yolov5x)
        """
        self.activation_time = activation_time
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.detected_object = "no object detected"

        # Target labels from COCO dataset
        self.target_labels = ["book", "cell phone"]

    def detect(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam.")
            return None

        start_time = time.time()
        end_time = start_time + self.activation_time

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            detections = results.pandas().xyxy[0]

            for _, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                # Only set detected_object if it hasn't been set to a target yet
                if self.detected_object == "no object detected" and label in self.target_labels:
                    self.detected_object = label

                # Draw detection boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            # Show current detected_object on screen
            cv2.putText(frame, f"Detected: {self.detected_object}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            cv2.imshow("YOLOv5 Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.detected_object


if __name__ == "__main__":
    detector = ObjectDetector(activation_time=10)
    result = detector.detect()
    print(f"Final Detected object: {result}")
