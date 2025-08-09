import time
import torch
import cv2
import warnings
import ultralytics  # required for YOLOv5 hub loading
warnings.filterwarnings("ignore")


class GUN:
    def __init__(self, serial_inst=None, model="yolov5s", size=400, device='cpu'):
        # Load YOLOv5 model from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', model, pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.square_w = size
        self.square_h = size
        self.serialInst = serial_inst
        self.pgmCode = {"FIRE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPRAY": 5}

    @staticmethod
    def isBound(cx, cy, square_x, square_w, square_y, square_h):
        """Check if point (cx, cy) is inside given rectangle."""
        return square_x <= cx <= square_x + square_w and square_y <= cy <= square_y + square_h
    
    @staticmethod
    def dist(cx, cy, fx, fy):
        """Euclidean distance between (cx, cy) and (fx, fy)."""
        return ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
    
    def run(self, serial):
        self.serialInst = serial
        text = set()
        cap = cv2.VideoCapture(0)

        allowed_classes = ['person', 'cell phone', 'notebook']
        start_time = time.time()  # Start timer

        while True:
            # Auto-close after 10 seconds
            if time.time() - start_time >= 10:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            square_x = (frame_w - self.square_w) // 2
            square_y = (frame_h - self.square_h) // 2

            # YOLO detection
            results = self.model(frame)
            detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

            # Draw centered square
            cv2.rectangle(frame, (square_x, square_y),
                        (square_x + self.square_w, square_y + self.square_h),
                        (255, 0, 0), 2)

            # Filter detections by allowed classes
            people_idx = [i for i, det in enumerate(detections)
                        if self.model.names[int(det[5])] in allowed_classes]
            people = detections[people_idx]

            # Skip if no allowed objects
            if people.numel() == 0:
                cv2.imshow("YOLOv5 Detection (CPU)", frame)
                cv2.waitKey(1)
                continue

            # Find the biggest bounding box by area
            max_area = -1
            poi_idx = -1
            for idx, det in enumerate(people):
                x1, y1, x2, y2, conf, cls_id = det
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    poi_idx = idx

            # Draw all detections
            for det in people:
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = self.model.names[int(cls_id)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Highlight POI
            poi = people[poi_idx]
            x1, y1, x2, y2, conf, cls_id = poi
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Movement or fire command
            if self.isBound(cx, cy, square_x, self.square_w, square_y, self.square_h):
                text = {"FIRE"}
            else:
                text.clear()
                if cx <= square_x:
                    text.add('LEFT')
                elif cx >= square_x + self.square_w:
                    text.add('RIGHT')
                
                if cy <= square_y:
                    text.add('UP')
                elif cy >= square_y + self.square_h:
                    text.add('DOWN')

            # Show command
            cv2.putText(frame, " ".join(list(text)), (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            
            inst = list(text)[0]
            inst = str(self.pgmCode[inst]) + '\n'
            self.serialInst.write(inst.encode())
            
            # Show frame
            cv2.imshow("YOLOv5 Detection (CPU)", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mine = GUN(model="yolov5s", size=400, device='cpu')
    mine.run()