from ultralytics import YOLO
import cv2


class RunModel:

    def calculate_iou(self, box_1, box_2):
        x1, y1, x2, y2 = box_1
        x3, y3, x4, y4 = box_2

        # נקודת התחלה של החפיפה
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)

        # נקודת סיום של החפיפה
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        # רוחב וגובה החפיפה
        # max זה כמו ערך מוחלט
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)

        # שטח החפיפה
        intersection_area = inter_width * inter_height

        # שטח כל תיבה
        # max זה כמו ערך מוחלט למה צלע לא יכול להיות שלילי
        box_1_area = max(0, x2 - x1) * max(0, y2 - y1)
        box_2_area = max(0, x4 - x3) * max(0, y4 - y3)

        # השטח הכולל
        union_area = box_1_area + box_2_area - intersection_area

        # כדי לא לחלק ב0
        if union_area == 0:
            return 0

        iou = intersection_area / union_area

        return iou

    def extract_boxes(self, results):
        boxes = []

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            boxes.append([x1, y1, x2, y2])

        return boxes

    def are_results_similar(self, boxes_1, boxes_2, iou_threshold=0.5):
        # קודם בודקים שיש אותה כמות תיבות
        if len(boxes_1) != len(boxes_2):
            return False

        matched_boxes = 0

        for box_1 in boxes_1:
            found_match = False

            for box_2 in boxes_2:
                iou = self.calculate_iou(box_1, box_2)

                if iou >= iou_threshold:
                    found_match = True
                    break

            if found_match:
                matched_boxes += 1

        return matched_boxes == len(boxes_1)

    def run_model(self, image_path: str, show_table_image: bool, show_cordinates: bool):
        model = YOLO("runs/detect/size_arrow_model/weights/best.pt")

        confidence_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]

        previous_boxes = None

        for confidence in confidence_thresholds:
            results = model.predict(
                source=image_path,
                conf=confidence,
                show=False,
            )

            current_boxes = self.extract_boxes(results)

            # אם זו לא ההרצה הראשונה, משווים להרצה הקודמת
            if previous_boxes is not None:
                if self.are_results_similar(previous_boxes, current_boxes, iou_threshold=0.5):
                    print("Model result is stable")

                    image_with_boxes = results[0].plot()

                    if show_table_image:
                        cv2.imshow("model result", image_with_boxes)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    if show_cordinates:
                        number_of_arrows = 0

                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence_score = box.conf[0].item()

                            number_of_arrows += 1

                            print("Box coordinates:")
                            print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            print(f"confidence={confidence_score}")

                        print("Number of arrows detected:", number_of_arrows)

                    return current_boxes

            previous_boxes = current_boxes

        print("Model result is not stable")
        return None