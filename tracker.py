import cv2
import sys

class tracker ():
    def __init__(self, tracker_type, bounding_box, frame):
        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        self.is_init = False
        self._bounding_box = bounding_box
        p1 = (int(self._bounding_box[0]), int(self._bounding_box[1]))
        p2 = (int(self._bounding_box[0] + self._bounding_box[2]), int(self._bounding_box[1] + self._bounding_box[3]))
        self.rectangle = (p1, p2)

    def update(self, frame, bounding_box=None):
        """
        :param bounding_box: bounding box of pervious frame if changed
        :param frame: current image
        :return: is_success and new bounding_box
        """
        self._bounding_box = bounding_box if bounding_box else self._bounding_box
        if not self.is_init:
            self.tracker.init(frame, self._bounding_box)
            self.is_init = True
        if not self.is_init:
            return self._is_success, self._bounding_box
        self._is_success, self._bounding_box = self.tracker.update(frame.copy())
        self.save_points()
        return self._is_success, self._bounding_box

    def get_bounding_box(self):
        return self._bounding_box

    def save_points(self, ):
        if self._is_success:
            # Tracking success
            p1 = (int(self._bounding_box[0]), int(self._bounding_box[1]))
            p2 = (int(self._bounding_box[0] + self._bounding_box[2]), int(self._bounding_box[1] + self._bounding_box[3]))
            self.rectangle = (p1,p2)
        else:
            # Tracking failure
            self.rectangle = None
            # cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def visualize(self, frame):
        if self.rectangle:
            cv2.rectangle(frame, self.rectangle[0], self.rectangle[1], (255, 0, 0), 2, 1)
