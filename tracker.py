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
        self._is_success = self.tracker.init(frame, bounding_box)
        self._prev_frame = frame
        self._bounding_box = bounding_box
    def update(self, frame, bounding_box=None):
        """
        :param bounding_box: bounding box of pervious frame if changed
        :param frame: current image
        :return: is_success and new bounding_box
        """
        if bounding_box:
            self.tracker.init(self.frame, bounding_box)
            self._bounding_box = bounding_box

        self._is_success, self._bounding_box = tracker.update(frame)
        return self._is_success, self._bounding_box
    def visualize(self, frame):
        if self._is_success:
            # Tracking success
            p1 = (int(self._bounding_box[0]), int(self._bounding_box[1]))
            p2 = (int(self._bounding_box[0] + self._bounding_box[2]), int(self._bounding_box[1] + self._bounding_box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
