import numpy as np
import cv2

class Visualize:
    def __init__(self):
        self.objects = {}
        self.timer = 0
        self.MAX_time = 20
        self.colour = [(255, 0, 0), (0, 0, 255), (60, 179, 113), (238, 130, 238), (106, 90, 205), (255, 165, 0),
                       (32, 23, 133), (255, 208, 34), (255, 99, 71), (87, 99, 71), (87, 255, 71), (87, 159, 177),
                       (0, 70, 58), (0, 215, 255), (184, 0, 255), (184, 0, 64), (184, 90, 64), (85, 90, 129)]
        self.font = cv2.FONT_HERSHEY_PLAIN

    def rectangle(self, people, pt1, pt2, idx):
        cv2.rectangle(people, pt1, pt2, self.colour[idx % len(self.colour)], 2)

    def put_text(self, image, txt, pos):
        cv2.putText(image, txt, pos, self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_line(self, id, canvas):
        if (self.objects[id][0] == -1 or self.objects[id][1] == -1):
            return canvas
        cv2.line(canvas, (self.objects[id][0], self.objects[id][1]), (self.objects[id][2], self.objects[id][3])
                 , self.objects[id][4], 3)
        return canvas

    def add_points(self, trackers):
        self.timer += 1
        for d in trackers:
            d = d.astype(np.int32)
            if d[4] in self.objects and self.timer - self.objects[d[4]][5] < self.MAX_time:
                prev = self.objects[d[4]]
                self.objects[d[4]] = \
                    (prev[2], prev[3], (d[0]+d[2])/2, (d[1]+d[3])/2, self.colour[d[4] % len(self.colour)], self.timer)
            else:
                self.objects[d[4]] = (-1, -1, (d[0]+d[2])/2, (d[1]+d[3])/2, self.colour[d[4] % len(self.colour)], self.timer)