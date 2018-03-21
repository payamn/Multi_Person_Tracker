"""
@author: Payam Nikdel
"""
import numpy as np
import cv2
import math

class Visualize:
    def __init__(self):
        self.objects = {}
        self.timer = 0
        self.MAX_time = 20
        self.RADIUS = 10

        self.colour = [(255, 0, 0), (0, 0, 255), (60, 179, 113), (238, 130, 238), (106, 90, 205), (255, 165, 0),
                       (32, 23, 133), (255, 208, 34), (255, 99, 71), (87, 99, 71), (87, 255, 71), (87, 159, 177),
                       (0, 70, 58), (0, 215, 255), (184, 0, 255), (184, 0, 64), (184, 90, 64), (85, 90, 129)]
        self.font = cv2.FONT_HERSHEY_PLAIN

    def rectangle(self, img, pt1, pt2, idx):
        cv2.rectangle(img, pt1, pt2, self.colour[idx % len(self.colour)], 2)

    def put_text(self, image, txt, pos):
        cv2.putText(image, txt, pos, self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    def heat_map_circle(self, image, point_center):
        start_point = (max(point_center[0] - self.RADIUS, 0), max(point_center[1] - self.RADIUS, 0))
        end_point = (
        min(point_center[0] + self.RADIUS, image.shape[1] - 1), min(point_center[1] + self.RADIUS, image.shape[0] - 1))
        for x in range(start_point[0], end_point[0] + 1):
            for y in range(start_point[1], end_point[1] + 1):
                distance = self.euclideanDistance(point_center, (x, y), self.RADIUS)
                if distance != -1:
                    if (image[y, x, 0] + ((self.RADIUS - distance) / self.RADIUS) * 200 < 255):
                        image[y, x, 0] += ((self.RADIUS - distance) / self.RADIUS) * 200
                    else:
                        image[y, x, 0] = 255
        return image

    def heat_map_rectangle(self, id, canvas):
        temp = np.zeros(canvas.shape, np.uint8)
        self.draw_line(id, temp, color=(10,10,10), thicknfess=20, start_distance=20)
        canvas = temp + canvas

        return canvas

    def euclideanDistance(self, center, point, radius):  # returns a float
        distance = math.sqrt((float(center[0]) - point[0]) ** 2 + (float(center[1]) - point[1]) ** 2)
        if distance > radius:
            return -1
        else:
            return distance

    def draw_line(self, id, canvas, color = None, thickness=3, start_distance = 0):
        if (self.objects[id][0] == -1 or self.objects[id][1] == -1):
            return canvas
        color = self.objects[id][4] if color is None else color
        distance = math.sqrt((self.objects[id][3]- self.objects[id][1]) ** 2 + (self.objects[id][2] - self.objects[id][0]) ** 2)
        if (distance == 0):
            return canvas
        t = start_distance/distance
        # m = (self.objects[id][3]- self.objects[id][1])/ (self.objects[id][2] - self.objects[id][0])
        # x = self.objects[id][0] + start_distance / math.sqrt(1+m*m)
        # y = m*(x-self.objects[id][0])+self.objects[id][1]
        x = (1 - t) * self.objects[id][0] + t * self.objects[id][2]
        y = (1 - t) * self.objects[id][1] + t * self.objects[id][3]
        cv2.line(canvas, (int(x), int(y)), (self.objects[id][2], self.objects[id][3])
                 , color, thickness)
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