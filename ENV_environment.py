from os import stat
import cv2
import math
from ENV_vehicle import Vehicle
from ENV_track import Track

class env():
    def __init__(self, speed_X=1, dist_to_px = 20, render = 0):
        self.speed_X = speed_X
        self.dist_to_px = dist_to_px
        self.print_info = 0
        self.vehicles = []
        self._obstaclesON = 0
        self.num_vehicles = 1

    def gen_vehicles(self, vel = 0, acc = 0, yaw = 0):
        for i in range(self.num_vehicles):
            v = Vehicle(speed_X=self.speed_X, dist_to_px=self.dist_to_px)
            v.vel = vel
            v.acc = acc
            v.yaw = yaw
            self.vehicles.append(v)
        return self.vehicles
    
    def gen_track(self):
        self.trk = Track(self.dist_to_px)
        self.trk.dist_to_px = self.dist_to_px
        self.trk.obstaclesON = self._obstaclesON
        return self.trk

    def drawCar(self, x0, y0, width, height, angle, img, status):

        # _angle = angle * math.pi / 180.0
        _angle = angle
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        pt0 = (int(x0 - a * height - b * width),
            int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width),
            int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        yellow = (0,255,255)
        red = (0,0,255)
        blue = (255,0,0)
        green = (0,255,0)
        cv2.line(img, pt1, pt2, (0, 0, 255), 2) #Car front
        carColor = yellow
        if status ==-1:
            carColor = blue
        if status ==1:
            carColor = green
        cv2.line(img, pt0, pt1, carColor, 2)
        cv2.line(img, pt2, pt3, carColor, 2)
        cv2.line(img, pt3, pt0, carColor, 2)
        return img

    def render(self):
        # print(self.trk)
        self.track = self.trk.screen.copy()
        # for vehicle in self.vehicles:
        #     print(vehicle.loc)
        for vehicle in self.vehicles:
            # print(vehicle.loc)
            for point in vehicle._vis_pts:
                cv2.circle(self.track, tuple(point), 2, (0,255,0),3)
            cv2.circle(self.track,(int(vehicle.loc[0]),int(vehicle.loc[1])), 4, (0,255,255),3)    
            self.track = self.drawCar(int(vehicle.loc[0]), int(vehicle.loc[1]), 20, 40, vehicle.yaw, self.track, vehicle.done)
        cv2.imshow("xyz",self.track)
        cv2.waitKey(1)
    
    
    





