# Bicycle model implementation
from os import SEEK_CUR
import numpy as np
import matplotlib.pyplot as plt
class Vehicle():
    def __init__(self, mass = 1000, mu = 0.5, vel = 1,acc = 0,max_acc = 10 ,max_vel = 40,
                g=9.8, loc = [0.0,0.0], dist_to_px = 20, track=None, speed_X=1):
        # scaling 
        self.dist_to_px = dist_to_px
        self.dt = 1e-3
        self.g = 9.8
        self.done = 0
        self.reward = 0
        self.time = 0
        # speed_X is to increase the speed of simulation at the  cost of precision.
        self.mass = mass
        self.mu = mu
        self.lr = 1
        self.lf = 1
        self.max_vel = max_vel*speed_X
        self.max_acc = max_acc*(speed_X**2)
        # max angle 70
        self.max_steer = np.deg2rad(70)

        self.loc = np.array(loc)
        self.loc =  self.loc.astype(float)
        # print(self.loc)
        # slip angle -> angle b/w acceleration and vel
        self.slip_angle = 0
        # angle b/w vel and heading line
        self.course_angle = 0
        # heading angle
        self.yaw = 0
        self.yaw_rate = 0

        self.vel = vel*speed_X 
        self.acc = acc*(speed_X**2)

        # scaling factor
        self.throttle_fac = 10
        self.speed_X =speed_X
        # number of vision points. 
        self.num_vis_pts = 8
        self.anglesToSee = []
        for i in range(self.num_vis_pts):
            self.anglesToSee.append(np.deg2rad(i*180/(self.num_vis_pts-1)))

        self.max_vis = 10*self.dist_to_px

        self._vis_pts = [] #units are in pixels and will comprise of coordinates instead of distances.
        self.vis_pts = [] #units are in meters and will contain distances of the track points from the car. The order being 0 to 180 degrees
        
        # importing track from track.py
        self.track = track

    def reset(self):
        self.slip_angle = 0
        self.course_angle = 0
        self.yaw = 0
        self.vel = 0
        self.acc = 0
        self.yaw_rate = 0
        self.reward = 0


    def get_vision_points(self):
        h,w,_ =self.track.shape
        # angles = []
        # for i in range(self.num_vis_pts):
        #     angles.append(np.deg2rad(i*180/(self.num_vis_pts-1)))
        self._vis_pts = []
        self.vis_pts = []
        self.pt_1 = np.array(self.loc)
        self.pt_2 = self.pt_1
        for angle in self.anglesToSee:
            count = 1
            done = 1
            dist = self.max_vis
            while 1:
                dist = np.linalg.norm((self.pt_1-self.pt_2))
                if dist >= self.max_vis:
                    break
                self.pt_2 = np.array([max(min(self.pt_1[0] + np.cos(angle - self.yaw)*count,w-1), 0), 
                                      max(self.pt_1[1] - np.sin(angle - self.yaw)*count, 0)])
                self.pt_2 = self.pt_2.astype(int)
                
                if (self.track[self.pt_2[1]][self.pt_2[0]] == np.array([0,0,0])).all() or self.pt_2[1]==0:
                    break
                count+=1
            self.vis_pts.append(round(dist/self.dist_to_px, 2))
            self._vis_pts.append(self.pt_2)
            
            self.pt_2 = self.pt_1


    def get_angle(self, steer):
        # output angle in radians
        return (steer*np.pi/2)
    
    def move(self, throttle, _steer):
        steer = (_steer - 0.5)*2
        # print(steer)
        self.time += self.dt*self.speed_X
        vel_x = self.vel*np.sin(self.yaw + self.course_angle)
        vel_y = self.vel*np.cos(self.yaw + self.course_angle)

        # acceleration update
        self.acc = throttle*self.throttle_fac*(self.speed_X**2)
        self.acc = np.clip(self.acc, -self.max_acc,self.max_acc)
        
        # velocity update
        self.vel += (self.acc - 1*(self.g*self.mu)*(self.speed_X**2))*(self.dt)
        self.vel = np.clip(self.vel, 0,self.max_vel)

        # Angles update
        self.yaw_rate = self.vel/self.lr*np.sin(self.course_angle)
        self.course_angle = np.arctan(self.lr/(self.lf+self.lr)*np.tan(self.get_angle(steer)))
        self.yaw += self.yaw_rate*self.dt
        if abs(self.yaw)>np.pi*2:
            self.yaw -= np.pi*2
        self.prev_loc = self.loc
        self.prev_yaw = self.yaw
        self.loc[1] -= vel_y*self.dt*self.dist_to_px 
        self.loc[0] += vel_x*self.dt*self.dist_to_px

        # Drawing track_points
        self.get_vision_points()
        self.done =  0
        try:
            if(self.loc[1] <= 50 ):
                self.done = 1
            elif (self.track[int(self.loc[1])][int(self.loc[0])] == np.array([0,0,0])).all():
                self.reset()
                self.done = -1      
                # print("yes")          
        except IndexError:
            self.done = -1

        self.get_reward()
        # print(self.reward)
        return self.vis_pts,self.loc, self.done, self.reward

    def get_reward(self):
        self.reward = 0
        if self.done == 1:
            self.reward += 20
        else:
            if self.done == -1:
                self.reward -=10
            # print("yo")
            self.reward -= 0.1
            self.reward -= (self.loc[1] - self.prev_loc[1])*30
            self.reward -= abs(self.yaw - self.prev_yaw)*10

        # print(self.reward)
        return round(self.reward,2)
        

    






    
    
    