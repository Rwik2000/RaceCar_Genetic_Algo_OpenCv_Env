import numpy as np
import cv2
import random
from scipy.linalg import pascal
class Track():
    def __init__(self, dist_to_px,scr_width = 800, scr_height = 600, trk_width = 10):
        # Track params
        self.scr_width = scr_width
        self.scr_height = scr_height
        self.trk_width = trk_width #in meters
        self.dist_to_px = dist_to_px
        self.trk_width_px = trk_width*self.dist_to_px

        self.turns = [0,1,2] #random turns

        # Car params
        self.car_front_clearance = 5*self.dist_to_px #5 stands for meter, self.dist_to_px is conversion to pixel space
    
    # points based on which bezier plot will be made
    def _gen_bnd_pts(self):
        left_bot_pt = np.array([np.random.randint(0, self.scr_width - self.trk_width_px), self.scr_height-self.car_front_clearance])
        right_bot_pt = np.array([left_bot_pt[0]+self.trk_width_px, self.scr_height- self.car_front_clearance])

        left_top_pt = np.array([np.random.randint(0, self.scr_width - self.trk_width_px), 0])
        right_top_pt = np.array([left_top_pt[0]+self.trk_width_px, 0])

        turns = random.choice(self.turns)
        left_pts = [left_bot_pt]
        right_pts = [right_bot_pt]
        for i in range(turns):
            left_mid_pt = np.array([np.random.randint(0, self.scr_width - self.trk_width_px), (left_top_pt[1]*(i+1)+left_bot_pt[1])//turns])
            left_pts.append(left_mid_pt)
            right_mid_pt = np.array([left_mid_pt[0]+self.trk_width_px, (left_top_pt[1]*(i+1)+left_bot_pt[1])//turns])
            right_pts.append(right_mid_pt)
        left_pts.append(left_top_pt)
        right_pts.append(right_top_pt)
        return left_pts, right_pts

    # drawing the bezier trajectory
    def _find_bez_traj(self, coordinates, points):
        n=len(coordinates)

        pascal_coord=pascal(n,kind='lower')[-1]
        t=np.linspace(0,1,points)

        bezier_x=np.zeros(points)
        bezier_y=np.zeros(points)

        for i in range(n):
            k=(t**(n-1-i))
            l=(1-t)**i
            bezier_x+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][0]
            bezier_y+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][1]
        bezier_xd=[]
        bezier_yd=[]
        for i in range(len(bezier_x)):
            bezier_xd.append(int(bezier_x[i]))
            bezier_yd.append(int(bezier_y[i]))

        bezier_coordinates = np.transpose([bezier_xd, bezier_yd])
        return bezier_coordinates

    # Generating the track
    def gen_track(self):
        left_boundary_pts, right_boundary_pts = self._gen_bnd_pts()
        left_final_points = self._find_bez_traj(left_boundary_pts, 20)
        right_final_points = self._find_bez_traj(right_boundary_pts, 20)

        self.screen = np.zeros((self.scr_height, self.scr_width))
        self.screen = cv2.line(self.screen, tuple(left_final_points[0]), tuple(right_final_points[0]), (255,255,255))
        for i in range(len(left_final_points)-1):
            self.screen = cv2.line(self.screen, tuple(left_final_points[i]), tuple(left_final_points[i+1]), (255,255,255))
            self.screen = cv2.line(self.screen, tuple(right_final_points[i]), tuple(right_final_points[i+1]), (255,255,255))

        temp = ((left_boundary_pts[0][0]+right_boundary_pts[0][0])//2, left_boundary_pts[0][1]-2)
        self.screen = np.float32(self.screen)
        self.screen = cv2.cvtColor(self.screen, cv2.COLOR_GRAY2BGR)
        val = 5
        cv2.floodFill(self.screen, None, seedPoint=temp, newVal=(255, 255, 255), loDiff=(val, val, val, val), upDiff=(val, val, val, val))

        # Random spawn location of the car
        spawn_loc = [np.random.randint(left_boundary_pts[0][0],right_boundary_pts[0][0]), left_boundary_pts[0][1]-5]
        return self.screen, spawn_loc



