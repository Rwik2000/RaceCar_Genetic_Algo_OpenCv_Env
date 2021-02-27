import time
import numpy as np
import cv2
import random
from ENV_environment import env

ENV = env(speed_X=70)
trk01 = ENV.gen_track()
ENV.num_vehicles = 2
initial_vel = 5
Cars = ENV.gen_vehicles()
ENV.print_info = 1

trk01_scr,_ = trk01.gen_track()
cv2.imshow("track", trk01_scr)
cv2.waitKey(0)
# for i in range(20):
#     trk01_scr, spawn_loc = trk01.gen_track()
#     outcome="DEAD"
#     for i in range(ENV.num_vehicles):
#         ENV.vehicles[i].vel = initial_vel*ENV.speed_X
#     start = time.time()
#     locus = trk01_scr.copy()
#     tot_reward = 0
#     cflag = 0
#     while 1:
#         input_scr = trk01_scr.copy()
#         dones = np.zeros(ENV.num_vehicles)
#         for i in range(ENV.num_vehicles):
#             # temp_car = ENV.vehicles[i]
#             if dones[i] ==0:
#                 ENV.vehicles[i].track = input_scr
#                 if cflag == 0:
#                     ENV.vehicles[i].loc = spawn_loc.copy()

#                 '''Add your Code here'''
#                 throttle = 0.3
#                 steer = 0.5
#                 steer_val = random.choice([-1,1])*steer

#                 vis_pts,_ ,dones[i], reward = ENV.vehicles[i].move(1,steer_val)
#                 # print(vis_pts)
#                 # print(ENV.vehicles[i].vis_pts)

#         '''If render not required, comment the next line'''
#         ENV.render()
#         # print()
#         cflag+=1
#         # print(dones)
#         # print("----------------------")
#         if 0 not in dones:
#             if 1 in dones:
#                 print("SUCCESS")
#             else:
#                 print("FAILURE")
#             print(time.time() - start)
#             break

