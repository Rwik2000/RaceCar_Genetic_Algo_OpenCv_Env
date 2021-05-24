# from comet_ml import Experiment
# from ENV_sample_run import Cars
from time import sleep
import cv2
import numpy as np
from numpy.lib.financial import ipmt
from numpy.lib.function_base import select
from GEN_agentFile import Agent
# from GEN_environment import Car, Environment
from ENV_environment import env
from GEN_config import Args, configure
# from API_KEYS import api_key, project_name
import torch
import os
import time
from csv import writer
from tqdm import tqdm
configs, use_cuda,  device = configure()

## SET LOGGING
# experiment = Experiment(project_name = project_name,  api_key = api_key)
# experiment.log_parameters(configs.getParamsDict())
    

def getTrainTest( isTest = False, experiment = None,):
    if isTest:
        return experiment.test()
    return experiment.train()


def mutateWeightsAndBiases(agents, configs:Args):
    nextAgents = []

    if configs.test == True:
        for i in range(configs.num_vehicles):
            pair = agents[i]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            nextAgents.append(agentNet)
    else:
        for i in range(configs.num_vehicles):
            pair = agents[i % len(agents)]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            for param in agentNet.net.parameters():
                param.data += configs.mutationPower * torch.randn_like(param)
            nextAgents.append(agentNet)

    return nextAgents

def saveWeightsAndBiases(agentDicts, generation, configs:Args):
    loc = configs.saveLocation +'generation_'+str(generation) +  '/' 
    os.makedirs(loc, exist_ok = True)
    for i in range(len(agentDicts)):
        torch.save(agentDicts[i], loc + str(i) +  '-AGENT.pkl')



if __name__ == "__main__":
    print('-------------BEGINNING EXPERIMENT--------------')
    
    currImageNumber = 1
    if configs.addToVAEdata:
        while os.path.exists("VAE_dataset/images/VAE_img_%s.jpg" % currImageNumber):
            currImageNumber += 1
        print(currImageNumber)
    currentAgents = []
    if configs.checkpoint != 0:
        for spawnIndex in range(configs.nSurvivors):
            statedict = torch.load(configs.saveLocation +'generation_'+str(configs.checkpoint) +  '/'  + str(spawnIndex) +  '-AGENT.pkl')
            currentAgents.append(statedict)
        
        currentAgents = mutateWeightsAndBiases(currentAgents, configs)
        print('-> Loaded agents from checkpoint', configs.checkpoint)
    else:
        for spawnIndex in range(configs.num_vehicles):
            agent = Agent(configs, device)
            currentAgents.append(agent)
    ENV = env(speed_X=70, dist_to_px=30)
    ENV.num_vehicles = configs.num_vehicles
    ENV._obstaclesON = 0
    # print()
    # print(ENV.num_vehicles)
    # print(len(currentAgents))
    # print()
    # env = Environment(configs)

    # with getTrainTest(configs.test, experiment):
    action = np.zeros((configs.num_vehicles, 2))
    state = np.ones((configs.num_vehicles, configs.num_vis_pts))*configs.max_vis
    dead = np.zeros((configs.num_vehicles, ))
    rewards = np.zeros((configs.num_vehicles, ))
    success = 0
    failure = 0
    
    for generationIndex in range(configs.checkpoint, configs.checkpoint + 10000):
        VAEdataloc = []
        winner = -1
        trk01 = ENV.gen_track()
        Cars = ENV.gen_vehicles()
        _, trk01_scr, spawn_loc = trk01.gen_track_top()
        print("number of turns : ", trk01._turns)
        cflag = 0

        action = np.zeros((configs.num_vehicles, 2))
        # print(configs.max_vis)
        state = np.ones((configs.num_vehicles, configs.num_vis_pts))*configs.max_vis
        dead = np.zeros((configs.num_vehicles, ))
        rewards = np.zeros((configs.num_vehicles, ))
        nextAgents = []

        startTime = time.time()
        thresh_time = 180
        # for timestep in range(configs.deathThreshold):
        init_vel = np.random.uniform(0, 40)
        # init_yaw = np.random.uniform(-70,70)
        count = 0
        carLoc_list = [[]]*len(currentAgents)
        for timeindex in tqdm(range(60), total=60):
            input_scr = trk01_scr.copy()
            for agentIndex in range(len(currentAgents)):
                if count==0:
                    # print(ENV.vehicles[agentIndex].yaw)
                    ENV.vehicles[agentIndex].yaw = 0
                if dead[agentIndex] == 0:
                    action[agentIndex] = currentAgents[agentIndex].chooseAction(state[agentIndex])
                    # print(action[agentIndex])
                    action[agentIndex][0] = action[agentIndex][0].clip(0.0, 1.0)
                    # print(action[agentIndex])
                    action[agentIndex][1] = action[agentIndex][1].clip(0,1.0)
                    ENV.vehicles[agentIndex].track = input_scr
                    if cflag == 0:
                        ENV.vehicles[agentIndex].loc = spawn_loc.copy()
                        ENV.vehicles[agentIndex].vel = init_vel
                        # ENV.vehicles[agentIndex].yaw = -np.deg2rad(init_yaw)

                    # throttle = max(0.5, action[agentIndex][0])
                    throttle = 0.8
                    action[agentIndex][0] = throttle
                    steer = action[agentIndex][1]
                    # print(throttle, steer)
                    vis_pts, carLoc ,dead[agentIndex], reward = ENV.vehicles[agentIndex].move(throttle,steer)
                    # print(reward)
                    rewards[agentIndex] += reward

                    if configs.addToVAEdata:
                        carLoc_list[agentIndex].append(carLoc[:])
                        # VAEdataloc.append(carLoc[:])

                    state[agentIndex] = vis_pts
                    if rewards[agentIndex] < -30 and dead[agentIndex] ==0:
                        # print("DEAD - Lack of rewards")
                        ENV.vehicles[agentIndex].reset()
                        ENV.vehicles[agentIndex].done = -1
                        dead[agentIndex] = -1
                        rewards[agentIndex] -= 20
                        
            count+=1
                    
            # print(action)
            if 0 not in dead:
                break
            # print("hey")
            if generationIndex%1 == 0 and timeindex % 10 == 0:
                ENV.render()
            cflag+=1
        avgScore = np.mean(rewards)
        winner = np.where(rewards == np.amax(rewards))[0]
        winner = np.random.choice(winner, size = 1, replace=False)[0]
        # print()
        if rewards.any() > 0:
            success+=1
            if configs.addToVAEdata:
                img_name = "VAE_img_%s.jpg" % currImageNumber
                img_loc = "VAE_dataset/images/"+img_name
               
                cv2.imwrite(img_loc, trk01_scr)
                with open(configs.VAE_csvloc, 'a', newline='') as csv_file:
                    print("WINNER: ", winner, "REWARD : ", rewards[winner])
                    csv_row = [img_name, np.round(carLoc_list[winner],decimals=3).tolist(), spawn_loc]
                    writer_object = writer(csv_file) 
                    writer_object.writerow(csv_row) 
                    csv_file.close()
            currImageNumber+=1
            
        else:
            failure+=1


        if configs.test == True:
            print('Generation', generationIndex - configs.checkpoint,'Complete in ',time.time() - startTime , 'seconds')
        else:
            print('Generation', generationIndex,'Complete in ',time.time() - startTime , 'seconds')
        print('FITNESS = ', avgScore)
        print('---------------')

        if not configs.test:
            # print(currentAgents)
            # print(rewards)
            temp = [[currentAgents[agentIndex], rewards[agentIndex]] for agentIndex in range(len(currentAgents))]
            currentAgents = sorted(temp, key = lambda ag: ag[1], reverse = True)
            nextAgents = currentAgents[:configs.nSurvivors]
            # print(nextAgents)
            currentAgents = mutateWeightsAndBiases(nextAgents, configs)
            if (generationIndex + 1) % 5 == 0:
                saveWeightsAndBiases(nextAgents, generationIndex, configs)
        # else:
        #     env.saveImage()
    print("SUCCESS : ", success, " FAILURE:", failure)
        
            

