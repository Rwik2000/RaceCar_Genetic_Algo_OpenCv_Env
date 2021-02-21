# from comet_ml import Experiment
# from ENV_sample_run import Cars
from time import sleep
import numpy as np
from GEN_agentFile import Agent
# from GEN_environment import Car, Environment
from ENV_environment import env
from GEN_config import Args, configure
# from API_KEYS import api_key, project_name
import torch
import os
import time
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
    ENV = env(speed_X=70)
    ENV.num_vehicles = configs.num_vehicles
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

        
    for generationIndex in range(0, 100):
        trk01 = ENV.gen_track()
        Cars = ENV.gen_vehicles()
        trk01_scr, spawn_loc = trk01.gen_track()
        cflag = 0

        action = np.zeros((configs.num_vehicles, 2))
        # print(configs.max_vis)
        state = np.ones((configs.num_vehicles, configs.num_vis_pts))*configs.max_vis
        dead = np.zeros((configs.num_vehicles, ))
        rewards = np.zeros((configs.num_vehicles, ))
        nextAgents = []

        startTime = time.time()
        thresh_time = 90
        # for timestep in range(configs.deathThreshold):
        init_vel = np.random.uniform(0, 40)
        init_yaw = np.random.uniform(-70,70)
        while time.time() - startTime <= thresh_time:
            input_scr = trk01_scr.copy()
            for agentIndex in range(len(currentAgents)):
                if dead[agentIndex] == 0:
                    action[agentIndex] = currentAgents[agentIndex].chooseAction(state[agentIndex])
                    action[agentIndex][0] = action[agentIndex][0].clip(0.0, 1.0)
                    # print(action[agentIndex])
                    action[agentIndex][1] = action[agentIndex][1].clip(0,1.0)
                    ENV.vehicles[agentIndex].track = input_scr
                    if cflag == 0:
                        ENV.vehicles[agentIndex].loc = spawn_loc.copy()
                        ENV.vehicles[agentIndex].vel = init_vel
                        ENV.vehicles[agentIndex].yaw = -np.deg2rad(init_yaw)

                    
                    throttle = action[agentIndex][0]
                    steer = action[agentIndex][1]
                    vis_pts,_ ,dead[agentIndex], reward = ENV.vehicles[agentIndex].move(1,steer)
                    rewards[agentIndex] += reward

                    state[agentIndex] = vis_pts
                    if rewards[agentIndex] < -5:
                        print("Dead due to lack of rewards")
                        ENV.vehicles[agentIndex].done = -1
                        dead[agentIndex] == -1
                        rewards[agentIndex] -= 10
            # print(rewards)
            if 0 not in dead:
                break
            #     break
            # break

                    # print(throttle, steer)
            # print(action)
            if 0.0 not in dead:
                break
            if generationIndex%1 == 0:
                ENV.render()
            cflag+=1
        avgScore = np.mean(rewards)
        # experiment.log_metric("fitness", np.mean(avgScore) , step= generationIndex)

        print('Generation', generationIndex,'Complete in ',time.time() - startTime , 'seconds')
        print('FITNESS = ', avgScore)
        print('---------------')
        
        if not configs.test:
            temp = [[currentAgents[agentIndex], rewards[agentIndex]] for agentIndex in range(len(currentAgents))]
            currentAgents = sorted(temp, key = lambda ag: ag[1], reverse = True)
            nextAgents = currentAgents[:configs.nSurvivors]

            currentAgents = mutateWeightsAndBiases(nextAgents, configs)
            if (generationIndex + 1) % 5 == 0:
                saveWeightsAndBiases(nextAgents, generationIndex, configs)
        else:
            env.saveImage()

        
            

