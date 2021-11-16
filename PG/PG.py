### Model architecture...



from typing_extensions import Required
import gym
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
from torch.nn.functional import one_hot, log_softmax, softmax, normalize



class PNet(nn.Module):

  '''
  A simple neural net that acts as a func approximator to Policy : outputs logits for actions ,,
  '''
  def __init__(self,seed, state_size, action_size):
    super(PNet, self).__init__()
    self.state_size = state_size
    self.action_size = action_size

    self.seed = torch.manual_seed(seed)
    self.fc1_size = 64
    self.fc2_size = 64

    self.fc1 = nn.Linear(self.state_size,self.fc1_size )
    self.fc2 = nn.Linear(self.fc1_size,self.fc2_size )
    self.out = nn.Linear(self.fc2_size,self.action_size )

  def forward(self,s):
    s = normalize(s, dim=0)
    s = F.relu(self.fc1(s))
    s = F.relu(self.fc2(s))
    logits = self.out(s)

    action = self.get_action(logits)
    action_loP = F.log_softmax(logits)[0][action.item()] ## log of the prob of chosen action..
    return action, action_loP

  def get_action(self, net_out):
    action = Categorical(logits = net_out).sample()  ### returns the index : action from logits
    return action


class PGAgent(PNet):
  def __init__(self, env, seed,  device,
               BATCH_SIZE  = 64,
               GAMMA = 0.99,
               LR = 5e-3,
               SCORE_WINDOW_LEN = 100,### no. of episodes for plotting learning curve
               TARGET_SCORE = 180,):
    '''
    env : used environment, 
    save_dir_name : name of the dir where the results are to be saved
    Batch size: no . of episodes in each epoch
    gamma: discount factor
    lr: learning rate of the PNet
    Target score : score to save the model and stop training
    '''

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    super(PGAgent, self).__init__(seed, state_size, action_size)


    self.BATCH_SIZE = BATCH_SIZE
    self.LR = LR
    self.GAMMA = GAMMA
    self.TARGET_SCORE = TARGET_SCORE
    self.SCORE_WINDOW_LEN = SCORE_WINDOW_LEN
    
    self.seed = random.seed(seed)
    print(self.seed)
    self.device = device
    self.env = env

    self.PNet = PNet(seed, state_size, action_size).to(self.device)
    self.optimizer = optim.Adam(self.PNet.parameters(), lr = self.LR)

    ### adding rewards for learning curve visualizing ..
    self.total_rewards = deque([], maxlen = SCORE_WINDOW_LEN)

  def train(self,save_dir_name,
            epochs = 2000, apply_reward2go = True, apply_AdvNorm = True):




    self.epochs = epochs
    self.apply_reward2go = apply_reward2go
    self.apply_AdvNorm = apply_AdvNorm

    #### Writing results in a txt file for furhter reference...
    f= open( save_dir_name + "/train_log.txt","a")
    f.write("\nHYPERPARAMETERS\n")
    comment = str(input('Enter a comment..'))
    f.write('Comment: {}\n'.format(comment) )
    f.write("\n-------------------\n")
    f.write('BATCH_SIZE: {}\n'.format(self.BATCH_SIZE))
    f.write('GAMMA: {}\n'.format(self.GAMMA))
    f.write('LR: {}\n'.format(self.LR))
    f.write('SEED: {}\n'.format(self.seed))
    f.write('SCORE_WINDOW_LEN: {}\n'.format(self.SCORE_WINDOW_LEN))
    f.write('device: {}\n'.format(self.device))
    f.write('TARGET_SCORE: {}\n'.format(self.TARGET_SCORE))
    f.write('MAX_EPOCHS:{}\n'.format(self.epochs))
    f.write('APPLY_REWARD2GO:{}\n'.format(self.apply_reward2go))
    f.write('APPLY_ADV_NORM:{}\n'.format(self.apply_AdvNorm))
    f.write("\n-------------------\n")
    f.write("TRAINING")
    f.write("\n-------------------\n")
    ##############################################################
    '''
    for each epoch --> simulate BATCH_SIZE no. of episodes --> find loss by mean(loP*advantange_estimate)
    '''
    self.epoch_list_weighted_rewards  = torch.empty(size = (0,), dtype = torch.float, device=self.device) ## list of loP*A for each episode --> for averaging later
    self.epoch_reward_list = [] ## list containing reward for each epoch
    for self.epoch_no in range(self.epochs):
      for self.epi_no in range(self.BATCH_SIZE):

        (epi_sum_weighted_loP_rew2go, epi_sum_of_rewards) = self.simulate_trajectory() ## Simulate an episode 
        self.total_rewards.append(epi_sum_of_rewards)
        self.epoch_list_weighted_rewards = torch.cat( (self.epoch_list_weighted_rewards,epi_sum_weighted_loP_rew2go ), dim=0)
        
      self.epoch_loss = (-1* torch.mean(self.epoch_list_weighted_rewards)).to(self.device)
      self.optimizer.zero_grad()
      self.epoch_loss.backward()
      self.optimizer.step()
    

      epoch_avg_reward = np.mean(self.total_rewards)
      print('\rEpoch {}\tAverage Reward {:.3f}'.format(self.epoch_no, epoch_avg_reward )) 
      f= open( save_dir_name + "/train_log.txt","a")
      f.write('\rEpoch {}\tAverage Reward {:.3f}'.format(self.epoch_no, epoch_avg_reward ))
      f.close()

      self.epoch_reward_list.append(epoch_avg_reward)  
      self.epoch_list_weighted_rewards  = torch.empty(size = (0,), dtype = torch.float, device=self.device) ## Need to re-initialise weighted rewards for an epoch ,,

      if(epoch_avg_reward >= self.TARGET_SCORE):
        self.TARGET_SCORE = epoch_avg_reward
        torch.save(self.PNet.state_dict(),save_dir_name + '/best.pth')
        print("Model saved !!!")
        
        


    torch.save(self.PNet.state_dict(),save_dir_name + '/PNet_last.pth')
    print("Saved in ", save_dir_name + '/last.pth')
    ### Plotting and saving learning curve
    plt.plot(np.arange(len(self.epoch_reward_list)),self.epoch_reward_list)
    plt.xlabel('Epochs')
    plt.ylabel('mean-{} episode reward'.format(self.SCORE_WINDOW_LEN))
    plt.title('PG Agent Training ')
    plt.savefig(save_dir_name + '/train_plot.png')
    #plt.show()

    ### Saving epoch scores
    with open(save_dir_name + '/avg_score.npy', 'wb') as f:
      np.save(f, np.array(self.epoch_reward_list) )
    

    

  def simulate_trajectory(self):
    '''
    function:
    1. Simulates an end to end trajectory
    2. Saves rewards for each transition
    3. finds avg_rewards: avg_reward[i] := avg of all rewards until that time step --> for base line
    4. When done :
    ----> finds the discoutned rewards at each step 
    -----> subtracts average_rewards from discoutned rewards : baseline --> state specific average
    -----> Also finds the sum of rewards : total reward earned during the episode (without discounting)
    -----> return weighed sum of log prob and rew 2 go
    ---------------------------------------
    Returns : 
    1. epi_weighted_log_prob_rew2go--> epi_weighted_loP_rew2go
    2. epi_reward_sum
    '''

    ### reset the state to start with
    state = self.env.reset()

    ### Init storing elements 
    epi_loP  = torch.empty(size = (0,),  device=self.device) ## loP of choosen action
    epi_rewards = np.empty(shape = (0,), dtype = np.float) ## rewards obtained at t instant
    epi_rewards_running_mean = np.empty(shape = (0,), dtype = np.float) ## running avg of [0:t] rewards acts as baseline
    
    done = False
    while not done :

      action,action_loP =  self.PNet.forward(torch.from_numpy(state).float().unsqueeze(0).to(self.device) ) ## for a given transition --> find the action chosen given state and its loP

      ### Choosing action and finding reward
      state, rew, done, _= self.env.step(action.item())
      epi_rewards = np.concatenate((epi_rewards, np.array([rew])), axis = 0)
      epi_rewards_running_mean = np.concatenate((epi_rewards_running_mean, np.array([np.mean(epi_rewards)])), axis=0)
      epi_loP = torch.cat((epi_loP,action_loP.unsqueeze(0)),dim=0)
    
    ### if episode is done .. 

    # Apply reward to go strategy ....
    epi_discount_return = PGAgent.apply_discount_factor(epi_rewards, self.GAMMA, self.apply_reward2go)

    # Apply advantage normalisation

    if(self.apply_AdvNorm):
      ### base line subtracted : here base line is a state specific baseline
      epi_advantage_estimate = epi_discount_return  - epi_rewards_running_mean
    else:
      epi_advantage_estimate = epi_discount_return

    ## finding the rewards accumulated in the episode for learning curve visual
    epi_sum_of_rewards = np.sum(epi_rewards)
    
    epi_weighted_loP_rew2go = epi_loP * torch.tensor(epi_advantage_estimate).float().to(self.device)

    epi_sum_weighted_loP_rew2go = torch.sum(epi_weighted_loP_rew2go).unsqueeze(dim=0)
    return epi_sum_weighted_loP_rew2go, epi_sum_of_rewards

  def act(self,state):
    self.PNet.eval()
    with torch.no_grad():
      action,_ =  self.PNet.forward(torch.from_numpy(state).float().unsqueeze(0).to(self.device) ) ## for a given transition --> find the action chosen given state and its loP
    return action.item()



  @staticmethod
  def apply_discount_factor(epi_rewards : np.array, gamma : float, apply_reward2go:bool)-> np.array :
    '''
    given a np array of rewards, apply discounting to each instant 

    '''

    epi_rewards_discount = np.zeros(epi_rewards.shape)
    if(apply_reward2go):
      '''
      Applying reward 2 go
      implementing rewards 2 go
      epi_rewards[t] :  reward at t instant
      epi_rewards_discount[t] : discounted return from t:end  --> sum(gamma^t-t'*reward) 
      '''
      for t in range(epi_rewards.shape[0]):
        gammas = gamma*np.ones(shape = epi_rewards[t:].shape[0])
        gammas_discount = np.power(gammas, np.arange(epi_rewards[t:].shape[0]))

        epi_rewards_discount[t] =  np.sum(epi_rewards[t:] * gammas_discount)

    else:
      '''
      Applying temporal structure
      '''
      gammas = gamma*np.ones(shape=epi_rewards.shape[0])
      gammas_discount = np.power(gammas, np.arange(epi_rewards.shape[0]))
      epi_discounted_reward = np.sum(gammas_discount* epi_rewards)
      epi_rewards_discount = epi_discounted_reward*np.ones(shape=epi_rewards.shape[0])

    return epi_rewards_discount

  @staticmethod
  def find_model_dir_title(dir_path, continue_training):
    '''
    Given a dir where all exp are stored .. finds the last dir and creates a name for the new dir
    If we want to continur training it finds the latest directory and intialises the model with last tranined models
    '''
    print('I got ',continue_training)
    dir_len = sum(os.path.isdir(dir_path +'/'+ i) for i in  os.listdir(dir_path))

    if(continue_training==1):
      dir_len = dir_len -1
    return dir_path + '/exp'+str(dir_len)


###### running model ...
if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("")

    ap.add_argument("-i", "--image", required=True, help="image name in Data dir ")
    # ap.add_argument("-s", "--save_as", required=True, help="save as ")
    args = vars(ap.parse_args())

    img_path = '../Data/' + str(args["image"]  + '.jpg')
    #save_as = '../Data/' + str(args["save_as"] + '.jpg' )

    '''
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CP_env = gym.make('CartPole-v0')
my_seed = 0



agent = PGAgent(CP_env, seed = my_seed, device = device,
                BATCH_SIZE  = 64,
               GAMMA = 0.99,
               LR = 5e-3,
               SCORE_WINDOW_LEN = 100,### no. of episodes for plotting learning curve
               TARGET_SCORE = 19bool0,)
continue_training = int(input('continue_training..?'))
save_dir_name = PGAgent.find_model_dir_title(dir_path = '/content/drive/MyDrive/RL_implementations/Cart_Pole-V0/PG', continue_training = continue_training)


if(continue_training==0):
  print('Created\t',save_dir_name )
  os.mkdir(save_dir_name)

agent.train(save_dir_name,
            epochs = 500, apply_reward2go = True, apply_AdvNorm = True)
    
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





    env = ap.add_argument("-env", "--environment", required=True, help = "Environment to be run")
    seed = ap.add_argument("-seed", "--seed", required=True, help = "seed to run")
    device  = ap.add_argument("-dev", "--device", required = True,help = "device")

    BATCH_SIZE= ap.add_argument("-bs", "--batch_size", required = True,help = "batch size")
    GAMMA =  ap.add_argument("-gam", "--gamma", required = True,help = "gamma")
    LR  = ap.add_argument("-lr", "--lr", required = True,help = "learning rate")
    SCORE_WINDOW_LEN = 100
    TARGET_SCORE  = ap.add_argument("-ts", "--target_score", required = True,help = "score to start saving models") 
    dir_path = ap.add_argument("-dir_path", "--save_dir_path", required= True,help = "path of the dir for saving results" )
    continue_train = ap.add_argument("-cont_train", "--continue_training", required=True, help = "asks if need to pick up from last train")
    epochs =  ap.add_argument("-epochs", "--epochs", required=True, help = "epochs")

    agent = PGAgent( env = env, seed = seed, device = device, BATCH_SIZE= BATCH_SIZE, GAMMA = GAMMA,
            LR = LR, SCORE_WINDOW_LEN= SCORE_WINDOW_LEN, TARGET_SCORE= TARGET_SCORE)
    

    save_dir_name = PGAgent.find_model_dir_title(dir_path, bool(continue_train) )

    if(continue_train==0):
        print('Created\t',save_dir_name )
        os.mkdir(save_dir_name)

    agent.train(save_dir_name,
            epochs = epochs, apply_reward2go = True, apply_AdvNorm = True)
