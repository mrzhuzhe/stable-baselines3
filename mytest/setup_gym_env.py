# https://pythonprogramming.net/custom-environment-reinforcement-learning-stable-baselines-3-tutorial/

import gym
from gym import spaces

import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
	apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
	score += 1
	return apple_position, score

def collision_with_boundaries(snake_head):
	if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
		return 1
	else:
		return 0

def collision_with_self(snake_position):
	snake_head = snake_position[0]
	if snake_head in snake_position[1:]:
		return 1
	else:
		return 0

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        N_DISCRETE_ACTIONS = 4
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                        shape=(5+4+ SNAKE_LEN_GOAL,), dtype=np.float32)        

    def step(self, action):
        self.prev_actions.append(action)
        #"""
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        #"""


        # Takes step after fixed time
        """
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        """


        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10

        apple_reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 10000
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            """
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)
            """
            self.done = True
            #print("collision_with_self", collision_with_self(self.snake_position) == 1)

        """
        self.total_reward = len(self.snake_position) - 3  # default length is 3        
        """

        """
        # [Bug] calculate collision twice
        apple_reward = 0
		# Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 10000
        """


        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
        #self.total_reward = len(self.snake_position) - 3 - euclidean_dist_to_apple
        self.total_reward = ((250 - euclidean_dist_to_apple) + apple_reward)/100

        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = -10
        info = {}


        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        #_snake_position_list = []
        #for i in range(SNAKE_LEN_GOAL):        
        #    if i >= len(self.snake_position):
        #        _snake_position_list += [-1, -1]
        #    else:
        #        _snake_position_list += [self.snake_position[i][0], self.snake_position[i][1]]

        #print(_snake_position_list)
        # left right up down
        coner4 = []
        for i in [-1, 1]:
            coner4.append([self.snake_head[0] + i*10, self.snake_head[1]])
        for j in [-1, 1]:
            coner4.append([self.snake_head[0], self.snake_head[1] + j*10])
        conerAllowed = [0 if conner in self.snake_position else 1 for conner in coner4]            

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + conerAllowed + list(self.prev_actions)
        #+ _snake_position_list 
        observation = np.array(observation)

        return observation, self.reward, self.done, info
    def reset(self):
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250,250]
        self.prev_reward = 0

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        
        #_snake_position_list = []
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1) # to create history            
            #if i >= len(self.snake_position):
            #    _snake_position_list += [-1, -1]
            #else:
            #    _snake_position_list += [self.snake_position[i][0], self.snake_position[i][1]]

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + [0, 1, 1, 1] + list(self.prev_actions)
        #+ _snake_position_list 
        
        observation = np.array(observation)

        return observation
    def render(self, mode='human'):
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        if self.done == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)

        # Takes step after fixed time
        #"""
        t_end = time.time() + 0.1
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        #"""
        
    #def close (self):
    #    ...