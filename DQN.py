import pygame
import random
import numpy as np 
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from collections import deque
import pickle

height = 200
width = 200
window = pygame.display.set_mode((width, height))
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
pygame.display.set_caption("blob")
fps = 10
clock = pygame.time.Clock()
pygame.init()


class Blob:
    def __init__(self, color, x, y):
        self.x = x # random.randrange(0, width, 20)
        self.y = y # random.randrange(0, height, 20)
        self.color = color

    def draw(self):
        pygame.draw.rect(window, self.color, [self.x, self.y, 20, 20])

    def move(self, action):
        if action == 0: self.y -= 20 # move UP
        elif action == 1: self.y += 20 # move DOWN
        elif action == 2: self.x -= 20 # move LEFT
        elif action == 3: self.x += 20 # move RIGHT
        else: pass
        # collision detection
        if self.x <= 0: self.x = 0
        if self.x >= width: self.x = width-20
        if self.y <= 0: self.y = 0
        if self.y >= height: self.y = height-20


class Game:
    def __init__(self):
        self.agent = Blob(white, 0, 0)#Blob(white, random.randrange(0, width, 20), random.randrange(0, height, 20))
        self.goal = Blob(green, 180, 180)
        self.enemy = Blob(red, 20, 60)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.enemy1 = Blob(red, 140, 80)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.enemy2 = Blob(red, 120, 100)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.action_space = [0, 1, 2, 3]
        self.done = False
        self.observation_space = []
        for x in range(0, width, 20):
            for y in range(0, height, 20):
                self.observation_space.append([x, y])

    def render(self):
        window.fill(black)
        self.agent.draw()
        self.goal.draw()
        self.enemy.draw()
        self.enemy1.draw()
        self.enemy2.draw()
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                return True 
        pygame.display.update()
        clock.tick(fps)

    def step(self, action):
        self.agent.move(action) 
        if self.agent.x == self.goal.x and self.agent.y == self.goal.y: 
            self.reward = 30
            self.done = True 
            print("Goal")
        elif self.agent.x == self.enemy.x and self.agent.y == self.enemy.y: 
            self.reward = -5
            self.done = True 
        elif self.agent.x == self.enemy1.x and self.agent.y == self.enemy1.y: 
            self.reward = -5
            self.done = True 
        elif self.agent.x == self.enemy2.x and self.agent.y == self.enemy2.y: 
            self.reward = -5
            self.done = True 
        else: 
            self.reward = -1
        self.next_step = [self.agent.x, self.agent.y]
        return self.next_step, self.reward, self.done

    def reset(self):
        self.agent = Blob(white, 0, 0)#Blob(white, random.randrange(0, width, 20), random.randrange(0, height, 20))
        self.goal = Blob(green, 180, 180)
        self.enemy = Blob(red, 20, 60)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.enemy1 = Blob(red, 140, 80)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.enemy2 = Blob(red, 120, 100)#Blob(red, random.randrange(0, width, 20), random.randrange(0, width, 20))
        self.action_space = [0, 1, 2, 3]
        self.done = False
        return [self.agent.x, self.agent.y]

'''
make a random policy nn
copy that nn for target nn
pass state of env 
select the action depending on epsilon 
store the experices in replay memory 
select the random samples from the replay memory 
replay memory consists tuples like (state, action, reward, next state) 
pass the state from the replay memory to the policy nn again
select the q value for that action in the replay memory for that particular state from the output of the policy nn
pass the 'next state' from the replay memory to target nn for the max q value of the next state
calculate the optimal q value with the max q value of next state (q*) 
q*(state, action) = reward + discount*(max(q(next state, next action))) 
create the data with states as input and their optimal q values as labels
fit the created data with states and their optimal q values at the end of each episode
update the target nn with the weights and biases of ever updating policy nn per N episodes
'''

def create_model():
    model = Sequential()
    model.add(Dense(128, activation = 'relu', input_shape = (2,)))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(4))
    model.compile(optimizer = 'adam', loss = 'mse')
    return model 

def deepQNetworks(training = False, testing = False, model = None, replay_mem = None, episodes_to_watch = 10):
    if training:
        run = True
        epsilon = 0.608
        epsilon_decay_rate = 0.999
        epsilon_min = 0.001
        discount = 0.99
        max_steps = 100
        episodes = 1100
        start_episode = 499
        all_rewards = []
        target_update_counter = 0
        if model == None: 
            policy_nn = create_model()
            replay_memory = deque(maxlen = 10000)
        else: 
            policy_nn = model 
            replay_memory = replay_mem
        target_nn = create_model()
        target_nn.set_weights(policy_nn.get_weights())
        env = Game()
        for episode in range(start_episode, episodes+1):
            current_reward = 0
            state = np.reshape(env.reset(), (1, 2))
            X = []
            y = []
            for step in range(max_steps):
                if random.random() >= epsilon:
                    action = np.argmax(policy_nn.predict(state))
                else:
                    action = random.choice(env.action_space)

                new_state, reward, done = env.step(action)
                new_state = np.reshape(new_state, (1, 2))
                replay_memory.append((state, action, reward, new_state)) 
                state = new_state 
                current_reward += reward 

                if done:
                    if len(replay_memory) >= 64:
                        for exp in replay_memory:
                            state = exp[0]
                            action = exp[1]
                            reward = exp[2]
                            new_state = exp[3]
                            q_values = policy_nn.predict(state)
                            optimal_q_value = reward + discount * (np.max(target_nn.predict(new_state))) 
                            q_values[0][action] = optimal_q_value
                            new_q_values = q_values
                            X.append(state)
                            y.append(new_q_values)

                        z = list(zip(X, y))
                        random.shuffle(z)
                        X, y = zip(*z)
                        X = np.vstack(X)
                        y = np.vstack(y)
                        policy_nn.fit(X, y, batch_size = 64, verbose = 0)

                    target_update_counter += 1 
                    if target_update_counter == 10:
                        target_nn.set_weights(policy_nn.get_weights()) 
                        target_update_counter = 0

                    break
            print(f"episode: {episode}, with reward: {current_reward}, and epsilon: {epsilon}, RM length: {len(replay_memory)}") 

            all_rewards.append(current_reward)
            if epsilon <= epsilon_min:
                epsilon = epsilon_min
            else:
                epsilon *= epsilon_decay_rate
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    run = False
            if run == False:
                break
            
        policy_nn.save('policy2.h5')
        pickle.dump(replay_memory, open("policy2-RM", "wb"))

        rewards = 0
        for r in range(len(all_rewards)):
            rewards += all_rewards[r] 
            if r % 100 == 0:
                avg_reward_per_100_eps = rewards/100
                print(f"Average reward at episode {r}: {avg_reward_per_100_eps}")
                rewards = 0

    if testing:
        run = True
        if model is None:
            print("model not found!")
        else:
            test_model = model 
            env = Game()
            max_steps = 100
            for episode in range(episodes_to_watch):
                state = np.reshape(env.reset(), (1, 2))
                for step in range(max_steps):
                    env.render()
                    action = np.argmax(test_model.predict(state))
                    new_state, _, done = env.step(action)
                    new_state = np.reshape(new_state, (1, 2))
                    state = new_state

                    if done: break
                for event in pygame.event.get():
                    if event.type is pygame.QUIT:
                        run = False
                if run == False:
                    break

model = load_model('policy1.h5')
# replay_memory = pickle.load(open("policy1-RM", "rb"))
deepQNetworks(testing = True, model = model, episodes_to_watch = 30)
# deepQNetworks(training = True, model = model, replay_mem = replay_memory)
# deepQNetworks(training = True)

pygame.quit()
quit()