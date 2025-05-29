import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


device = "cuda" if torch.cuda.is_available() else "cpu"


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-3,
        history_length=0,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=int(1e5))

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 2. sample next batch and perform batch update:
        states, actions, next_states, rewards, terminals = (
            self.replay_buffer.next_batch(self.batch_size)
        )

        # states = torch.FloatTensor(states).to(device) for cartpole
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        # next_states = torch.FloatTensor(next_states).to(device) for cartpole
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
        terminals = torch.FloatTensor(terminals).to(device)
        q_values = self.Q(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 2.1 compute td targets and loss

        with torch.no_grad():

            next_q_values = self.Q_target(next_states)
            max_next_q = next_q_values.max(1)[0]

            td_targets = rewards + self.gamma * max_next_q * (1 - terminals)

        loss = self.loss_function(current_q, td_targets)
        # 2.2 update the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic, timestep):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            pass
            # TODO: take greedy action (argmax)
            # action_id = ...
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) this is for cartpole
            state_tensor = (
                torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            q_values = self.Q(state_tensor)
            action_id = q_values.argmax().item()

        else:
            pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 0
            # ACCELERATE = 3
            # BRAKE = 4
            actions = [0, 1, 2, 3, 4]
            if timestep < 30:
                probs = [0.3, 0.1, 0.1, 0.5, 0.0]
            elif timestep < 500:
                probs = [0.4, 0.1, 0.1, 0.25, 0.15]
            else:
                probs = [0.4, 0.1, 0.1, 0.1, 0.3]
            action_id = np.random.choice(actions, p=probs)
            # action_id = np.random.randint(self.num_actions) for cartpole

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
