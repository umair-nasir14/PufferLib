import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PufferLib.pufferlib.frameworks.cleanrl import RecurrentPolicy, sample_logits


import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=2):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class CNN_Policy(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=1, flat_size=64*7*7):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
        )

class CustomRecurrentPolicy(RecurrentPolicy):
    def get_action_and_value(self, x, state=None, action=None):
        print(f"Input shape to get_action_and_value: {x.shape}")
        result = self.policy(x, state)
        print(f"Result from self.policy: {[type(r) for r in result]}")
        print(f"Result shapes: {[r.shape if isinstance(r, torch.Tensor) else type(r) for r in result]}")
        
        logits, value, _, _, state = result
        
        logits = torch.as_tensor(logits).float()
        value = torch.as_tensor(value).float()
        
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        print(f"Logits shape after processing: {logits.shape}")
        print(f"Value shape after processing: {value.shape}")
        
        action, logprob, entropy = self.sample_logits(logits, action)
        return action, logprob, entropy, value, state

    def sample_logits(self, logits, action=None):
        print(f"Logits shape in sample_logits: {logits.shape}")
        probs = F.softmax(logits, dim=-1)
        print(f"Probs shape: {probs.shape}")
        
        if action is None:
            action = torch.argmax(probs, dim=-1).unsqueeze(-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * probs).sum(-1)
        return action, log_prob.gather(-1, action), entropy

    def act(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs).float()
            obs = obs.to("cpu")
            
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension if missing
            
            action, _, _, _, _ = self.get_action_and_value(obs)
            return action.squeeze().cpu().numpy()

# Keep the standalone function for compatibility
def sample_logits(logits, action=None):
    logits = torch.as_tensor(logits).float()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=-1)
    if action is None:
        action = torch.argmax(probs, dim=-1).unsqueeze(-1)
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -(log_prob * probs).sum(-1)
    return action, log_prob.gather(-1, action), entropy

class Policy(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=512):
        super().__init__()
        
        self.encoder = nn.Linear(np.prod(
            env.single_observation_space.shape), hidden_size)

        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        if self.is_multidiscrete:
            action_nvec = env.single_action_space.nvec
            self.decoder = nn.ModuleList([pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, n), std=0.01) for n in action_nvec])
        else:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value_head = nn.Linear(hidden_size, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, observations):
        device = self.encoder.weight.device
        observations = observations.to(device)
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Encodes a batch of observations into a batch of hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        # Ensure observations are on the same device as the encoder
        observations = observations.to(self.encoder.weight.device)
        return torch.relu(self.encoder(observations.float())), None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        value = self.value_head(hidden)
        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoder]
            return actions, value

        actions = self.decoder(hidden)
        return actions, value

class CNNPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.observation_shape = env.single_observation_space.shape
        
        # Embedding layer
        self.embedding = nn.Embedding(6, 6)  # 6 unique indices, embedding size 16
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened output after convolutions
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_head = nn.Linear(256, env.single_action_space.n)
        self.value_head = nn.Linear(256, 1)

    def _get_conv_output_size(self):
        # Helper function to calculate the size of the flattened output after convolutions
        sample_input = torch.zeros(1, *self.observation_shape)
        x = self.embedding(sample_input.long())
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        conv_output = self.conv2(self.conv1(x))
        #conv_output = self.conv3(self.conv2(self.conv1(x)))
        return int(np.prod(conv_output.size()))

    def forward(self, obs):
        # Convert obs to LongTensor
        x = obs.long() if isinstance(obs, torch.Tensor) else torch.LongTensor(obs)
        
        # Embedding
        x = self.embedding(x)
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        # Flatten
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Action and value heads
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        return action_logits, value
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_shape = env.single_observation_space.shape

        # Embedding layer
        self.embedding = nn.Embedding(6, 32)

        # Use a CNN to process spatial information
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size()
        
        # Adjust the fc_reduce layer to match the conv output size
        self.fc_reduce = nn.Linear(conv_output_size, 512)
        self.fc1 = nn.Linear(512, 256)

        # LSTM layer
        self.lstm_hidden_size = 256
        self.lstm_num_layers = 2
        self.lstm = nn.LSTM(256, self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)

        self.fc2 = nn.Linear(256, 128)
        # Action and value heads
        self.action_head = nn.Linear(128, env.single_action_space.n)
        self.value_head = nn.Linear(128, 1)

    def _get_conv_output_size(self):
        sample_input = torch.zeros(1, *self.observation_shape)
        x = self.embedding(sample_input.long())
        if x.dim() == 5:
            x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()[1:]))

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
        return (h0, c0)

    def forward(self, obs, hidden):
        print(f"Input shape: {obs.shape}")

        x = obs.long() if not obs.dtype == torch.long else obs
        x = x.to(self.fc1.weight.device)

        # Embedding
        x = self.embedding(x)
        
        if obs.dim() == 2:
        # Handle the case where obs is (batch_size, flattened_obs)
            obs = obs.view(obs.size(0), 1, 6, 11)  # Reshape to expected format
        elif obs.dim() == 3:
            obs = obs.unsqueeze(1)  # Add channel dimension if missing
        elif obs.dim() == 4:
            pass  # This is the expected shape, do nothing
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
         # Handle different input dimensions
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            pass
        elif x.dim() == 5:
            b, n, c, h, w = x.size()
            x = x.view(b * n, c, h, w)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")
        
        # Rearrange dimensions for CNN
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        print(f"Shape before fc_reduce: {x.shape}")
        
        # Dynamically adjust fc_reduce layer if needed
        if x.size(1) != self.fc_reduce.in_features:
            self.fc_reduce = nn.Linear(x.size(1), 512).to(x.device)
        
        # Dimensionality reduction
        x = F.relu(self.fc_reduce(x))
        x = F.relu(self.fc1(x))
        
        # Prepare for LSTM: (batch_size, seq_len=1, input_size=256)
        x = x.unsqueeze(1)
        
        # Adjust hidden state if batch sizes don't match
        if x.size(0) != hidden[0].size(1):
            hidden = self.init_hidden(x.size(0))
            hidden = tuple(h.to(x.device) for h in hidden)
        
        # LSTM
        x, lstm_hidden = self.lstm(x, hidden)
        
        x = F.relu(self.fc2(x.squeeze(1)))
        # Action and Value Heads
        action_logits = self.action_head(x)
        value = self.value_head(x)

        return action_logits, value, lstm_hidden