import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch as snn
import snntorch.surrogate as surrogate
import numpy as np

class DSQN(nn.Module):
    def __init__(self, input_size=69, hidden_size=256, output_size=3, beta=0.9):
        super().__init__()
        self.beta = beta  # Decay rate for neurons
        spike_grad_lstm = surrogate.straight_through_estimator()

        # Spiking LSTM Layers
        self.slstm1 = snn.SLSTM(input_size, hidden_size, spike_grad=spike_grad_lstm)
        self.slstm2 = snn.SLSTM(hidden_size, hidden_size, spike_grad=spike_grad_lstm)

        # Fully Connected Layer for Q-Values
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, num_steps=10):
        """ Forward pass through the SLSTM SNN """

        # Initialize SLSTM states at t=0
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()

        # Store outputs for multiple time steps
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            spk1, syn1, mem1 = self.slstm1(x, syn1, mem1)  # First SLSTM layer
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)  # Output SLSTM layer

            spk2_rec.append(spk2)  # Record spikes
            mem2_rec.append(mem2)  # Record membrane potential
        spk_mean = torch.mean(torch.stack(spk2_rec), dim=0)  # Compute mean spike activation

        return spk_mean

class SNNTrainer:
    def __init__(self, model, target_model, learning_rate, gamma):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, run):
        self.model.train()

        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            run = (run,)

        prediction = self.model(state)
        target = prediction.clone()

        for index in range(len(run)):
            Q_new = reward[index]
            if run[index]:  
                best_action = torch.argmax(self.model(next_state[index].unsqueeze(0))).item()
                target_pred = self.target_model(next_state[index].unsqueeze(0))
                Q_new = reward[index] + self.gamma * target_pred[0][best_action]

            target[index][torch.argmax(action[index]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()
