import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinerlkit.policy import SACPolicy

class ConformalQLPolicy(SACPolicy):
    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float = 0.2,
        calibration_alpha: float = 0.1,
        lambda_conf: float = 1.0,
        calibration_size: int = 1000
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.action_space = action_space
        self._calibration_alpha = calibration_alpha
        self._lambda_conf = lambda_conf
        self._calibration_size = calibration_size
        
        # Flag to track if calibration set is initialized
        self._is_calibrated = False
        self.calibration_errors = None

    def _initialize_calibration_set(self, batch: Dict) -> torch.Tensor:
        """
        Initialize calibration set from a batch of data
        """
        with torch.no_grad():
            obss = batch["observations"]
            actions = batch["actions"]
            next_obss = batch["next_observations"]
            rewards = batch["rewards"]
            terminals = batch["terminals"]
            
            # Get target values
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            )
            next_q -= self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q
            
            # Get current predictions
            q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
            
            # Compute absolute errors for both critics
            errors1 = torch.abs(q1 - target_q)
            errors2 = torch.abs(q2 - target_q)
            
            # Combine errors from both critics
            all_errors = torch.cat([errors1, errors2])
            
            # If batch is larger than calibration_size, randomly sample
            if len(all_errors) > self._calibration_size:
                idx = torch.randperm(len(all_errors))[:self._calibration_size]
                all_errors = all_errors[idx]
            
        return all_errors

    def compute_conformal_interval(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal prediction interval based on calibration errors
        """
        if errors is None:
            return torch.tensor(float('inf')).to(self.actor.device)
            
        sorted_errors = torch.sort(errors)[0]
        index = int(np.ceil((1 - self._calibration_alpha) * (len(errors) + 1))) - 1
        index = min(max(index, 0), len(errors) - 1)
        return sorted_errors[index]

    def learn(self, batch: Dict) -> Dict[str, float]:
        # Initialize calibration set if not done yet
        if not self._is_calibrated:
            self.calibration_errors = self._initialize_calibration_set(batch)
            self._is_calibrated = True

        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        batch_size = obss.shape[0]

        # Update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        
        # Get conformal prediction interval from fixed calibration set
        conformal_q = self.compute_conformal_interval(self.calibration_errors)
        
        # Modified actor loss with conformal regularization
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a) + 
                     self._lambda_conf * conformal_q).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Rest of the learning process remains the same
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions),
                self.critic2_old(next_obss, next_actions)
            )
            next_q -= self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "conformal_interval": conformal_q.item()
        }

        return result
