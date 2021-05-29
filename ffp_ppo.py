"""
    PPO Trainer for FFPTrader

    Created on 2021.04.07
    @author: Younghyun Kim
"""
import os
from copy import deepcopy
import math
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from environ import Environ, env_config
from ffp_trader import FFPTrader

TRAINER_CONFIG = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'trajectory_size': 128,
            'lr': 1e-4,
            'ppo_eps': 0.2,  # clipping value for (new / old)
            'ppo_epoches': 10,
            'ppo_batch_size': 64,
            'test_iters': 1000,
            'device': 'cuda:0',
            'model_path': './models/',
            'model_name': 'FFPTrader',
            'load_model_path': None,
            'load_model': False,
            'logdir': './logdir/',
         }


class FFPTPPOTrainer:
    """
        PPO Trainer for FFP Trader
    """
    def __init__(self, trader, ffp_env, config=TRAINER_CONFIG):
        " Initiation "
        self.trader = trader  # FFPTrader
        self.env = ffp_env  # Environ

        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.trajectory_size = config['trajectory_size']
        self.lr = config['lr']

        self.ppo_eps = config['ppo_eps']
        self.ppo_epoches = config['ppo_epoches']
        self.ppo_batch_size = config['ppo_batch_size']
        self.test_iters = config['test_iters']
        self.device = config['device']
        self.model_path = config['model_path']
        self.model_name = config['model_name']
        self.load_model_path = config['load_model_path']
        self.load_model = config['load_model']
        self.logdir = config['logdir']

        self.trader = self.trader.to(self.device)

        if self.load_model:
            self.load_trader(self.load_model_path)

        self.opt = optim.Adam(self.trader.parameters(),
                              lr=self.lr)

    def save_trader(self, path):
        """
            학습 모델 저장
        """
        torch.save(self.trader.state_dict(), path)

    def load_trader(self, path):
        """
            학습된 모델 가져오기
        """
        self.trader.load_state_dict(
                torch.load(path, map_location=self.device))
        self.trader.eval()

    @staticmethod
    def calc_logprob(mu_v, logstd_v, actions_v, cmin=1e-3):
        p1 = - ((mu_v - actions_v) ** 2) /\
                (2 * torch.exp(logstd_v).clip(min=cmin))
        p2 = - torch.log(
            torch.sqrt(2 * math.pi * torch.exp(logstd_v)))

        return p1 + p2

    def calc_adv_ref(self, traj_done,
                     rewards_v, values_v, device='cpu'):
        """
            Calculation for advantage and 1-step ref value

            Args:
                traj_done: done list
                rewards_v: reward values tensor
                value_v: critic values tensor
            Return:
                tuple with advantage numpy array and
                reference values
        """
        rewards = rewards_v.data.cpu().numpy().ravel()
        values = values_v.data.cpu().numpy().ravel()

        # generalized advantage estimator:
        #  smoothed version of the advantage
        last_gae = 0.0
        result_adv, result_ref = [], []

        for val, next_val, reward, done in\
            zip(reversed(values[:-1]), reversed(values[1:]),
                reversed(rewards[:-1]), reversed(traj_done[:-1])):
            if done:  # done
                delta = reward - val  # exp[2]: reward
                last_gae = delta
            else:
                delta = reward + self.gamma * next_val - val
                last_gae =\
                    delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(
                list(reversed(result_adv))).to(device)
        ref_v = torch.FloatTensor(
                list(reversed(result_ref))).to(device)

        return adv_v, ref_v

    def train(self, episodes=20000, seed=0):
        """
            Train PPO

            Args:
                episodes: # of episodes
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        writer = SummaryWriter(self.logdir)

        self.trader.train()

        for _ in range(episodes):
            trajectory = []
            step_idx = 0
            done = False

            obs = self.env.reset()
            order_series, total_bid_ask,\
                price_series, possession =\
                self._processing_obs(obs)

            while not done:
                policy, value = self.trader(order_series,
                                            total_bid_ask,
                                            price_series,
                                            possession)

                action, actions =\
                    self.get_action(policy,
                                    self.trader.policy_logstd)

                next_obs, reward, done = self.env.step(action)

                trajectory.append([(order_series, total_bid_ask,
                                    price_series, possession),
                                   actions,
                                   value, reward, done])

                order_series, total_bid_ask,\
                    price_series, possession =\
                    self._processing_obs(next_obs)

                if len(trajectory) < self.trajectory_size or\
                        not done:
                    continue

                # states
                traj_orders = [t[0][0] for t in trajectory]
                traj_tba = [t[0][1] for t in trajectory]
                traj_prices = [t[0][2] for t in trajectory]
                traj_poss = [t[0][3] for t in trajectory]

                traj_actions = [t[1] for t in trajectory]
                traj_values = [t[2] for t in trajectory]
                traj_rewards = [t[3] for t in trajectory]
                traj_done = [t[4] for t in trajectory]

                traj_orders_v = torch.cat(traj_orders)
                traj_tba_v = torch.cat(traj_tba)
                traj_prices_v = torch.cat(traj_prices)
                traj_poss_v = torch.cat(traj_poss)

                traj_actions_v =\
                    torch.cat(traj_actions).to(self.device)
                traj_values_v =\
                    torch.cat(traj_values)
                traj_rewards_v =\
                    torch.FloatTensor(traj_rewards).to(self.device)

                traj_adv_v, traj_ref_v =\
                    self.calc_adv_ref(traj_done, traj_rewards_v,
                                      traj_values_v,
                                      device=self.device)

                mu_v, _ = self.trader(traj_orders_v, traj_tba_v,
                                      traj_prices_v, traj_poss_v)
                old_logprob_v =\
                    self.calc_logprob(mu_v,
                                      self.trader.policy_logstd,
                                      traj_actions_v)

                # normalize advantages
                traj_adv_v =\
                    (traj_adv_v - torch.mean(traj_adv_v)) /\
                    torch.std(traj_adv_v)

                # drop last entry from the trajectory,
                # an our adv and ref value calculated without it
                trajectory = trajectory[:-1]
                old_logprob_v = old_logprob_v[:-1].detach()

                sum_loss_value = 0.0
                sum_loss_policy = 0.0
                count_steps = 0

                batch_size = self.ppo_batch_size

                for epoch in range(self.ppo_epoches):
                    for batch_s in range(0, len(trajectory),
                                         batch_size):
                        orders_v =\
                            traj_orders_v[batch_s:batch_s+batch_size]
                        tba_v = traj_tba_v[batch_s:batch_s+batch_size]
                        prices_v =\
                            traj_prices_v[batch_s:batch_s+batch_size]
                        poss_v = traj_poss_v[batch_s:batch_s+batch_size]

                        actions_v =\
                            traj_actions_v[batch_s:batch_s+batch_size]
                        batch_adv_v =\
                            traj_adv_v[batch_s:batch_s+batch_size]
                        batch_ref_v =\
                            traj_ref_v[batch_s:batch_s+batch_size]
                        batch_old_logprob_v =\
                            old_logprob_v[batch_s:batch_s+batch_size]

                        # trader training
                        self.opt.zero_grad()
                        mu_v, value_v =\
                            self.trader(orders_v, tba_v,
                                        prices_v, poss_v)
                        loss_value_v = F.mse_loss(value_v.squeeze(-1),
                                                  batch_ref_v)

                        logprob_pi_v =\
                            self.calc_logprob(mu_v,
                                              self.trader.policy_logstd,
                                              actions_v)
                        ratio_v =\
                            torch.exp(
                                logprob_pi_v - batch_old_logprob_v)
                        surr_obj_v = batch_adv_v * ratio_v
                        clipped_surr_v =\
                            batch_adv_v *\
                            torch.clamp(ratio_v,
                                        1. - self.ppo_eps,
                                        1. + self.ppo_eps)
                        loss_policy_v =\
                            -torch.min(surr_obj_v,
                                       clipped_surr_v).mean()

                        total_loss = loss_value_v + loss_policy_v
                        total_loss.backward()

                        self.opt.step()

                        sum_loss_value += loss_value_v.item()
                        sum_loss_policy += loss_policy_v.item()
                        count_steps += 1
                trajectory.clear()
                writer.add_scalar("advantage",
                                  traj_adv_v.mean().item(), step_idx)
                writer.add_scalar("values", traj_ref_v.mean().item(),
                                  step_idx)
                writer.add_scalar("loss_policy",
                                  sum_loss_policy / count_steps,
                                  step_idx)
                writer.add_scalar("loss_value",
                                  sum_loss_value / count_steps,
                                  step_idx)
                step_idx += 1
                self.save_trader(self.model_path+self.model_name+".pt")

    @staticmethod
    def get_action(policy_probs, policy_logstd):
        " get action "
        policies = policy_probs.detach().cpu().numpy()
        policies = policies.ravel()
        logstd = policy_logstd.data.cpu().numpy().ravel()

        actions = policies + np.exp(logstd) *\
                np.random.normal(size=logstd.shape)

        action = np.argmax(actions)

        actions =\
            torch.FloatTensor(actions.astype(float)).view(1, -1)

        return action, actions

    def _processing_obs(self, obs):
        """
            관찰 데이터를 가져와서 가공하는 메소드

            Args:
                obs: 관찰변수
                    * 구성요소
                        orderbooks (length X 2 X 30)
                        total_bid_ask (length X 2)
                        price_series (length)
                        possession (length)
        """
        order_series, total_bid_ask,\
            price_series, possession = obs

        order_series =\
            torch.FloatTensor(order_series.astype(float))
        total_bid_ask =\
            torch.FloatTensor(total_bid_ask.astype(float))
        price_series =\
            torch.FloatTensor(price_series.astype(float))
        possession =\
            torch.FloatTensor(possession.astype(float))

        order_series =\
            order_series.view(1, order_series.shape[0],
                              order_series.shape[1],
                              -1).to(self.device)
        total_bid_ask =\
            total_bid_ask.view(1,
                               total_bid_ask.shape[0],
                               -1).to(self.device)
        price_series =\
            price_series.view(1, -1, 1).to(self.device)
        possession = possession.view(1, -1, 1).to(self.device)

        return order_series, total_bid_ask,\
            price_series, possession
