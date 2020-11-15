"""
    Bit Investor 모델 학습을 위한 Agent 모듈

    Created on 2020.11.07
    @author: Younghyun Kim
"""
from copy import deepcopy
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from environ import Environment
from bit_allocator import BitInvestor


class BitAgent:
    """
        Bit Investor 학습을 위한 Asynchronous RL Agent Class
    """
    def __init__(self, price_data, in_channels=6,
                 mlength=60, moutdim=4,
                 hlength=24, houtdim=4,
                 dlength=250, doutdim=4,
                 tlength=1440,
                 buying_fee=0.004, selling_fee=0.004,
                 device='cuda:0',
                 model_path='./models/', load_model=False, logdir='./logdir/',
                 process_count=8,
                 lr=0.01, clip_grad=0.5,
                 train_batch=128, grad_batch=4,
                 reward_gamma=0.9, entropy_beta=0.01):

        self.lr = lr
        self.entropy_beta = entropy_beta
        self.grad_batch = grad_batch
        self.reward_gamma = reward_gamma
        self.clip_grad = clip_grad

        self.load_model = load_model
        self.model_path = model_path
        self.logdir = logdir

        self.process_count = process_count
        self.train_batch = train_batch

        self.device = device

        # manager parameters
        self.in_channels = in_channels
        self.mlength = mlength
        self.moutdim = moutdim
        self.hlength = hlength
        self.houtdim = houtdim
        self.dlength = dlength
        self.doutdim = doutdim
        self.tlength = tlength

        # environ parameters
        self.buying_fee = buying_fee
        self.selling_fee = selling_fee
        self.price_data = price_data

        # Environment
        self.env = Environment(price_data, tlength, buying_fee, selling_fee)

        # Bit Investor
        self.investor = BitInvestor(in_channels, mlength,
                                    moutdim, hlength, houtdim,
                                    dlength, doutdim)
        self.investor = self.investor.to(device)
        self.investor.eval()

        if load_model:
            self.investor.load_state_dict(torch.load(model_path,
                                                     map_location=device))

        # Optimizer
        self.optimizer = optim.Adam(self.investor.parameters(),
                                    lr=lr, eps=1e-3)

    def save_investor(self, path):
        """
            학습 모델 저장
        """
        torch.save(self.investor.state_dict(), path)

    def train(self, episodes, iters=10):
        """
            Grad Parallelism으로 학습 진행

            Args:
                episodes: 총 진행 에피소드 수
        """
        writer = SummaryWriter(self.logdir)
        self.investor.train()
        np.random.seed(0)
        torch.manual_seed(0)

        start = datetime.datetime.now()

        scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10000.0,
                                              gamma=0.99)

        mp.set_start_method('spawn', force=True)
        self.investor.share_memory()

        train_queue = mp.Queue(maxsize=self.process_count)
        episode_check_queue = mp.Queue(maxsize=self.process_count)
        timings_queue = mp.Queue(maxsize=self.process_count)

        ep_t = 0
        data_proc_list = []

        for proc_idx in range(self.process_count):
            timings_queue.put(episodes)
            data_proc = mp.Process(target=self.simulation,
                                   args=(self.investor, timings_queue,
                                         train_queue, episode_check_queue,
                                         1e-6, iters))
            data_proc.start()
            data_proc_list.append(data_proc)

        while episodes > 0:
            step_idx = 0
            grad_buffer = None

            while True:
                train_entry = train_queue.get()
                step_idx += 1

                if grad_buffer is None:
                    grad_buffer = train_entry
                else:
                    for tgt_grad, grad in zip(grad_buffer, train_entry):
                        tgt_grad += grad

                if step_idx % self.train_batch == 0:
                    for param, grad in zip(self.investor.parameters(),
                                           grad_buffer):
                        param.grad = torch.FloatTensor(grad).to(self.device)
                    nn_utils.clip_grad_norm_(self.investor.parameters(),
                                             self.clip_grad)
                    self.optimizer.step()
                    scheduler.step()
                    grad_buffer = None

                episode_info = episode_check_queue.get()
                done = episode_info[0]
                episodes -= done

                if len(episode_info) > 1:
                    timings_queue.put(episodes)

                    writer.add_scalar('Reward(Avg)', episode_check_queue[3], ep_t)
                    writer.add_scalar('Cum_Ret', episode_check_queue[4], ep_t)
                    writer.add_scalar('Mu', episode_check_queue[5], ep_t)
                    writer.add_scalar('Sig', episode_check_queue[6], ep_t)
                    writer.add_scalar('IR', episode_check_queue[7], ep_t)

                    ep_t += 1
                    print(ep_t, episode_check_queue[2], episode_check_queue[3])

                    self.save_investor(self.model_path)
        end = datetime.datetime.now()
        print((end - start).total_seconds())

    def simulation(self, investor, timings_queue, train_queue,
                   episode_check_queue, eps=1e-6, iters=10):
        """
            비트코인 투자 시뮬레이션

            Return:
                grads, done, reward, return
        """
        print("simulation beginning...")

        while True:
            episodes = timings_queue.get()

            if episodes <= 0:
                break

            for i in range(iters):
                t_pos = 0

                env = deepcopy(self.env)

                beg_pos = np.random.randint(self.env.min_beg,
                                            self.env.price_data.shape[0] - \
                                            (self.tlength - 1), 1).item()

                obs, done = env.reset(beg_pos=beg_pos)

                for j, ob in enumerate(obs):
                    obs[j] = ob.to(self.device)

                rewards_seqs = []
                rets_seqs = []
                states, probs_list, q_values_list = [], [], []

                weights_prev = torch.zeros(1, 2).to(self.device)

                init = True

                while not done:
                    t_pos += 1

                    actions, _, q_values = investor(*obs, weights_prev,
                                                    init=init, train=True)

                    next_obs, reward, ret, latest_w_prev, done =\
                        env.step(actions.to('cpu'),
                                 weights_prev.view(-1).to('cpu'))

                    states.append(obs)
                    probs_list.append(actions)
                    q_values_list.append(q_values)

                    rets_seqs.append(ret)
                    rewards_seqs.append(reward)

                    if actions[0, 0] > 0.5:
                        weights_prev = actions[0, 1:].view(1, -1)
                    else:
                        weights_prev = latest_w_prev.view(1, -1)

                    for j, ob in enumerate(next_obs):
                        next_obs[j] = ob.to(self.device)

                    init = False

                    with torch.no_grad():
                        next_actions, _, next_q_values = investor(*next_obs,
                                                                  weights_prev,
                                                                  init=init,
                                                                  train=True)

                    if done or t_pos == self.grad_batch:
                        grads = self.accumulate_grads(investor,
                                                      probs_list[-t_pos:],
                                                      q_values_list[-t_pos:],
                                                      next_actions, next_q_values,
                                                      rewards_seqs[-t_pos:],
                                                      weights_prev)
                        train_queue.put(grads)

                        if not done:
                            episode_check_queue.put([0])
                        else:
                            with torch.no_grad():
                                rewards_seqs = np.array(rewards_seqs)
                                rets_seqs = torch.tensor(rets_seqs).view(-1)
                                reward = rewards_seqs.sum() / \
                                        rewards_seqs.shape[0]

                                cum_ret = ((1. + rets_seqs).prod() - 1.).item()

                                mu = rets_seqs.mean()
                                sig = rets_seqs.sd()
                                ir = mu / (sig + eps)

                            if (i + 1) == iters:
                                ep = 1
                            else:
                                ep = 0
                            episode_check_queue.\
                                put([ep, i,
                                     env.price_data.index[beg_pos],
                                     reward, cum_ret, mu.item(), sig.item(),
                                     ir.detach().item()])

                    obs = next_obs

    def discounted_prediction(self, next_weights, next_q_values,
                              rewards, weights_prev):
        """
            보상값 G 계산
        """
        discounted_return = np.zeros_like(rewards)

        if next_weights[0, 0] > 0.5:
            running_add =\
                next_weights[:, 1:].matmul(next_q_values.transpose(0, 1))
        else:
            running_add =\
                weights_prev.unsqueeze(0).matmul(next_q_values.transpose(0, 1))

        for t in reversed(range(0, discounted_return.shape[0])):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_return[t] = running_add.item()

        return discounted_return

    def accumulate_grads(self, investor, weights, q_values, next_weights,
                         next_q_values, rewards, weights_prev):
        """
            Investor gradient 계산 및 적재
        """
        investor.zero_grad()

        discountd_return = self.discounted_prediction(next_weights, next_q_values,
                                                      rewards, weights_prev)
        g_return =\
                torch.FloatTensor(discountd_return.astype(float)).to(self.device)

        # Mean Q_value 계산
        weights = torch.cat(weights)

        qs = torch.zeros_like(weights)
        q_values = torch.cat(q_values)
        qs[:, 1:] = q_values
        qmax, _ = q_values.max(dim=-1)
        qs[:, 0] = 2 * qmax
        q_values = qs

        _, max_w_indices = weights.max(dim=-1)
        max_q = q_values[np.arange(q_values.shape[0]), max_w_indices]
        mean_q = (weights * q_values).sum(1).view(-1, 1).detach()

        # Advantage 계산
        adv = max_q.detach() - mean_q

        # Policy loss 계산
        log_prob_v = F.log_softmax(weights, dim=1)
        log_prob_actions_v = adv * log_prob_v[np.arange(weights.shape[0]),
                                              max_w_indices]
        loss_policy_v = -log_prob_actions_v.mean()

        # Entropy loss 계산
        entropy_v = (weights * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = self.entropy_beta * entropy_v

        # q_value loss 계산
        loss_value_v = F.mse_loss(g_return, max_q.squeeze(-1))

        # 전체 loss & grad 계산
        loss = loss_policy_v + entropy_loss_v + loss_value_v
        loss.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(investor.parameters(), self.clip_grad)

        grads = [param.grad.data.cpu().numpy() if param.grad is not None else None
                 for param in investor.parameters()]

        return grads
