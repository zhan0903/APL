from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
# import ray

import scipy.signal
# from utils.mpi_tools import mpi_statistics_scalar
# import statistics 

FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100 #100
_last_free_time = 0.0
_to_free = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def ray_get_and_free(object_ids):
#     """Call ray.get and then queue the object ids for deletion.

#     This function should be used whenever possible in RLlib, to optimize
#     memory usage. The only exception is when an object_id is shared among
#     multiple readers.

#     Args:
#         object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

#     Returns:
#         The result of ray.get(object_ids).
#     """

#     global _last_free_time
#     global _to_free

#     result = ray.get(object_ids)
#     # print("ray_get_and_free,object_ids",object_ids)

#     # print("ray_get_and_free,result",result)

#     if type(object_ids) is not list:
#         object_ids = [object_ids]
#     _to_free.extend(object_ids)

#     # batch calls to free to reduce overheads
#     now = time.time()
#     # print("len of to_free-------------------,",len(_to_free))
#     if (len(_to_free) > MAX_FREE_QUEUE_SIZE
#             or now - _last_free_time > FREE_DELAY_S):
#         ray.internal.free(_to_free)
#         _to_free = []
#         _last_free_time = now

#     return result


def aligned_array(size, dtype, align=64):
    """Returns an array of a given size that is 64-byte aligned.

    The returned array can be efficiently copied into GPU memory by TensorFlow.
    """

    n = size * dtype.itemsize
    empty = np.empty(n + (align - 1), dtype=np.uint8)
    data_align = empty.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    output = empty[offset:offset + n].view(dtype)

    assert len(output) == size, len(output)
    assert output.ctypes.data % align == 0, output.ctypes.data
    return output


def concat_aligned(items):
    """Concatenate arrays, ensuring the output is 64-byte aligned.

    We only align float arrays; other arrays are concatenated as normal.

    This should be used instead of np.concatenate() to improve performance
    when the output array is likely to be fed into TensorFlow.
    """

    if len(items) == 0:
        return []
    elif len(items) == 1:
        # we assume the input is aligned. In any case, it doesn't help
        # performance to force align it since that incurs a needless copy.
        return items[0]
    elif (isinstance(items[0], np.ndarray)
        and items[0].dtype in [np.float32, np.float64, np.uint8]):
        dtype = items[0].dtype
        flat = aligned_array(sum(s.size for s in items), dtype)
        batch_dim = sum(s.shape[0] for s in items)
        new_shape = (batch_dim, ) + items[0].shape[1:]
        output = flat.reshape(new_shape)
        assert output.ctypes.data % 64 == 0, output.ctypes.data
        np.concatenate(items, out=output)
        return output
    else:
        return np.concatenate(items)

class ReplayBuffer_offline(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # used for sac instead of bcq
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

    # used for bcq
    def sample(self, batch_size, train=True):
        # ind = np.random.randint(0, self.size, size=batch_size)

        if train:
            ind = np.random.choice(self.train_list, batch_size, replace=True)  # replaceable
        else:
            ind = np.random.choice(self.evaluation_list, batch_size, replace=True) # replaceable

        # if use_bootstrap:
        #     return (
        #     torch.FloatTensor(self.state[ind]).to(self.device),
        #     torch.FloatTensor(self.action[ind]).to(self.device),
        #     torch.FloatTensor(self.next_state[ind]).to(self.device),
        #     torch.FloatTensor(self.reward[ind]).to(self.device),
        #     torch.FloatTensor(self.not_done[ind]).to(self.device),
        #     torch.FloatTensor(self.bootstrap_mask[ind]).to(self.device),
        # )

        # batch = dict(
        #     state=torch.FloatTensor(self.state[ind]).to(self.device),
        #     action=torch.FloatTensor(self.action[ind]).to(self.device),
        #     next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
        #     reward=torch.FloatTensor(self.reward[ind]).to(self.device),
        #     not_done=torch.FloatTensor(self.not_done[ind]).to(self.device)
        # )

        # return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)


    def load(self, save_folder, size=-1, bootstrap_dim=None):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        self.all_idx = np.array([i for i in range(self.size)])

        self.train_size = int(0.9*self.size)  # 0.8
        self.train_list = np.random.choice(self.size, self.train_size, replace=False)

        self.evaluation_list = np.setdiff1d(self.all_idx, self.train_list)
        self.evaluation_size = len(self.evaluation_list)
        
        # # Adjust crt_size if we're using a custom size
        # size = min(int(size), self.max_size) if size > 0 else self.max_size
        # self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]
        # if bootstrap_dim is not None:
        #     self.bootstrap_dim = bootstrap_dim
        #     bootstrap_mask = np.random.binomial(n=1, size=(1, self.size, bootstrap_dim,), p=0.8)
        #     bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
        #     self.bootstrap_mask = bootstrap_mask[:self.size]


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    # store mutiple epsodes
    def add_epsodes(self,epsodes):
        for epsode in epsodes:
            # print(f"len of epsode:{len(epsode)}")
            for experience in epsode:
                self.store(experience[0],experience[1],experience[2],experience[3],experience[4])
        # self.num_added += len(epsode)
    
    # store one experience
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes=None, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        # self.act_scaled_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.act_old_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        if info_shapes: 
            self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
            self.sorted_info_keys = keys_as_sorted_list(self.info_bufs)
        else:
            self.sorted_info_keys = None
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, act_old,info=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_old_buf[self.ptr] = act_old
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        if self.sorted_info_keys:
            for i, k in enumerate(self.sorted_info_keys):
                self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # print(f"self.adv_buf:{self.adv_buf},type:{type(self.adv_buf)}")

        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        global_sum,global_n = np.sum(self.adv_buf),len(self.adv_buf)
        
        adv_mean = global_sum/global_n
        global_sum_sq = np.sum((self.adv_buf - adv_mean)**2)
        adv_std = np.sqrt(global_sum_sq/global_n)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        if self.sorted_info_keys: 
            assert False
            l = values_as_sorted_list(self.info_bufs)
        else:
            l = []

        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf,self.act_old_buf] + l

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input:
            vector x,
            [x0,
            x1,
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

