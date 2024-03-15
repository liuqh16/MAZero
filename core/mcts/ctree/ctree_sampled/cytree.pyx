# distutils: language=c++
# cython: language_level=3
from .ctree cimport CTree_batch
import numpy as np


cdef class Tree_batch:
    cdef CTree_batch *trees
    cdef int root_num, agent_num, action_space_size

    def __cinit__(self, int root_num, int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, unsigned int random_seed, float rho, float lam):
        self.root_num = root_num
        self.agent_num = agent_num
        self.action_space_size = action_space_size

        self.trees = new CTree_batch(root_num, agent_num, action_space_size, sampled_times, simulation_num, tree_value_stat_delta_lb, random_seed, rho, lam)

    def __dealloc__(self):
        del self.trees

    def prepare(self, rewards, values, policy_probs, beta, int sampled_times, float noise_eps, noises):
        rewards = rewards.reshape(-1)
        if not rewards.flags['C_CONTIGUOUS']:
            rewards = np.ascontiguousarray(rewards)
        cdef float[::1] rewards_memview = rewards               # root_num

        values = values.reshape(-1)
        if not values.flags['C_CONTIGUOUS']:
            values = np.ascontiguousarray(values)
        cdef float[::1] values_memview = values                 # root_num

        policy_probs = policy_probs.reshape(-1)
        if not policy_probs.flags['C_CONTIGUOUS']:
            policy_probs = np.ascontiguousarray(policy_probs)
        cdef float[::1] policy_probs_memview = policy_probs     # root_num * agent_num * action_space_size

        beta = beta.reshape(-1)
        if not beta.flags['C_CONTIGUOUS']:
            beta = np.ascontiguousarray(beta)
        cdef float[::1] beta_memview = beta                     # root_num * agent_num * action_space_size

        noises = noises.reshape(-1)
        if not noises.flags['C_CONTIGUOUS']:
            noises = np.ascontiguousarray(noises)
        cdef float[::1] noises_memview = noises                 # root_num * agent_num * action_space_size

        self.trees[0].prepare(&rewards_memview[0], &values_memview[0], &policy_probs_memview[0], &beta_memview[0], sampled_times, noise_eps, &noises_memview[0])


    def batch_selection(self, float pb_c_base, float pb_c_init, float discount):

        hidden_state_index_x_lst = np.empty(self.root_num, order='C', dtype='int32')
        hidden_state_index_y_lst = np.empty(self.root_num, order='C', dtype='int32')
        last_actions = np.empty(self.root_num * self.agent_num, order='C', dtype='int32')

        cdef int[::1] idx_buf = hidden_state_index_x_lst
        cdef int[::1] idy_buf = hidden_state_index_y_lst
        cdef int[::1] act_buf = last_actions 

        self.trees[0].cbatch_selection(pb_c_base, pb_c_init, discount,
                                       &idx_buf[0], &idy_buf[0], &act_buf[0])   # memory buffer to receive the results

        return (
            hidden_state_index_x_lst.tolist(),  # which iter it is expanded
            hidden_state_index_y_lst.tolist(),  # its id in batch, i.e. [0,1,2,..., root_num-1]
            last_actions.reshape(self.root_num, self.agent_num)
        )

    def batch_expansion_and_backup(self, int hidden_state_index_x, float discount, int sampled_times, rewards, values, policy_probs, beta):
        rewards = rewards.reshape(-1)
        if not rewards.flags['C_CONTIGUOUS']:
            rewards = np.ascontiguousarray(rewards)
        cdef float[::1] rewards_memview = rewards               # root_num

        values = values.reshape(-1)
        if not values.flags['C_CONTIGUOUS']:
            values = np.ascontiguousarray(values)
        cdef float[::1] values_memview = values                 # root_num

        policy_probs = policy_probs.reshape(-1)
        if not policy_probs.flags['C_CONTIGUOUS']:
            policy_probs = np.ascontiguousarray(policy_probs)
        cdef float[::1] policy_probs_memview = policy_probs     # root_num * agent_num * action_space_size

        beta = beta.reshape(-1)
        if not beta.flags['C_CONTIGUOUS']:
            beta = np.ascontiguousarray(beta)
        cdef float[::1] beta_memview = beta                     # root_num * agent_num * action_space_size

        self.trees[0].cbatch_expansion_and_backup(hidden_state_index_x, discount, sampled_times,
                                                  &rewards_memview[0], &values_memview[0], &policy_probs_memview[0], &beta_memview[0])

    def get_roots_values(self):
        root_values = np.empty(self.root_num, order='C', dtype='float32')
        cdef float[::1] arr = root_values
        self.trees[0].get_roots_values(&arr[0])
        return root_values

    def get_roots_marginal_visit_count(self):
        visit_count = np.empty(self.root_num * self.agent_num * self.action_space_size, order='C', dtype='int32')
        cdef int[::1] arr = visit_count
        self.trees[0].get_roots_marginal_visit_count(&arr[0])
        return visit_count.reshape(self.root_num, self.agent_num, self.action_space_size)

    def get_roots_marginal_priors(self):
        priors = np.empty(self.root_num * self.agent_num * self.action_space_size, order='C', dtype='float32')
        cdef float[::1] arr = priors
        self.trees[0].get_roots_marginal_priors(&arr[0])
        return priors.reshape(self.root_num, self.agent_num, self.action_space_size)

    def get_roots_sampled_visit_count(self):
        res = []
        cdef int num_children_of_root
        cdef int[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_visit_count = np.empty(num_children_of_root, order='C', dtype='int32')
            arr = sampled_visit_count
            self.trees[0].get_root_sampled_visit_count(i, &arr[0])
            res.append(sampled_visit_count)
        return res

    def get_roots_sampled_actions(self):
        res = []
        cdef int num_children_of_root
        cdef int[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_actions = np.empty(num_children_of_root * self.agent_num, order='C', dtype='int32')
            arr = sampled_actions
            self.trees[0].get_root_sampled_actions(i, &arr[0])
            res.append(sampled_actions.reshape(num_children_of_root, self.agent_num))
        return res

    def get_roots_sampled_pred_probs(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_pred_probs = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_pred_probs
            self.trees[0].get_root_sampled_pred_probs(i, &arr[0])
            res.append(sampled_pred_probs)
        return res

    def get_roots_sampled_beta(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_beta = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_beta
            self.trees[0].get_root_sampled_beta(i, &arr[0])
            res.append(sampled_beta)
        return res

    def get_roots_sampled_beta_hat(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_beta_hat = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_beta_hat
            self.trees[0].get_root_sampled_beta_hat(i, &arr[0])
            res.append(sampled_beta_hat)
        return res

    def get_roots_sampled_priors(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_priors = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_priors
            self.trees[0].get_root_sampled_priors(i, &arr[0])
            res.append(sampled_priors)
        return res
    
    def get_roots_sampled_imp_ratio(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_imp_ratio = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_imp_ratio
            self.trees[0].get_root_sampled_imp_ratio(i, &arr[0])
            res.append(sampled_imp_ratio)
        return res

    def get_roots_sampled_pred_values(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_pred_values = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_pred_values
            self.trees[0].get_root_sampled_pred_values(i, &arr[0])
            res.append(sampled_pred_values)
        return res

    def get_roots_sampled_mcts_values(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_mcts_values = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_mcts_values
            self.trees[0].get_root_sampled_mcts_values(i, &arr[0])
            res.append(sampled_mcts_values)
        return res

    def get_roots_sampled_rewards(self):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_rewards = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_rewards
            self.trees[0].get_root_sampled_rewards(i, &arr[0])
            res.append(sampled_rewards)
        return res

    def get_roots_sampled_qvalues(self, float discount):
        res = []
        cdef int num_children_of_root
        cdef float[::1] arr
        for i in range(self.root_num):
            num_children_of_root = self.trees[0].get_num_children_of_root(i)
            sampled_qvalues = np.empty(num_children_of_root, order='C', dtype='float32')
            arr = sampled_qvalues
            self.trees[0].get_root_sampled_qvalues(i, &arr[0], discount)
            res.append(sampled_qvalues)
        return res

    def print(self):
        """
        Debug
        """
        self.trees[0].print()
