# distutils: language=c++
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../common_lib/utils.cpp":
    pass

cdef extern from "lib/cnode.cpp":
    pass

cdef extern from "lib/cnode.h" namespace "tree":

    cdef cppclass CTree_batch:
        CTree_batch(int root_num, int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, unsigned int random_seed, float rho, float lam) except +

        void prepare(float* rewards, float* values, float* policy_probs, float* beta, int sampled_times, float noise_eps, float* noises) except +

        void cbatch_selection(float pb_c_base, float pb_c_init, float discount, int* idx_ptr, int* idy_ptr, int* act_ptr) except +

        void cbatch_expansion_and_backup(int hidden_state_index_x, float discount, int sampled_times, float* rewards, float* values, float* policy_probs, float* beta) except +

        void get_roots_values(float* p)                 # shape = (batch_size, )
        void get_roots_marginal_visit_count(int* p)     # shape = (batch_size, num_agents, action_space_size)
        void get_roots_marginal_priors(float* p)        # shape = (batch_size, num_agents, action_space_size)

        int get_num_children_of_root(int tree_id)                               # return deg(root)
        void get_root_sampled_actions(int tree_id, int* p)                      # shape = (deg(root), num_agents)
        void get_root_sampled_visit_count(int tree_id, int* p)                  # shape = (deg(root), )
        void get_root_sampled_pred_probs(int tree_id, float* p)                 # shape = (deg(root), )
        void get_root_sampled_beta(int tree_id, float* p)                       # shape = (deg(root), )
        void get_root_sampled_beta_hat(int tree_id, float* p)                   # shape = (deg(root), )
        void get_root_sampled_priors(int tree_id, float* p)                     # shape = (deg(root), )
        void get_root_sampled_imp_ratio(int tree_id, float* p)                  # shape = (deg(root), )
        void get_root_sampled_pred_values(int tree_id, float* p)                # shape = (deg(root), )
        void get_root_sampled_mcts_values(int tree_id, float* p)                # shape = (deg(root), )
        void get_root_sampled_rewards(int tree_id, float* p)                    # shape = (deg(root), )
        void get_root_sampled_qvalues(int tree_id, float* p, float discount)    # shape = (deg(root), )

        void print()          # debug
