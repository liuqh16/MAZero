#ifndef CNODE_H
#define CNODE_H

#include "../../common_lib/utils.h"

#include <vector>
#include <random>

namespace tree
{
    struct CNode
    {
        int visit_count, num_children, hidden_state_index_x;
        float reward, pred_value, prior, pred_prob, beta, beta_hat;
        bool is_root;
        tools::SubTreeValueSet subtree_info;
        std::vector<CNode *> children;                 // shape = (num_children, )
        std::vector<std::vector<int>> children_action; // shape = (num_children, num_agents)

        // WARNING: num_children, hidden_state_index_x, children_action, reward, pred_value should only be modified once in `CTree::expand`

        CNode(float prior, float pred_prob, float beta, float beta_hat, bool is_root, float rho, float lam);
        ~CNode();

        bool expanded();
        float value();
        float get_qsa(float discount);
        float get_mean_q(float parent_q, float discount);

        // shape = (agent_num, action_space_size)
        void get_marginal_visit_count(tools::Array2D<int>); // marginal visit_count of sampled actions
        void get_marginal_priors(tools::Array2D<float>);    // marginal node prior of sampled actions

        // shape = (num_children,)
        void get_sampled_visit_count(int *);                     // visit_count of sampled actions
        void get_sampled_pred_probs(float *);                    // pred_probs of model prediction
        void get_sampled_beta(float *);                          // beta of action sampling
        void get_sampled_beta_hat(float *);                      // \hat{beta} of action sampling
        void get_sampled_priors(float *);                        // priors
        void get_sampled_imp_ratio(float *);                     // importance ratio (beta_hat / beta * pred_prob)
        void get_sampled_pred_values(float *);                   // pred_values, return 0 if not visited
        void get_sampled_mcts_values(float *);                   // mcts_values, return 0 if not visited
        void get_sampled_rewards(float *);                       // reward of child nodes
        void get_sampled_qvalues(float *values, float discount); // qvalue = reward + discount * mcts_value
    };

    struct SearchResult
    {
        int idx;             // index_x of parent node
        int search_len;      // path depth ( search_len + 1 = #nodes on path )
        int *action;         // action selected by parent node
        CNode *leaf;         // leaf node to be expanded
        CNode **search_path; // path[0] == root, path[search_len] == leaf

        SearchResult(int length_max);
        ~SearchResult();
    };

    struct CTree
    {
    private:
        std::mt19937 gen; // random number generator
    public:
        int agent_num, action_space_size, sampled_times, tot_nodes;
        float rho, lam;
        CNode *node_pool_ptr;
        CNode *root;
        tools::CMinMaxStats minmax_stat; // maintaining qsa for all nodes in the tree except the root.
        SearchResult result;

        CTree(int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, CNode *node_pool_ptr, unsigned int random_seed, float rho, float lam);
        ~CTree();

        void prepare(float reward, float value, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises);
        void expand(CNode *node, int hidden_state_index_x, float reward, float value, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises);

        float ucb_score(CNode *child, float parent_mean_q, int total_children_visit_counts, float pb_c_base, float pb_c_init, float discount);
        int select_child(CNode *node, float pb_c_base, float pb_c_init, float discount, float mean_q);
        void select_path(float pb_c_base, float pb_c_init, float discount);

        void back_propagate(float value, float discount);
        void expand_and_backprop(int hidden_state_index_x, float discount, int sampled_times, float reward, float value, tools::Array2D<float> policy_prob, tools::Array2D<float> beta);

        void get_root_value(float *);

        // shape = (agent_num, action_space_size)
        void get_root_marginal_visit_count(tools::Array2D<int>);
        void get_root_marginal_priors(tools::Array2D<float>);

        // shape = ((deg(root), agent_num)
        void get_root_sampled_actions(tools::Array2D<int>);
        // shape = ((deg(root),)
        void get_root_sampled_visit_count(int *);
        void get_root_sampled_pred_probs(float *);
        void get_root_sampled_beta(float *);
        void get_root_sampled_beta_hat(float *);
        void get_root_sampled_priors(float *);
        void get_root_sampled_imp_ratio(float *);
        void get_root_sampled_pred_values(float *);
        void get_root_sampled_mcts_values(float *);
        void get_root_sampled_rewards(float *);
        void get_root_sampled_qvalues(float *values, float discount);

        void print(); // debug
    };

    struct CTree_batch
    {
        int root_num, pool_size_per_root, agent_num, action_space_size, thread_num;
        CTree *trees;
        CNode *node_pool;

        CTree_batch(int root_num, int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, unsigned int random_seed, float rho, float lam);
        ~CTree_batch();

        void prepare(float *rewards, float *values, float *policy_probs, float *beta, int sampled_times, float noise_eps, float* noises);

        void cbatch_selection(float pb_c_base, float pb_c_init, float discount, int *idx_buf, int *idy_buf, int *act_buf);

        void cbatch_expansion_and_backup(int hidden_state_index_x, float discount, int sampled_times, float *rewards, float *values, float *policy_probs, float *beta);

        void get_roots_values(float *buf); // shape = (root_num, )

        void get_roots_marginal_visit_count(int *buf); // shape = (root_num, agent_num, action_space_size)
        void get_roots_marginal_priors(float *buf);    // shape = (root_num, agent_num, action_space_size)

        int get_num_children_of_root(int tree_id);
        void get_root_sampled_actions(int tree_id, int *buf);                   // shape = (deg(root), agent_num)
        void get_root_sampled_visit_count(int tree_id, int *buf);               // shape = (deg(root), )
        void get_root_sampled_pred_probs(int tree_id, float *buf);              // shape = (deg(root), )
        void get_root_sampled_beta(int tree_id, float *buf);                    // shape = (deg(root), )
        void get_root_sampled_beta_hat(int tree_id, float *buf);                // shape = (deg(root), )
        void get_root_sampled_priors(int tree_id, float *buf);                  // shape = (deg(root), )
        void get_root_sampled_imp_ratio(int tree_id, float *buf);               // shape = (deg(root), )
        void get_root_sampled_pred_values(int tree_id, float *buf);             // shape = (deg(root), )
        void get_root_sampled_mcts_values(int tree_id, float *buf);             // shape = (deg(root), )
        void get_root_sampled_rewards(int tree_id, float *buf);                 // shape = (deg(root), )
        void get_root_sampled_qvalues(int tree_id, float *buf, float discount); // shape = (deg(root), )

        void print(); // debug
    };
}

#endif