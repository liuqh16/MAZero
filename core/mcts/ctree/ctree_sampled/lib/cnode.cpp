#include "cnode.h"

#include <cmath>
#include <stack>
#include <sys/time.h>
#include <map>
#include <cstdlib>
#include <cstring>
#include <omp.h>

namespace tree
{

    CNode::CNode(float prior, float pred_prob, float beta, float beta_hat, bool is_root, float rho, float lam)
        : visit_count(0), num_children(0), hidden_state_index_x(-1),
          reward(0.), pred_value(0.),
          prior(prior), pred_prob(pred_prob), beta(beta), beta_hat(beta_hat),
          is_root(is_root),
          subtree_info(rho, lam),
          children(), children_action()
    {
        /*
        Overview:
            Initialization of CNode with prior value and root flag.
        Arguments:
            - prior: the prior value of this node.
            - is_root: whether the current node is a root node.
        */
    }

    CNode::~CNode() {}

    bool CNode::expanded()
    {
        /*
        Overview:
            Return whether the current node is expanded.
        */
        return this->num_children > 0;
    }

    float CNode::value()
    {
        /*
        Overview:
            Return the average mcts value of the current node.
        */
        if (! this->expanded())
        {
            return 0;
        }
        else
        {
            return this->subtree_info.value_estimation();
        }
    }

    float CNode::get_qsa(float discount)
    {
        /*
        Overview:
            Compute the q value of the current node.
        Arguments:
            - discount: the discount factor of reward.
        */
        return this->reward + discount * this->value();
    }

    void CNode::get_marginal_visit_count(tools::Array2D<int> logits)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            CNode *child = this->children[i];
            for (size_t j = 0; j < logits.d1; ++j)
            {
                logits(j, this->children_action[i][j]) += child->visit_count;
            }
        }
    }

    void CNode::get_marginal_priors(tools::Array2D<float> priors)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            CNode *child = this->children[i];
            for (size_t j = 0; j < priors.d1; ++j)
            {
                priors(j, this->children_action[i][j]) += child->prior;
            }
        }
    }

    void CNode::get_sampled_visit_count(int *logits)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            logits[i] = this->children[i]->visit_count;
        }
    }

    void CNode::get_sampled_pred_probs(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->pred_prob;
        }
    }

    void CNode::get_sampled_beta(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->beta;
        }
    }

    void CNode::get_sampled_beta_hat(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->beta_hat;
        }
    }

    void CNode::get_sampled_priors(float *priors)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            priors[i] = this->children[i]->prior;
        }
    }

    void CNode::get_sampled_imp_ratio(float *imp_ratio)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            imp_ratio[i] = this->children[i]->beta_hat / this->children[i]->beta * this->children[i]->pred_prob;
        }
    }

    void CNode::get_sampled_pred_values(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->pred_value;
        }
    }

    void CNode::get_sampled_mcts_values(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->value();
        }
    }

    void CNode::get_sampled_rewards(float *rewards)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            rewards[i] = this->children[i]->reward;
        }
    }

    void CNode::get_sampled_qvalues(float *values, float discount)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->get_qsa(discount);
        }
    }

    //*********************************************************

    SearchResult::SearchResult(int length_max)
    {
        search_path = (CNode **)malloc(sizeof(CNode *) * (length_max + 2));
    }
    SearchResult::~SearchResult()
    {
        free(search_path);
    }

    //*********************************************************

    CTree::CTree(int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, CNode *node_pool_ptr, unsigned int seed, float rho, float lam)
        : gen(seed), agent_num(agent_num), action_space_size(action_space_size), sampled_times(sampled_times), tot_nodes(0),
          rho(rho), lam(lam),
          node_pool_ptr(node_pool_ptr), root(node_pool_ptr), minmax_stat(tree_value_stat_delta_lb), result(simulation_num)
    {
        /*
        Overview:
            The initialization of CTree.
        */
    }

    CTree::~CTree()
    {
        for (int i = 0; i < this->tot_nodes; ++i)
        {
            this->node_pool_ptr[i].~CNode();
        }
    }

    void CTree::prepare(float reward, float value, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises)
    {
        /*
        Overview:
            Expand the root node.
        Arguments:
            - reward: the reward of the root node.
            - value: the predicted value of the root node.
            - policy_probs: the probability of the child nodes.
            - beta: the sampling distribution of the child nodes.
            - sampled_times: the number of sampled times.
        */
        new (this->root) CNode(1., 1., 1., 1., true, this->rho, this->lam);
        ++(this->tot_nodes);
        this->expand(this->root, 0, reward, value, policy_probs, beta, sampled_times, noise_eps, noises);
        this->root->visit_count += 1;
        this->root->subtree_info.update(value, 0);
    }

    void CTree::expand(CNode *node, int hidden_state_index_x, float reward, float value, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - node: pointer to the current node.
            - hidden_state_index_x: The index of latent state in the search path of the current node.
            - reward: the predicted reward from parent node to the current node.
            - value: the predicted value of the current node.
            - policy_probs: the probability of the child nodes.
            - beta: the sampling distribution of the child nodes.
            - sampled_times: the number of sampled times.
        */
        node->hidden_state_index_x = hidden_state_index_x;
        node->reward = reward;
        node->pred_value = value;

        // compute beta_hat via beta sampling
        std::map<long, float> beta_hat;
        std::map<long, std::vector<int>> action_map;
        std::vector<std::discrete_distribution<>> dists;
        dists.reserve(this->agent_num);
        for (int i = 0; i < this->agent_num; ++i)
        {
            dists.push_back(std::discrete_distribution<>(&beta(i, 0), &beta(i + 1, 0)));
        }
        for (int k = 0; k < sampled_times; ++k)
        {
            long key = 0;
            std::vector<int> sampled_action(this->agent_num, 0);
            for (int i = 0; i < this->agent_num; ++i)
            {
                sampled_action[i] = dists[i](this->gen);
                key = key * 23333 + sampled_action[i]; // use a prime constant 23333 to prevent from degeneration
            }
            beta_hat[key] += 1.0;
            action_map[key] = sampled_action;
        }

        node->num_children = beta_hat.size();
        node->children.reserve(node->num_children);
        node->children_action.reserve(node->num_children);
        // add children
        for (auto it : beta_hat)
        {
            long key = it.first;
            float count = it.second;
            std::vector<int> sampled_action = action_map[key];
            float betahat_prob = count / sampled_times;
            float beta_prob = 1.0;
            float pred_prob = 1.0;
            float prior = 1.0;
            for (int i = 0; i < this->agent_num; ++i)
            {
                beta_prob *= beta(i, sampled_action[i]);
                pred_prob *= policy_probs(i, sampled_action[i]);
                if(noise_eps > 0){
                    float p = policy_probs(i, sampled_action[i]) * (1-noise_eps) + noises(i, sampled_action[i]) * noise_eps;
                    // p ** (1/tau)  \propto  beta
                    prior *= p;
                }else{
                    prior *= policy_probs(i, sampled_action[i]);
                }
            }
            prior = prior * betahat_prob / beta_prob;
            new (this->node_pool_ptr + this->tot_nodes) CNode(prior, pred_prob, beta_prob, betahat_prob, false, this->rho, this->lam);
            ++(this->tot_nodes);
            node->children.push_back(this->node_pool_ptr + this->tot_nodes - 1);
            node->children_action.push_back(sampled_action);
        }
    }

    float CTree::ucb_score(CNode *child, float parent_q, int total_children_visit_counts, float pb_c_base, float pb_c_init, float discount)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - parent_q: parent pred q.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - discount: the discount factor of reward.
        Outputs:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0)
        {
            value_score = 0;
        }
        else
        {
            value_score = child->get_qsa(discount) - parent_q;
        }

        value_score = this->minmax_stat.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    int CTree::select_child(CNode *node, float pb_c_base, float pb_c_init, float discount, float parent_q)
    {
        /*
        Overview:
            Select the child node of the current node according to ucb scores.
        Arguments:
            - node: the current node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - discount: the discount factor of reward.
            - parent_q: parent pred q.
        Outputs:
            - child_index: the index of child to select.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        for (int child_index = 0; child_index < node->num_children; ++child_index)
        {
            CNode *child = node->children[child_index];
            float temp_score = ucb_score(child, parent_q, node->visit_count - 1, pb_c_base, pb_c_init, discount);

            if (max_score < temp_score)
            {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(child_index);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(child_index);
            }
        }

        int child_index = 0;
        if (max_index_lst.size() > 0)
        {
            auto rand_index = this->gen() % max_index_lst.size();
            child_index = max_index_lst[rand_index];
        }
        return child_index;
    }

    void CTree::select_path(float pb_c_base, float pb_c_init, float discount)
    {
        /*
        Overview:
            Search node path from the root node and store in this->result.
        Arguments:
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount: the discount factor of reward.
        */
        CNode *node = this->root;
        this->result.search_len = 0;
        this->result.search_path[this->result.search_len] = node;

        while (node->expanded())
        {
            int child_index;
            if(node->is_root && node->visit_count <= node->num_children){
                child_index = node->visit_count - 1;
            }else{
                child_index = select_child(node, pb_c_base, pb_c_init, discount, node->pred_value);
            }
            // next
            this->result.action = &(node->children_action[child_index][0]);
            node = node->children[child_index];
            this->result.search_len += 1;
            this->result.search_path[this->result.search_len] = node;
        }

        CNode *parent = this->result.search_path[this->result.search_len - 1];
        this->result.idx = (parent->hidden_state_index_x);
        this->result.leaf = node;
    }

    void CTree::back_propagate(float value, float discount)
    {
        /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - value: the value to propagate along the search path.
            - discount: the discount factor of reward.
        */
        float bootstrap_value = value;
        int path_len = this->result.search_len;

        for (int i = path_len; i >= 0; --i)
        {
            CNode *node = this->result.search_path[i];

            if (i != path_len && i != 0)
            {
                // not leaf, not root
                CNode* father = this->result.search_path[i - 1];
                this->minmax_stat.remove(node->get_qsa(discount) - father->pred_value);
            }

            node->visit_count += 1;
            node->subtree_info.update(bootstrap_value, path_len-i);

            if (i != 0)
            {
                // not root
                CNode* father = this->result.search_path[i - 1];
                this->minmax_stat.insert(node->get_qsa(discount) - father->pred_value);
            }

            bootstrap_value = node->reward + discount * bootstrap_value;
        }
    }

    void CTree::expand_and_backprop(int hidden_state_index_x, float discount, int sampled_times, float reward, float value, tools::Array2D<float> policy_prob, tools::Array2D<float> beta)
    {
        /*
        Overview:
            Expand the leaf node and update the info along the search path.
        Arguments:
            - hidden_state_index_x: The index of latent state in the search path of the current node.
            - discount: the discount factor of reward.
            - sampled_times: the number of sampled times.
            - reward: the predicted reward from parent node to the current node.
            - value: the predicted value of the current node.
            - policy_probs: the probability of the child nodes.
            - beta: the sampling distribution of the child nodes.
        */
        tools::Array2D<float> _(nullptr, 0, 0);   // placeholder
        expand(this->result.leaf, hidden_state_index_x, reward, value, policy_prob, beta, sampled_times, 0., _);
        back_propagate(value, discount);
    }

    void CTree::get_root_value(float *val)
    {
        *val = this->root->value();
    }
    void CTree::get_root_marginal_visit_count(tools::Array2D<int> logits)
    {
        root->get_marginal_visit_count(logits);
    }
    void CTree::get_root_marginal_priors(tools::Array2D<float> priors)
    {
        root->get_marginal_priors(priors);
    }
    void CTree::get_root_sampled_actions(tools::Array2D<int> actions)
    {
        tools::my_assert(root->children_action.size() == actions.d1 && root->children_action[0].size() == actions.d2,
                         "Error in `CTree::get_root_sampled_actions`: dimensions of `root->children_action` does not match that of the receiving buffer.");
        for (size_t i = 0; i < actions.d1; ++i)
            for (size_t j = 0; j < actions.d2; ++j)
                actions(i, j) = root->children_action[i][j];
    }
    void CTree::get_root_sampled_visit_count(int *logits)
    {
        root->get_sampled_visit_count(logits);
    }
    void CTree::get_root_sampled_pred_probs(float *probs)
    {
        root->get_sampled_pred_probs(probs);
    }
    void CTree::get_root_sampled_imp_ratio(float *imp_ratio)
    {
        root->get_sampled_imp_ratio(imp_ratio);
    }
    void CTree::get_root_sampled_beta(float *probs)
    {
        root->get_sampled_beta(probs);
    }
    void CTree::get_root_sampled_beta_hat(float *probs)
    {
        root->get_sampled_beta_hat(probs);
    }
    void CTree::get_root_sampled_priors(float *priors)
    {
        root->get_sampled_priors(priors);
    }
    void CTree::get_root_sampled_pred_values(float *values)
    {
        root->get_sampled_pred_values(values);
    }
    void CTree::get_root_sampled_mcts_values(float *values)
    {
        root->get_sampled_mcts_values(values);
    }
    void CTree::get_root_sampled_rewards(float *rewards)
    {
        root->get_sampled_rewards(rewards);
    }
    void CTree::get_root_sampled_qvalues(float *values, float discount)
    {
        root->get_sampled_qvalues(values, discount);
    }

    void CTree::print()
    {
        for (int i = 0; i < this->tot_nodes; ++i)
        {
            fprintf(stderr, "node %d info:\n", i);
            auto u = this->node_pool_ptr[i];
            fprintf(stderr, "\tvisit count: %d, idx: %d\n", u.visit_count, u.hidden_state_index_x);
            fprintf(stderr, "\treward: %f, prior: %f, estimate_value: %f\n", u.reward, u.prior, u.subtree_info.value_estimation());
            fprintf(stderr, "\tchildren (%d in total):\n", u.num_children);
            for (int j = 0; j < u.num_children; ++j)
            {
                fprintf(stderr, "\t\tid: %ld,\t action: ", u.children[j] - this->node_pool_ptr);
                for (auto it : u.children_action[j])
                    fprintf(stderr, "%d ", it);
                fprintf(stderr, "\n");
            }
        }
    }

    //*********************************************************

    CTree_batch::CTree_batch(int root_num, int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, unsigned int random_seed, float rho, float lam)
    {
        /*
        Overview:
            The initialization of CTree_batch.
        */
        this->root_num = root_num;
        this->agent_num = agent_num;
        this->action_space_size = action_space_size;
        this->pool_size_per_root = sampled_times * (simulation_num + 2);
        this->thread_num = 1;

        // allocate memory
        this->node_pool = (CNode *)malloc(sizeof(CNode) * this->root_num * this->pool_size_per_root);
        this->trees = (CTree *)malloc(sizeof(CTree) * this->root_num);
        tools::my_assert(this->node_pool && this->trees, "Error in `CTree_batch::CTree_batch`: `malloc` fails for `node_pool` or `trees`.");

        // init each tree
        for (int i = 0; i < this->root_num; ++i)
        {
            auto ptr_i = this->node_pool + i * this->pool_size_per_root;
            unsigned int seed_i = random_seed * 2333 + i;
            new (this->trees + i) CTree(agent_num, action_space_size, sampled_times, simulation_num, tree_value_stat_delta_lb, ptr_i, seed_i, rho, lam);
        }
    }

    CTree_batch::~CTree_batch()
    {
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].~CTree();
        }
        free(this->trees);
        free(this->node_pool);
    }

    void CTree_batch::prepare(float *rewards_buf, float *values_buf, float *policy_probs_buf, float *beta_buf, int sampled_times, float noise_eps, float* noises_buf)
    {
        /*
        Overview:
            Expand the root nodes of a batch.
        Arguments:
            - rewards_buf: batch root node rewards (root_num,).
            - values_buf: batch root node values (root_num,).
            - policy_probs_buf: batch root node probs (root_num, agent_num, action_space_size)
            - beta_buf: batch root node sampling-dists (root_num, agent_num, action_space_size)
            - sampled_times: the number of sampled times.
        */
        float *rewards = rewards_buf;
        float *values = values_buf;
        tools::Array3D<float> policy_probs(policy_probs_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> beta(beta_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> noises(noises_buf, this->root_num, this->agent_num, this->action_space_size);

        for (int i = 0; i < this->root_num; ++i)
        {
            tools::Array2D<float> prob_i(&policy_probs(i, 0, 0), policy_probs.d2, policy_probs.d3);
            tools::Array2D<float> beta_i(&beta(i, 0, 0), beta.d2, beta.d3);
            tools::Array2D<float> noise_i(&noises(i, 0, 0), noises.d2, noises.d3);
            this->trees[i].prepare(rewards[i], values[i], prob_i, beta_i, sampled_times, noise_eps, noise_i);
        }
    }

    void CTree_batch::cbatch_selection(float pb_c_base, float pb_c_init, float discount, int *idx_buf, int *idy_buf, int *act_buf)
    {
        /*
        Overview:
            Search node path from the root nodes of a batch.
        Arguments:
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount: the discount factor of reward.
            - idx_buf: buffer to store the search results of hidden_state_index(x): (root_num,).
            - idx_buf: buffer to store the search results of batch_index(y): (root_num,).
            - act_buf: buffer to store the search results of leaf node actions: (root_num, agent_num).
        */
        int *idx_arr = idx_buf;
        int *idy_arr = idy_buf;
        tools::Array2D<int> act_arr(act_buf, this->root_num, this->agent_num);

        #pragma omp parallel for num_threads(this->thread_num)
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].select_path(pb_c_base, pb_c_init, discount);
            idx_arr[i] = this->trees[i].result.idx;
            idy_arr[i] = i;
            for (int j = 0; j < this->agent_num; ++j)
                act_arr(i, j) = this->trees[i].result.action[j];
        }
    }

    void CTree_batch::cbatch_expansion_and_backup(int hidden_state_index_x, float discount, int sampled_times, float *rewards_buf, float *values_buf, float *policy_probs_buf, float *beta_buf)
    {
        /*
        Overview:
            Expand and backup the leaf nodes of a batch.
        Arguments:
            - hidden_state_index_x: The index of latent state in the search path of the current node.
            - discount: the discount factor of reward.
            - sampled_times: the number of sampled times.
            - rewards_buf: batch root node rewards (root_num,).
            - values_buf: batch root node values (root_num,).
            - policy_probs_buf: batch root node probs (root_num, agent_num, action_space_size)
            - beta_buf: batch root node sampling-dists (root_num, agent_num, action_space_size)
        */
        float *rewards = rewards_buf;
        float *values = values_buf;
        tools::Array3D<float> policy_probs(policy_probs_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> beta(beta_buf, this->root_num, this->agent_num, this->action_space_size);

        #pragma omp parallel for num_threads(this->thread_num)
        for (int i = 0; i < this->root_num; ++i)
        {
            tools::Array2D<float> prob_i(&policy_probs(i, 0, 0), policy_probs.d2, policy_probs.d3);
            tools::Array2D<float> beta_i(&beta(i, 0, 0), beta.d2, beta.d3);
            this->trees[i].expand_and_backprop(hidden_state_index_x, discount, sampled_times, rewards[i], values[i], prob_i, beta_i);
        }
    }

    void CTree_batch::get_roots_values(float *buf)
    {
        float *val = buf;
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_value(val + i);
        }
    }


    void CTree_batch::get_roots_marginal_visit_count(int *buf)
    {
        memset(buf, 0, sizeof(int) * this->root_num * this->agent_num * this->action_space_size);
        tools::Array3D<int> arr(buf, this->root_num, this->agent_num, this->action_space_size);
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_marginal_visit_count(tools::Array2D<int>(&arr(i, 0, 0), this->agent_num, this->action_space_size));
        }
    }

    void CTree_batch::get_roots_marginal_priors(float *buf)
    {
        memset(buf, 0, sizeof(float) * this->root_num * this->agent_num * this->action_space_size);
        tools::Array3D<float> arr(buf, this->root_num, this->agent_num, this->action_space_size);
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_marginal_priors(tools::Array2D<float>(&arr(i, 0, 0), this->agent_num, this->action_space_size));
        }
    }

    int CTree_batch::get_num_children_of_root(int tree_id)
    {
        return this->trees[tree_id].root->num_children;
    }

    void CTree_batch::get_root_sampled_actions(int tree_id, int *buf)
    {
        tools::Array2D<int> arr(buf, this->trees[tree_id].root->num_children, this->agent_num);
        this->trees[tree_id].get_root_sampled_actions(arr);
    }

    void CTree_batch::get_root_sampled_visit_count(int tree_id, int *buf)
    {
        memset(buf, 0, sizeof(int) * this->trees[tree_id].root->num_children);
        int *arr = buf;
        this->trees[tree_id].get_root_sampled_visit_count(arr);
    }

    void CTree_batch::get_root_sampled_pred_probs(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_pred_probs(arr);
    }

    void CTree_batch::get_root_sampled_imp_ratio(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_imp_ratio(arr);
    }

    void CTree_batch::get_root_sampled_beta(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_beta(arr);
    }

    void CTree_batch::get_root_sampled_beta_hat(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_beta_hat(arr);
    }

    void CTree_batch::get_root_sampled_priors(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_priors(arr);
    }

    void CTree_batch::get_root_sampled_rewards(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_rewards(arr);
    }

    void CTree_batch::get_root_sampled_pred_values(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_pred_values(arr);
    }

    void CTree_batch::get_root_sampled_mcts_values(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_mcts_values(arr);
    }

    void CTree_batch::get_root_sampled_qvalues(int tree_id, float *buf, float discount)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        float *arr = buf;
        this->trees[tree_id].get_root_sampled_qvalues(arr, discount);
    }

    void CTree_batch::print()
    {
        for (int i = 0; i < root_num; ++i)
        {
            fprintf(stderr, "---------- Tree %d info ----------\n", i);
            this->trees[i].print();
            fprintf(stderr, "\n");
        }
    }
}

// ===============================================================