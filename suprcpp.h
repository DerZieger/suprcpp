#pragma once
#include <torch/types.h>
#include <torch/nn.h>
#include <unordered_map>

struct SUPRInferred{
    torch::Tensor v_final;
    torch::Tensor v_posed;
    torch::Tensor v_shaped;
    torch::Tensor J_transformed;
    torch::Tensor f;
};



class SUPRCPP : public torch::nn::Module{
public:
    SUPRCPP(const std::string &path, const int num_betas = 10, const bool constrained = true);

    void display_info(std::ostream &stream=std::cout) const;

    SUPRInferred forward(torch::Tensor &pose, torch::Tensor &betas, torch::Tensor &trans) const;

    int getNumPose() const;

    std::vector<long> getParentVec() const;

private:
    bool m_constrained;
    int m_num_pose;
    int m_num_joints;
    int m_num_verts;
    int m_num_betas;
    std::vector<long> m_parent;
    bool m_well_formed;
};
