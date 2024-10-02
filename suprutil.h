#pragma  once

#include "cnpy.h"
#include "torch/torch.h"
#include <nlohmann/json.hpp>

//converts numpy data file into at::tensors for torch
template<typename T>
static torch::Tensor totorch(cnpy::npz_t &n, const std::string &s, torch::ScalarType type,
                             torch::ScalarType type2) {
    cnpy::NpyArray v = n[s];
    std::vector<int64_t> shape;
    for (unsigned long i: v.shape) shape.push_back(static_cast<long>(i));
    T *dat = v.data<T>();
    torch::Tensor ret = torch::from_blob(dat, shape, type).to(type2).detach().clone();
    return ret;
}

//converts json data file into at::tensors for torch
template<typename T>
static torch::Tensor totorch(nlohmann::json &j, const std::string &s, torch::ScalarType type,
                             torch::ScalarType type2) {
    nlohmann::json jdata = j[s];
    std::vector<int64_t> shape = jdata["shape"].get<std::vector<int64_t>>();
    std::vector<T> data = jdata["data"].get<std::vector<T>>();
    T *dat = data.data();
    torch::Tensor ret = torch::from_blob(dat, shape, type).to(type2).detach().clone();
    return ret;
}


torch::Tensor toTorchFloating(cnpy::npz_t &n, const std::string &s);

torch::Tensor toTorchInt(cnpy::npz_t &n, const std::string &s, torch::ScalarType type);

std::pair<torch::Tensor, torch::Tensor> torch_fast_rotutils(const torch::Tensor &pose, const torch::Tensor &indx_spherical, const torch::Tensor &indx_double_hinge1, const torch::Tensor &indx_double_hinge1_axis, const torch::Tensor &indx_double_hinge2, const torch::Tensor &indx_double_hinge2_axis, const torch::Tensor &indx_hinge, const torch::Tensor &indx_hinge_axis, const torch::Tensor &reverse_indx, const torch::Tensor &axis_indx , const torch::Tensor &axis);

torch::Tensor quat_feat(const torch::Tensor &theta);

torch::Tensor rodrigues(const torch::Tensor &theta);

torch::Tensor quat2mat(const torch::Tensor &quat);

torch::Tensor torch_compute_rot_hinge(const torch::Tensor &pose, const torch::Tensor &axis);

torch::Tensor torch_quaternion_multiply(const torch::Tensor &q1, const torch::Tensor &q2);