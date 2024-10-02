#include "suprcpp.h"
#include "suprutil.h"
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

SUPRCPP::SUPRCPP(const std::string &path, const int num_betas, const bool constrained) : m_constrained(constrained), m_num_pose(0), m_num_joints(0), m_num_verts(0), m_num_betas(num_betas), m_well_formed(false) {
    using namespace torch::indexing;
    if(std::filesystem::exists(path)){

        std::map<std::string, cnpy::NpyArray> data = cnpy::npz_load(path);

        torch::Tensor kintree = totorch<int64_t>(data, "kintree_table", torch::kInt64, torch::kInt64);

        this->register_buffer("kintree_table", kintree);
        m_num_joints = static_cast<int>(kintree.sizes()[1]);
        m_num_pose = m_num_joints * 3;

        torch::Tensor v_template = toTorchFloating(data, "v_template");
        this->register_buffer("v_template", v_template);
        m_num_verts = static_cast<int>(v_template.sizes()[0]);

        torch::Tensor shapedirs = toTorchFloating(data, "shapedirs").index(
                {Slice(), Slice(), Slice(None, m_num_betas)});
        this->register_buffer("shapedirs", shapedirs);

        torch::Tensor faces = totorch<int32_t>(data, "f", torch::kInt32, torch::kInt64);
        this->register_buffer("faces", faces);

        torch::Tensor J_regressor = toTorchFloating(data, "J_regressor");
        this->register_buffer("J_regressor", J_regressor);

        torch::Tensor weights = toTorchFloating(data, "weights");
        this->register_buffer("weights", weights);

        torch::Tensor posedirs = toTorchFloating(data, "posedirs");
        this->register_buffer("posedirs", posedirs);


        m_constrained = (m_constrained && std::filesystem::exists(path.substr(0, path.find_last_of('.')) + ".json"));
        if (m_constrained) {
            std::ifstream extras(path.substr(0, path.find_last_of('.')) + ".json");
            nlohmann::json extra = nlohmann::json::parse(extras);
            if (extra.contains("axis_meta")) {
                nlohmann::json &jdata = extra["axis_meta"]["data"];
                m_num_pose = jdata["num_pose"]["data"];


                torch::Tensor indx_spherical = totorch<int32_t>(jdata, "indx_spherical", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_spherical", indx_spherical);

                torch::Tensor indx_double_hinge1 = totorch<int32_t>(jdata, "indx_double_hinge1", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_double_hinge1", indx_double_hinge1);

                torch::Tensor indx_double_hinge1_axis = totorch<int32_t>(jdata, "indx_double_hinge1_axis", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_double_hinge1_axis", indx_double_hinge1_axis);

                torch::Tensor indx_double_hinge2 = totorch<int32_t>(jdata, "indx_double_hinge2", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_double_hinge2", indx_double_hinge2);

                torch::Tensor indx_double_hinge2_axis = totorch<int32_t>(jdata, "indx_double_hinge2_axis", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_double_hinge2_axis", indx_double_hinge2_axis);

                torch::Tensor indx_hinge = totorch<int32_t>(jdata, "indx_hinge", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_hinge", indx_hinge);

                torch::Tensor indx_hinge_axis = totorch<int32_t>(jdata, "indx_hinge_axis", torch::kInt32, torch::kInt64);
                this->register_buffer("indx_hinge_axis", indx_hinge_axis);

                torch::Tensor reverse_indx = totorch<int32_t>(jdata, "reverse_indx", torch::kInt32, torch::kInt64);
                this->register_buffer("reverse_indx", reverse_indx);

                torch::Tensor axis_indx = totorch<int32_t>(jdata, "axis_indx", torch::kInt32, torch::kInt64);
                this->register_buffer("axis_indx", axis_indx);

                torch::Tensor axis = totorch<double>(jdata, "axis", torch::kFloat64, torch::kFloat32);
                this->register_buffer("axis", axis);

            } else {
                m_constrained = false;
            }
        }

        std::unordered_map<long, int> id_to_col;
        for (int i = 0; i < kintree.sizes()[1]; ++i) {
            id_to_col[kintree[1][i].item<long>()] = i;
        }
        m_parent.clear();
        m_parent.reserve(id_to_col.size() - 1);
        for (int i = 1; i < kintree.sizes()[1]; ++i) {
            m_parent.emplace_back(id_to_col[kintree[0][i].item<long>()]);
        }
        torch::Tensor parent = torch::from_blob(m_parent.data(), {static_cast<int64_t>(m_parent.size())}, torch::kInt64);
        this->register_buffer("parent", parent);

        m_well_formed=true;
    } else {
        std::cerr << "error loading the model at " << path << std::endl;
    }
}

void SUPRCPP::display_info(std::ostream &stream) const {
    if (m_constrained) {
        stream << "Kinematic Tree: Constrained\n";
    } else {
        stream << "Kinematic Tree: Un-Constrained\n";
    }
    stream << "Number of Pose Parameters: " << m_num_pose << "\n";
    stream << "Number of Joints: " << m_num_joints << "\n";
    stream << "'Number of Vertices: " << m_num_verts << std::endl;
}

SUPRInferred SUPRCPP::forward(torch::Tensor &pose, torch::Tensor &betas, torch::Tensor &trans) const {
    if (!m_well_formed){
        std::cerr<<"SUPR wasn't loaded correctly"<<std::endl;
        return {};
    }
    using namespace torch::indexing;

    torch::OrderedDict<std::string, torch::Tensor> buffers = this->named_buffers();

    torch::Device d = pose.device();
    int batch_size = static_cast<int>(pose.size(0));
    torch::Tensor v_template = buffers["v_template"].index({None});
    torch::Tensor shapedirs = buffers["shapedirs"].view({-1, m_num_betas}).index({None, Slice()}).expand({batch_size, -1, -1});
    torch::Tensor beta = betas.index({Slice(), Slice(), None});

    torch::Tensor v_shaped = torch::matmul(shapedirs, beta).view({-1, m_num_verts, 3}) + v_template;

    //Computing the shape correctives
    torch::Tensor pad_v_shaped = torch::cat({v_shaped.view({-1, m_num_verts * 3}), torch::ones({batch_size, 1}).to(d)}, 1);
    torch::Tensor &J_regressor = buffers["J_regressor"];
    torch::Tensor J = torch::einsum("ji,ai->aj", {J_regressor, pad_v_shaped}).view({-1, m_num_joints, 3});

    // Replacing that with the Fast Rot Utils Module....
    torch::Tensor torch_feat;
    torch::Tensor R;
    if (m_constrained) {
        std::tie(torch_feat, R) = torch_fast_rotutils(pose, buffers["indx_spherical"], buffers["indx_double_hinge1"], buffers["indx_double_hinge1_axis"], buffers["indx_double_hinge2"], buffers["indx_double_hinge2_axis"], buffers["indx_hinge"], buffers["indx_hinge_axis"], buffers["reverse_indx"], buffers["axis_indx"], buffers["axis"]);
        torch_feat = torch_feat.view({batch_size, -1});
    } else {
        torch_feat = quat_feat(pose.view({-1, m_num_joints, 3})).view({batch_size, -1});
        R = rodrigues(pose.view({-1, m_num_joints, 3})).view({batch_size, m_num_joints, 3, 3});//check m_num_joints if equal
    }

    torch::Tensor posedirs = buffers["posedirs"].view({m_num_verts, 3, -1});
    //Computing the Pose-Depedent Corrective BlendShapes
    torch::Tensor v_posed = v_shaped + torch::einsum("ijk,lk->lij", {posedirs, torch_feat});

    torch::Tensor J_ = J.clone();
    J_.index({Slice(), Slice(1, None), Slice()}) -= J_.index({Slice(), buffers["parent"], Slice()});
    torch::Tensor G_ = torch::cat({R, J_.index({Slice(), Slice(), Slice(), None})}, -1);
    torch::Tensor pad_row = torch::tensor({0, 0, 0, 1}, torch::kFloat32).to(d).view({1, 1, 1, 4}).expand({batch_size, m_num_joints, -1, -1});
    G_ = torch::cat({G_, pad_row}, 2);
    std::vector<torch::Tensor> G_chain{G_.index({Slice(), 0})};
    for (int i = 1; i < m_num_joints; ++i) {
        G_chain.push_back(torch::matmul(G_chain[m_parent.at(i - 1)], G_.index({Slice(), i, Slice(), Slice()})));
    }
    torch::Tensor G = torch::stack(G_chain, 1);

    torch::Tensor rest = torch::cat({J, torch::zeros({batch_size, m_num_joints, 1}).to(d)}, 2).view({batch_size, m_num_joints, 4, 1});
    torch::Tensor zeros = torch::zeros({batch_size, m_num_joints, 4, 3}).to(d);
    rest = torch::cat({zeros, rest}, -1);
    rest = torch::matmul(G, rest);
    G = G - rest;
    torch::Tensor T = torch::matmul(buffers["weights"], G.permute({1, 0, 2, 3}).contiguous().view({m_num_joints, -1})).view({m_num_verts, batch_size, 4, 4}).transpose(0, 1);
    torch::Tensor rest_shape_h = torch::cat({v_posed, torch::ones_like(v_posed).index({Slice(), Slice(), Slice(0, 1)})}, -1);
    torch::Tensor v_final = torch::matmul(T, rest_shape_h.index({Slice(), Slice(), Slice(), None})).index({Slice(), Slice(), Slice(None, 3), 0});
    v_final = v_final + trans.index({Slice(), None, Slice()});
    torch::Tensor J_transformed = G.index({Slice(), Slice(), Slice(None, 3), 3}) + trans.index({Slice(), None, Slice()});

    return {v_final, v_posed, v_shaped, J_transformed, buffers["faces"]};
}

int SUPRCPP::getNumPose() const {
    return m_num_pose;
}

std::vector<long> SUPRCPP::getParentVec() const {
    return m_parent;
}
