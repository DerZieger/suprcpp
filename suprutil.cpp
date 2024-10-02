#include "suprutil.h"


torch::Tensor toTorchFloating(cnpy::npz_t &n, const std::string &s) {
    if (n[s].word_size == 4) {
        return totorch<float>(n, s, torch::kFloat32, torch::kFloat32);
    } else {
        return totorch<double>(n, s, torch::kFloat64, torch::kFloat32);
    }
}

torch::Tensor toTorchInt(cnpy::npz_t &n, const std::string &s, torch::ScalarType type) {
    if (n[s].word_size == 4) {
        return totorch<int32_t>(n, s, torch::kInt32, type);
    } else {
        return totorch<int64_t>(n, s, torch::kInt64, type);
    }
}


std::pair<torch::Tensor, torch::Tensor> torch_fast_rotutils(const torch::Tensor &pose, const torch::Tensor &indx_spherical, const torch::Tensor &indx_double_hinge1, const torch::Tensor &indx_double_hinge1_axis, const torch::Tensor &indx_double_hinge2, const torch::Tensor &indx_double_hinge2_axis, const torch::Tensor &indx_hinge, const torch::Tensor &indx_hinge_axis, const torch::Tensor &reverse_indx, const torch::Tensor &axis_indx, const torch::Tensor &axis2) {
    /*
    Compute the kinematic tree joint rotations matrices and the corresponding normalized
    quaternion features

    returns: rotation matrices, normalized quaternion features
    */
    using namespace torch::indexing;
    const torch::Tensor axis = axis2.index({Slice(), axis_indx, Slice()});

    const torch::Tensor spherical_pose = pose.index({Slice(), indx_spherical.flatten()});

    const torch::Tensor hinge_pose = pose.index({Slice(), indx_hinge.flatten()});
    const torch::Tensor hinge_axis_pose = axis.index({Slice(), indx_hinge_axis});

    const torch::Tensor hinge_pose1 = pose.index({Slice(), indx_double_hinge1});
    const torch::Tensor hinge_axis_pose1 = axis.index({Slice(), indx_double_hinge1_axis});

    const torch::Tensor hinge_pose2 = pose.index({Slice(), indx_double_hinge2});
    const torch::Tensor hinge_axis_pose2 = axis.index({Slice(), indx_double_hinge2_axis});

    std::vector<torch::Tensor> list_rotat_mat;
    std::vector<torch::Tensor> list_feat;

    int num_joints = static_cast<int>(indx_spherical.size(0));
    list_rotat_mat.push_back(rodrigues(spherical_pose.view({-1, num_joints, 3})));
    torch::Tensor quaternion_axis = quat_feat(spherical_pose.view({-1, num_joints, 3}));
    list_feat.push_back(quaternion_axis);

    torch::Tensor rot_mat = torch_compute_rot_hinge(hinge_pose.index({Slice(), Slice(), None}), hinge_axis_pose);

    list_rotat_mat.push_back(rot_mat);
    torch::Tensor axis_angle = hinge_axis_pose * hinge_pose.index({Slice(), Slice(), None});

    quaternion_axis = quat_feat(axis_angle);
    list_feat.push_back(quaternion_axis);

    torch::Tensor joint_pose1 = hinge_pose1.index({Slice(), Slice(), None});
    torch::Tensor joint_pose2 = hinge_pose2.index({Slice(), Slice(), None});

    const torch::Tensor &axis_pose1 = hinge_axis_pose1;
    const torch::Tensor &axis_pose2 = hinge_axis_pose2;

    torch::Tensor axis_angle1 = axis_pose1 * joint_pose1;


    torch::Tensor rot_mat1 = torch_compute_rot_hinge(joint_pose1, axis_pose1);
    int num_joints1 = static_cast<int>(indx_double_hinge1.size(0));
    torch::Tensor quat1 = quat_feat(axis_angle1);

    torch::Tensor axis_angle2 = axis_pose2 * joint_pose2;
    torch::Tensor rot_mat2 = torch_compute_rot_hinge(joint_pose2, axis_pose2);
    int num_joints2 = static_cast<int>(indx_double_hinge2.size(0));
    torch::Tensor quat2 = quat_feat(axis_angle2);

    torch::Tensor quat_multiply = torch_quaternion_multiply(torch::reshape(quat1, {-1, num_joints1, 4}), torch::reshape(quat2, {-1, num_joints2, 4}));
    rot_mat = torch::matmul(rot_mat1, rot_mat2);

    list_feat.push_back(quat_multiply);
    list_rotat_mat.push_back(rot_mat);

    torch::Tensor torch_rot_mat = torch::cat(list_rotat_mat, 1);
    torch::Tensor torch_feat = torch::cat(list_feat, 1);

    torch_feat = torch_feat.index({Slice(), reverse_indx, Slice()});
    torch_rot_mat = torch_rot_mat.index({Slice(), reverse_indx, Slice(), Slice()});
    return {torch_feat, torch_rot_mat};
}


torch::Tensor quat_feat(const torch::Tensor &theta) {
    // python implementation variable says l1, but does l2 norm
    torch::Tensor l1norm = torch::linalg_norm(theta + 1e-8, 2, 2);
    torch::Tensor angle = l1norm.unsqueeze(-1);
    torch::Tensor normalized = torch::div(theta, angle);
    angle = 0.5 * angle;
    torch::Tensor v_cos = torch::cos(angle);
    torch::Tensor v_sin = torch::sin(angle);
    torch::Tensor quat = torch::cat({v_sin * normalized, v_cos - 1}, 2);
    return quat;
}

torch::Tensor rodrigues(const torch::Tensor &theta) {
    // python implementation variable says l1, but does l2 norm
    torch::Tensor l1norm = torch::norm(theta + 1e-8, 2, 2);
    torch::Tensor angle = torch::unsqueeze(l1norm, -1);
    torch::Tensor normalized = torch::div(theta, angle);
    angle = angle * 0.5;
    torch::Tensor v_cos = torch::cos(angle);
    torch::Tensor v_sin = torch::sin(angle);
    torch::Tensor quat = torch::cat({v_cos, v_sin * normalized}, 2);

    return quat2mat(quat);
}

torch::Tensor quat2mat(const torch::Tensor &quat) {
    torch::Tensor norm_quat = quat;
    norm_quat = norm_quat / torch::norm(norm_quat, 2, 2, true);
    using namespace torch::indexing;

    torch::Tensor w = norm_quat.index({Slice(), Slice(), 0});
    torch::Tensor x = norm_quat.index({Slice(), Slice(), 1});
    torch::Tensor y = norm_quat.index({Slice(), Slice(), 2});
    torch::Tensor z = norm_quat.index({Slice(), Slice(), 3});
    int64_t B = quat.size(0);
    torch::Tensor w2 = torch::square(w);
    torch::Tensor x2 = torch::square(x);
    torch::Tensor y2 = torch::square(y);
    torch::Tensor z2 = torch::square(z);
    torch::Tensor wx = w * x;
    torch::Tensor wy = w * y;
    torch::Tensor wz = w * z;
    torch::Tensor xy = x * y;
    torch::Tensor xz = x * z;
    torch::Tensor yz = y * z;

    torch::Tensor rotMat = torch::stack({w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2}, 2).view({B, -1, 3, 3});
    return rotMat;
}

torch::Tensor torch_compute_rot_hinge(const torch::Tensor &pose, const torch::Tensor &axis) {
    using namespace torch::indexing;
    torch::Tensor cos_feat = torch::cos(pose);
    torch::Tensor sin_feat = torch::sin(pose);
    torch::Tensor ax = axis.index({Slice(), Slice(), 0}).index({Slice(), Slice(), None});
    torch::Tensor ay = axis.index({Slice(), Slice(), 1}).index({Slice(), Slice(), None});
    torch::Tensor az = axis.index({Slice(), Slice(), 2}).index({Slice(), Slice(), None});

    torch::Tensor row1 = torch::cat({torch::square(ax) + cos_feat * (1 - torch::square(ax)), ax * ay * (1 - cos_feat) + az * sin_feat, ax * az * (1 - cos_feat) - ay * sin_feat}, 2);
    torch::Tensor row2 = torch::cat({ax * ay * (1 - cos_feat) - az * sin_feat, torch::square(ay) + cos_feat * (1 - torch::square(ay)), ay * az * (1 - cos_feat) + ax * sin_feat}, 2);
    torch::Tensor row3 = torch::cat({ax * az * (1 - cos_feat) + ay * sin_feat, ay * az * (1 - cos_feat) - ax * sin_feat, torch::square(az) + cos_feat * (1 - torch::square(az))}, 2);
    torch::Tensor rot_mat = torch::stack({row1, row2, row3}, -1);
    rot_mat = rot_mat.permute({0, 1, 3, 2});
    return rot_mat;
}

torch::Tensor torch_quaternion_multiply(const torch::Tensor &q1, const torch::Tensor &q2) {
    using namespace torch::indexing;
    torch::Tensor x0 = q1.index({Slice(), Slice(), 0});
    torch::Tensor y0 = q1.index({Slice(), Slice(), 1});
    torch::Tensor z0 = q1.index({Slice(), Slice(), 2});
    torch::Tensor w0 = q1.index({Slice(), Slice(), 3});
    torch::Tensor x1 = q2.index({Slice(), Slice(), 0});
    torch::Tensor y1 = q2.index({Slice(), Slice(), 1});
    torch::Tensor z1 = q2.index({Slice(), Slice(), 2});
    torch::Tensor w1 = q2.index({Slice(), Slice(), 3});
    torch::Tensor xr = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0;
    torch::Tensor yr = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0;
    torch::Tensor zr = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0;
    torch::Tensor wr = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0;
    torch::Tensor quat_mult = torch::stack({xr, yr, zr, wr}, 2);
    return quat_mult;
}