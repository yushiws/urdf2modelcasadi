#include "robot_dynamics.hpp"

#include <rbdl/addons/urdfreader/urdfreader.h>
#include <rbdl/rbdl.h>
#include <rbdl/rbdl_utils.h>

using namespace RigidBodyDynamics;

namespace robot_dynamics {

RobotDynamics::RobotDynamics(const std::string &urdf_path,
                             const std::vector<std::string> &end_effector_names)
    : model_(new RigidBodyDynamics::Model) {
    RigidBodyDynamics::Addons::URDFReadFromFile(urdf_path.c_str(), model_.get(),
                                                true, false);
    for (const std::string &name : end_effector_names) {
        ee_ids.push_back(model_->GetBodyId(name.c_str()));
        if (ee_ids.back() == std::numeric_limits<unsigned int>::max())
            std::cout << "Failed to get end effector with name '" << name
                      << "'." << std::endl;
    }

    ee_cnt_ = ee_ids.size();
    q_size_ = model_->q_size;
    qdot_size_ = model_->qdot_size;
    if (q_size_ != qdot_size_ + 1)
        std::cout << "This model has more than one spherical joint. "
                     "This is not supported."
                  << std::endl;

    q_ = Eigen::VectorXd::Zero(q_size_);
    qdot_ = Eigen::VectorXd::Zero(qdot_size_);
    qddot_ = Eigen::VectorXd::Zero(qdot_size_);

    ee_pos_.resize(ee_cnt_, Eigen::Vector3d::Zero());
    ee_ori_.resize(ee_cnt_, Eigen::Quaterniond::Identity());

    ee_jacobian_.resize(ee_cnt_, Eigen::MatrixXd::Zero(6, qdot_size_));
    ee_dj_dq_.resize(ee_cnt_, Eigen::Vector6d::Zero());

    com_pos_.setZero();
    com_vel_.setZero();
    lin_momentum_.setZero();
    ang_momentum_.setZero();

    mass_matrix_ = Eigen::MatrixXd::Zero(qdot_size_, qdot_size_);
    grav_ = Eigen::VectorXd::Zero(qdot_size_);
    cori_ = Eigen::VectorXd::Zero(qdot_size_);

    cmm_ = Eigen::MatrixXd::Zero(6, qdot_size_);
    dcmm_dq_ = Eigen::Vector6d::Zero();

    // just for total_mass_
    Math::Vector3d com_pos;
    Utils::CalcCenterOfMass(*model_, q_, qdot_, nullptr, total_mass_, com_pos);

    Reset();
}

std::vector<std::string> RobotDynamics::GetLinkNameList() const {
    std::vector<std::string> names;

    for (unsigned int i = 0; i < model_->mBodies.size(); i++) {
        names.emplace_back(model_->GetBodyName(i));
    }
    return names;
}

void RobotDynamics::ComputeKinematics() {
    if (!base_pos_ori_loaded_ || !joint_pos_loaded_)
        std::cout << "Position level kinematics requires base & joint pos."
                  << std::endl;
    if (base_vel_loaded_ && joint_vel_loaded_)
        ComputeVelKin();
    else {
        UpdateKinematics(*model_, q_, Eigen::VectorXd::Zero(qdot_size_),
                         Eigen::VectorXd::Zero(qdot_size_));
        kinematics_calculated_ = true;
    }
}

void RobotDynamics::ComputeEEPosOri() {
    if (!kinematics_calculated_)
        ComputeKinematics();
    const Eigen::Vector3d point(0., 0., 0.);
    for (int i = 0; i < ee_cnt_; i++) {
        ee_pos_[i] =
            CalcBodyToBaseCoordinates(*model_, q_, ee_ids[i], point, false);
        ee_ori_[i] =
            CalcBodyWorldOrientation(*model_, q_, ee_ids[i], false).transpose();
    }
    ee_pos_ori_calculated_ = true;
}

void RobotDynamics::ComputeJacobian() {
    if (!kinematics_calculated_)
        ComputeKinematics();
    const Eigen::Vector3d point(0., 0., 0.);
    for (int i = 0; i < ee_cnt_; i++) {
        ee_jacobian_[i].setZero();
        CalcPointJacobian6D(*model_, q_, ee_ids[i], point, ee_jacobian_[i],
                            false);
    }
    jacobian_calculated_ = true;
}

void RobotDynamics::ComputeVelKin() {
    if (!base_pos_ori_loaded_ || !joint_pos_loaded_)
        std::cout << "Velocity level kinematics requires base & joint pos."
                  << std::endl;
    if (!base_vel_loaded_ || !joint_vel_loaded_)
        std::cout << "Velocity level kinematics requires base & joint vel."
                  << std::endl;
    UpdateKinematics(*model_, q_, qdot_, Eigen::VectorXd::Zero(qdot_size_));
    kinematics_calculated_ = true;
    kin_vel_calculated_ = true;
}

void RobotDynamics::ComputeJdotQdot() {
    if (!kin_vel_calculated_)
        ComputeVelKin();
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(qdot_size_);
    const Eigen::VectorXd qddot_zero = Eigen::VectorXd::Zero(qdot_size_);
    const Eigen::Vector3d point(0., 0., 0.);
    // CalcPointAcceleration6D returns xddot = J * qddot + Jdot * qdot
    // let qddot=0, we get Jdot * qdot
    for (int i = 0; i < ee_cnt_; i++) {
        ee_dj_dq_[i] = CalcPointAcceleration6D(*model_, q_, qdot_, qddot_zero,
                                               ee_ids[i], point, false);
    }
    dj_dq_calculated_ = true;
}

void RobotDynamics::ComputeCoMState() {
    Math::Vector3d com_pos;
    Math::Vector3d com_vel;
    Math::Vector3d ang_momentum;
    double m;
    Utils::CalcCenterOfMass(*model_, q_, qdot_, nullptr, m, com_pos, &com_vel,
                            nullptr, &ang_momentum, nullptr, true);
    com_pos_ = com_pos;
    com_vel_ = com_vel;
    com_state_calculated_ = true;
    lin_momentum_ = total_mass_ * com_vel_;
    ang_momentum_ = ang_momentum;
}

void RobotDynamics::ComputeDynamics() {
    if (!kin_vel_calculated_)
        ComputeVelKin();
    mass_matrix_.setZero();
    CompositeRigidBodyAlgorithm(*model_, q_, mass_matrix_, false);
    Eigen::VectorXd cori_and_grav = Eigen::VectorXd::Zero(qdot_size_);
    NonlinearEffects(*model_, q_, qdot_, cori_and_grav);
    grav_.setZero();
    const Eigen::VectorXd zero_qdot = Eigen::VectorXd::Zero(qdot_size_);
    NonlinearEffects(*model_, q_, zero_qdot, grav_);
    cori_ = cori_and_grav - grav_;
    dynamics_calculated_ = true;
}

void RobotDynamics::ComputeCentroidDynamics() {
    if (!dynamics_calculated_)
        ComputeDynamics();
    if (!com_state_calculated_)
        ComputeCoMState();

    Eigen::MatrixXd a_base = mass_matrix_.topRows<6>();
    // different from default special vectors who express
    // rigid body velocity as [rot_vel, lin_vel],
    // RBDL express the floating base's velocity as [v_b, w_b]
    // this results in a swap in a_base
    a_base.topRows<3>().swap(a_base.bottomRows<3>());
    // spatial force transform matrix from base to CoM
    // wrench_com = xform_com * wrench_base, wrench = [torque, force]
    Eigen::Matrix<double, 6, 6> xform_com =
        Eigen::Matrix<double, 6, 6>::Identity();
    // the transform matrix needs to transform external torque to CoM frame
    xform_com.topRightCorner<3, 3>() =
        Eigen::VecToSkewSymmetricMat(q_.head<3>() - com_pos_);
    // w_b is expressed in torso frame
    // so torque has to be transformed to fixed frame
    const Eigen::Quaterniond ori(q_(q_size_ - 1), q_(3), q_(4), q_(5));
    xform_com.topLeftCorner<3, 3>() = ori.toRotationMatrix();
    cmm_ = xform_com * a_base;
    Eigen::Vector6d spatial_coriolis;
    // do the same swap for coriolis
    spatial_coriolis.head<3>() = cori_.segment<3>(3);
    spatial_coriolis.tail<3>() = cori_.segment<3>(0);
    dcmm_dq_ = xform_com * spatial_coriolis;
}

void RobotDynamics::ComputeWrench() {
    if (!base_acc_loaded_ || !joint_acc_loaded_)
        std::cout << "Centroid Dynamics requires base & joint acc."
                  << std::endl;
    if (!cen_dyn_calculated_)
        ComputeCentroidDynamics();
    fg_ = cmm_ * qddot_ + dcmm_dq_;
    wrench_calculated_ = true;
}

void RobotDynamics::OutOfRangeError(int idx) const {
    std::cout << "Index '" << idx << "' is out of range, max: " << ee_cnt_
              << ". " << std::endl;
}

void RobotDynamics::NotComputedError(const char *item) const {
    std::cout << "Item " << item << " is not computed." << std::endl;
}

}  // namespace robot_dynamics