#include <casadi/casadi.hpp>

#include "model_interface.hpp"
#include "robot_dynamics.hpp"

#define ARM_Q 7

namespace Eigen {
typedef Matrix<casadi::SXElem, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Quaternion<casadi::SXElem> Quaternions;
}  // namespace Eigen

int main() {
    // ---------------------------------------------------------------------
    // Create a model based on a URDF file
    // ---------------------------------------------------------------------
    std::string urdf_filename =
        "../urdf2model/models/mabimobile/mabi_mobile.urdf";
    // Instantiate a Serial_Robot object called robot_model
    mecali::Serial_Robot robot_model;
    // Define (optinal) gravity vector to be used
    Eigen::Vector3d gravity_vector(0, 0, -9.81);
    // Create the model based on a URDF file
    robot_model.import_floating_base_model(urdf_filename, gravity_vector, true,
                                           true);

    // ---------------------------------------------------------------------
    // Look inside the robot_model object. What variables can you fetch?
    // ---------------------------------------------------------------------
    // Get some variables contained in the robot_model object
    std::string name = robot_model.name;
    int n_q = robot_model.n_q;
    int n_joints = robot_model.n_joints;
    int n_dof = robot_model.n_dof;
    int n_frames = robot_model.n_frames;
    std::vector<std::string> joint_names = robot_model.joint_names;
    std::vector<std::string> joint_types = robot_model.joint_types;
    Eigen::VectorXd gravity = robot_model.gravity;
    Eigen::VectorXd joint_torque_limit = robot_model.joint_torque_limit;
    Eigen::VectorXd joint_pos_ub = robot_model.joint_pos_ub;
    Eigen::VectorXd joint_pos_lb = robot_model.joint_pos_lb;
    Eigen::VectorXd joint_vel_limit = robot_model.joint_vel_limit;
    Eigen::VectorXd neutral_configuration = robot_model.neutral_configuration;
    // Print some information related to the imported model (boundaries, frames,
    // DoF, etc)
    robot_model.print_model_data();

    // ---------------------------------------------------------------------
    // Set functions for robot dynamics and kinematics
    // ---------------------------------------------------------------------
    double wheel_radius = 0.15;
    double wheel_distance = 0.58;

    // Define symbol
    casadi::SX x_sx = casadi::SX::sym(
        "x", n_q);  // x, y, z, q_w, q_x, q_y, q_z, q_1, q_2, ... , q_7
    casadi::SX xdot_sx = casadi::SX::sym("xdot", n_q);
    casadi::SX u_sx = casadi::SX::sym(
        "u", n_joints);  // wheel_l, wheel_r, q_1, q_2, ... , q_7
    casadi::SX z_sx =
        casadi::SX::sym("z", 5);  // x, y, z, theta, manipulability
    casadi::SX p_sx = casadi::SX::sym("p", 7);  // x, y, z, q_w, q_x, q_y, q_z
    casadi::SX multiplier = casadi::SX::sym("multiplier", n_q + 5);

    // Calculate xdot
    Eigen::MatrixXs mat_eigen;
    mat_eigen.setZero(6 + ARM_Q, 2 + ARM_Q);
    mat_eigen.block<ARM_Q, ARM_Q>(6, 2).setIdentity();
    mat_eigen.block<1, 2>(0, 0) << wheel_radius / 2., wheel_radius / 2.;
    mat_eigen.block<1, 2>(5, 0) << -wheel_radius / wheel_distance,
        wheel_radius / wheel_distance;
    mat_eigen.block<3, 2>(0, 0) =
        Eigen::Quaternions(x_sx.get_elements()[3], x_sx.get_elements()[4],
                           x_sx.get_elements()[5], x_sx.get_elements()[6])
            .toRotationMatrix() *
        mat_eigen.block<3, 2>(0, 0);
    mat_eigen.block<3, 2>(3, 0) =
        Eigen::Quaternions(x_sx.get_elements()[3], x_sx.get_elements()[4],
                           x_sx.get_elements()[5], x_sx.get_elements()[6])
            .toRotationMatrix() *
        mat_eigen.block<3, 2>(3, 0);
    Eigen::MatrixXs u_eigen =
        Eigen::Map<Eigen::MatrixXs>(u_sx.get_elements().data(), n_joints, 1);
    Eigen::MatrixXs qdot_eigen = mat_eigen * u_eigen;
    casadi::SXElem omega_x = qdot_eigen.coeff(3, 0);
    casadi::SXElem omega_y = qdot_eigen.coeff(4, 0);
    casadi::SXElem omega_z = qdot_eigen.coeff(5, 0);
    Eigen::MatrixXs omega_operator(4, 4);
    omega_operator << 0, -omega_x, -omega_y, -omega_z, omega_x, 0, omega_z,
        -omega_y, omega_y, -omega_z, 0, omega_x, omega_z, omega_y, -omega_x, 0;
    Eigen::MatrixXs quaternion_eigen(4, 1);
    quaternion_eigen << x_sx.get_elements()[3], x_sx.get_elements()[4],
        x_sx.get_elements()[5], x_sx.get_elements()[6];
    Eigen::MatrixXs xdot_eigen(n_q, 1);
    xdot_eigen.block<3, 1>(0, 0) = qdot_eigen.block<3, 1>(0, 0);
    xdot_eigen.block<4, 1>(3, 0) = 0.5 * omega_operator * quaternion_eigen;
    xdot_eigen.block<ARM_Q, 1>(7, 0) = qdot_eigen.block<ARM_Q, 1>(6, 0);

    // Calculate z
    casadi::Function fk_pos_ee =
        robot_model.forward_kinematics("position", "flange");
    std::vector<casadi::SXElem> x_reorder = x_sx.get_elements();
    x_reorder.insert(x_reorder.begin() + 7, x_reorder[3]);
    x_reorder.erase(x_reorder.begin() + 3);
    casadi::SX pos_err_sx =
        fk_pos_ee(casadi::SXVector{casadi::SX(x_reorder)})[0] -
        p_sx(casadi::Slice(0, 3), 0);
    std::vector<casadi::SXElem> pos_err_vector = pos_err_sx.get_elements();

    casadi::Function fk_rot_ee =
        robot_model.forward_kinematics("rotation", "flange");
    casadi::SX rot_sx = fk_rot_ee(casadi::SXVector{casadi::SX(x_reorder)})[0];
    Eigen::MatrixXs rot_ref_eigen =
        Eigen::Quaternions(p_sx.get_elements()[3], p_sx.get_elements()[4],
                           p_sx.get_elements()[5], p_sx.get_elements()[6])
            .toRotationMatrix();
    casadi::SX rot_ref_sx(std::vector<casadi::SXElem>(
        rot_ref_eigen.data(),
        rot_ref_eigen.data() + rot_ref_eigen.rows() * rot_ref_eigen.cols()));
    rot_ref_sx = casadi::SX::reshape(rot_ref_sx, 3, 3);
    casadi::SXElem trace_rot_err = sum((rot_sx * rot_ref_sx).get_elements());
    casadi::SXElem theta_err = acos(0.5 * (trace_rot_err - 1));

    std::vector<casadi::SXElem> q(x_reorder.end() - ARM_Q, x_reorder.end());
    casadi::SX jac_q = jacobian(pos_err_sx, casadi::SX(q));
    casadi::SXElem manipulability =
        1. / sqrt(det(mtimes(jac_q, jac_q.T()))).get_elements()[0];

    // Concatenate xdot, z as f_expl
    std::vector<casadi::SXElem> f_expl(
        xdot_eigen.data(),
        xdot_eigen.data() + xdot_eigen.rows() * xdot_eigen.cols());
    f_expl.insert(f_expl.end(), pos_err_vector.begin(), pos_err_vector.end());
    f_expl.push_back(theta_err);
    f_expl.push_back(manipulability);

    // Calculate f_impl and jacobian
    casadi::SX xdot_z_sx = casadi::SX::vertcat(casadi::SXVector{xdot_sx, z_sx});
    casadi::SX f_impl = xdot_z_sx - f_expl;

    casadi::SX jac_x = jacobian(f_impl, x_sx);
    casadi::SX jac_xdot = jacobian(f_impl, xdot_sx);
    casadi::SX jac_u = jacobian(f_impl, u_sx);
    casadi::SX jac_z = jacobian(f_impl, z_sx);

    casadi::SX x_xdot_z_u_sx =
        casadi::SX::vertcat(casadi::SXVector{x_sx, xdot_sx, z_sx, u_sx});
    casadi::SX adjoint = jtimes(f_impl, x_xdot_z_u_sx, multiplier, true);
    casadi::SX hess = jacobian(adjoint, x_xdot_z_u_sx);

    // Set functions
    casadi::Function impl_dae_fun(
        robot_model.name + "_impl_dae_fun",
        casadi::SXVector{x_sx, xdot_sx, u_sx, z_sx, p_sx},
        casadi::SXVector{f_impl});
    casadi::Function impl_dae_fun_jac_x_xdot_z(
        robot_model.name + "_impl_dae_fun_jac_x_xdot_z",
        casadi::SXVector{x_sx, xdot_sx, u_sx, z_sx, p_sx},
        casadi::SXVector{f_impl, jac_x, jac_xdot, jac_z});
    casadi::Function impl_dae_jac_x_xdot_u_z(
        robot_model.name + "_impl_dae_jac_x_xdot_u_z",
        casadi::SXVector{x_sx, xdot_sx, u_sx, z_sx, p_sx},
        casadi::SXVector{jac_x, jac_xdot, jac_u, jac_z});
    casadi::Function impl_dae_fun_jac_x_xdot_u(
        robot_model.name + "_impl_dae_fun_jac_x_xdot_u",
        casadi::SXVector{x_sx, xdot_sx, u_sx, z_sx, p_sx},
        casadi::SXVector{f_impl, jac_x, jac_xdot, jac_u});
    casadi::Function impl_dae_hess(
        robot_model.name + "_impl_dae_hess",
        casadi::SXVector{x_sx, xdot_sx, u_sx, z_sx, multiplier, p_sx},
        casadi::SXVector{hess});

    // ---------------------------------------------------------------------
    // Evaluate a kinematics or dynamics function
    // ---------------------------------------------------------------------
    // Test a function with numerical values
    std::vector<double> x_vec = {0.23,  0.354, -0.52, 1.,    0.,   0.,   0.,
                                 0.243, 1.32,  0.32,  1.386, 3.29, 2.10, 0.42};
    std::vector<double> xdot_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> u_vec = {54.23, -11.12, 2.43, 0.41, 1.37,
                                 1.72,  0.62,   3.12, 0.128};
    std::vector<double> z_vec = {0, 0, 0, 0, 0};
    std::vector<double> p_vec = {0, 0, 0, 0.7071, 0, 0.7071, 0};
    // Evaluate the function with a casadi::DMVector containing q_vec as input
    casadi::DM impl_res =
        impl_dae_fun(casadi::DMVector{x_vec, xdot_vec, u_vec, z_vec, p_vec})[0];
    std::cout << "Func result: " << impl_res << std::endl;

    // Calculate by RBDL
    std::shared_ptr<robot_dynamics::RobotDynamics> robot_model_;
    robot_model_.reset(
        new robot_dynamics::RobotDynamics(urdf_filename, {"flange"}));
    robot_model_->Reset();
    Eigen::VectorXd joint(6);
    joint << x_vec[7], x_vec[8], x_vec[9], x_vec[10], x_vec[11], x_vec[12];
    robot_model_->SetJointPos(joint);
    robot_model_->SetTorsoPosOri(
        Eigen::Vector3d({x_vec[0], x_vec[1], x_vec[2]}),
        Eigen::Quaterniond(x_vec[3], x_vec[4], x_vec[5], x_vec[6]));

    Eigen::MatrixXd joint_mat_;
    joint_mat_.setZero(6 + ARM_Q, 2 + ARM_Q);
    joint_mat_.block<ARM_Q, ARM_Q>(6, 2).setIdentity();
    joint_mat_.block<1, 2>(0, 0) << wheel_radius / 2., wheel_radius / 2.;
    joint_mat_.block<1, 2>(5, 0) << -wheel_radius / wheel_distance,
        wheel_radius / wheel_distance;
    joint_mat_.block<3, 2>(0, 0) =
        Eigen::Quaterniond(x_vec[3], x_vec[4], x_vec[5], x_vec[6])
            .toRotationMatrix() *
        joint_mat_.block<3, 2>(0, 0);
    joint_mat_.block<3, 2>(3, 0) =
        Eigen::Quaterniond(x_vec[3], x_vec[4], x_vec[5], x_vec[6])
            .toRotationMatrix() *
        joint_mat_.block<3, 2>(3, 0);

    Eigen::MatrixXd qdot_rbdl =
        joint_mat_ * Eigen::Map<Eigen::MatrixXd>(u_vec.data(), u_vec.size(), 1);
    double omega_x_rbdl = qdot_rbdl.coeff(3, 0);
    double omega_y_rbdl = qdot_rbdl.coeff(4, 0);
    double omega_z_rbdl = qdot_rbdl.coeff(5, 0);
    Eigen::MatrixXd omega_operator_rbdl(4, 4);
    omega_operator_rbdl << 0, -omega_x_rbdl, -omega_y_rbdl, -omega_z_rbdl,
        omega_x_rbdl, 0, omega_z_rbdl, -omega_y_rbdl, omega_y_rbdl,
        -omega_z_rbdl, 0, omega_x_rbdl, omega_z_rbdl, omega_y_rbdl,
        -omega_x_rbdl, 0;
    Eigen::MatrixXd quaternion_rbdl(4, 1);
    quaternion_rbdl << x_vec[3], x_vec[4], x_vec[5], x_vec[6];
    Eigen::MatrixXd quaternion_dot =
        0.5 * omega_operator_rbdl * quaternion_rbdl;
    std::cout << "RBDL result: [";
    for (int i = 0; i < 3; i++)
        std::cout << -1. * qdot_rbdl.coeff(i, 0) << ", ";
    for (int i = 0; i < 4; i++)
        std::cout << -1. * quaternion_dot.coeff(i, 0) << ", ";
    for (int i = 6; i < 6 + ARM_Q; i++)
        std::cout << -1. * qdot_rbdl.coeff(i, 0) << ", ";

    Eigen::Matrix3d ee_rot = robot_model_->EndEffectorOri(0).toRotationMatrix();
    Eigen::Matrix3d ee_rot_ref =
        Eigen::Quaterniond(p_vec[3], p_vec[4], p_vec[5], p_vec[6])
            .toRotationMatrix();
    for (int i = 0; i < 3; i++)
        std::cout << -1. * (robot_model_->EndEffectorPos(0)[i] - p_vec[i])
                  << ", ";

    std::cout << -1. *
                     acos(0.5 * ((ee_rot.transpose() * ee_rot_ref).trace() - 1))
              << ", ";
    Eigen::MatrixXd ee_jac =
        robot_model_->EndEffectorJacobian(0).block<3, 6>(3, 6);
    std::cout << -1. / sqrt((ee_jac * ee_jac.transpose()).determinant()) << "]"
              << std::endl;

    // ---------------------------------------------------------------------
    // Generate (or save) a function
    // ---------------------------------------------------------------------
    mecali::generate_code(impl_dae_fun, impl_dae_fun.name());
    mecali::generate_code(impl_dae_fun_jac_x_xdot_z,
                          impl_dae_fun_jac_x_xdot_z.name());
    mecali::generate_code(impl_dae_jac_x_xdot_u_z,
                          impl_dae_jac_x_xdot_u_z.name());
    mecali::generate_code(impl_dae_fun_jac_x_xdot_u,
                          impl_dae_fun_jac_x_xdot_u.name());
    mecali::generate_code(impl_dae_hess, impl_dae_hess.name());
}
