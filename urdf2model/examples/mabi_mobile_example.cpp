#include <casadi/casadi.hpp>

#include "model_interface.hpp"

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
        "x", n_q);  // x, y, z, q_w, q_x, q_y, q_z, q_1, q_2, ... , q_6
    casadi::SX xdot_sx = casadi::SX::sym("xdot", n_q);
    casadi::SX u_sx = casadi::SX::sym("u", n_joints);
    casadi::SX z_sx = casadi::SX::sym("z", 12);
    casadi::SX p_sx = casadi::SX::sym("p", 0);
    casadi::SX multiplier = casadi::SX::sym("multiplier", n_q + 12);

    // Calculate xdot
    Eigen::MatrixXs mat_eigen;
    mat_eigen.setZero(12, 8);
    mat_eigen.block<6, 6>(6, 2).setIdentity();
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
    xdot_eigen.block<6, 1>(7, 0) = qdot_eigen.block<6, 1>(6, 0);

    // Calculate z
    casadi::Function fk_pos_ee =
        robot_model.forward_kinematics("position", "EE_FORCETORQUESENSOR");
    casadi::Function fk_rot_ee =
        robot_model.forward_kinematics("rotation", "EE_FORCETORQUESENSOR");
    casadi::SX pos_sx = fk_pos_ee(std::vector<casadi::SX>{x_sx})[0];
    casadi::SX rot_sx = fk_rot_ee(std::vector<casadi::SX>{x_sx})[0];
    std::vector<casadi::SXElem> pos_vector = pos_sx.get_elements();
    std::vector<casadi::SXElem> rot_vector = rot_sx.get_elements();

    // Concatenate xdot, z as f_expl
    std::vector<casadi::SXElem> f_expl(
        xdot_eigen.data(),
        xdot_eigen.data() + xdot_eigen.rows() * xdot_eigen.cols());
    f_expl.insert(f_expl.end(), pos_vector.begin(), pos_vector.end());
    f_expl.insert(f_expl.end(), rot_vector.begin(), rot_vector.end());

    // Calculate f_impl and jacobian
    casadi::SX xdot_z_sx =
        casadi::SX::vertcat(std::vector<casadi::SX>{xdot_sx, z_sx});
    casadi::SX f_impl = xdot_z_sx - f_expl;

    casadi::SX jac_x = jacobian(f_impl, x_sx);
    casadi::SX jac_xdot = jacobian(f_impl, xdot_sx);
    casadi::SX jac_u = jacobian(f_impl, u_sx);
    casadi::SX jac_z = jacobian(f_impl, z_sx);

    casadi::SX x_xdot_z_u_sx =
        casadi::SX::vertcat(std::vector<casadi::SX>{x_sx, xdot_sx, z_sx, u_sx});
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
    std::vector<double> x_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> xdot_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> u_vec = {1, 1, 0, 0, 0, 0, 0, 0};
    std::vector<double> z_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> p_vec = {};
    // Evaluate the function with a casadi::DMVector containing q_vec as input
    casadi::DM impl_res =
        impl_dae_fun(casadi::DMVector{x_vec, xdot_vec, u_vec, z_vec, p_vec})[0];
    std::cout << "Function result: " << impl_res << std::endl;

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
