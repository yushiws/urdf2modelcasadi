#include <casadi/casadi.hpp>
#include <fstream>

#include "esdf_map.hpp"
#include "model_interface.hpp"
#include "robot_dynamics.hpp"

#define ARM_Q 7

namespace Eigen {
typedef Matrix<casadi::SXElem, 3, 1> Vector3s;
typedef Matrix<casadi::SXElem, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Quaternion<casadi::SXElem> Quaternions;
typedef AngleAxis<casadi::SXElem> AngleAxiss;
}  // namespace Eigen

int main() {
    // ---------------------------------------------------------------------
    // Create a model based on a URDF file
    // ---------------------------------------------------------------------
    std::string urdf_filename =
        "../urdf2model/models/mabimobile/mabi_mobile.urdf";
    // Instantiate a Serial_Robot object called robot_model
    mecali::Serial_Robot robot_model;
    // Create the model based on a URDF file
    robot_model.import_model(urdf_filename);

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
    casadi::SX q_sx = casadi::SX::sym("q", ARM_Q);
    casadi::Function fk_pos_ee =
        robot_model.forward_kinematics("position", "flange");
    casadi::SX jac_q = jacobian(fk_pos_ee(casadi::SXVector{q_sx})[0], q_sx);
    casadi::SX manip_jac = jacobian(sqrt(det(mtimes(jac_q, jac_q.T()))), q_sx);
    casadi::Function manip_jac_func(robot_model.name + "_manip_jac",
                                    casadi::SXVector{q_sx},
                                    casadi::SXVector{manip_jac});
    mecali::generate_code(manip_jac_func, manip_jac_func.name());
}
