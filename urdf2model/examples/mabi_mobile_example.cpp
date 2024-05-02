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
    double wheel_radius = 0.075;
    double wheel_distance = 0.6;
    double a = 1.02;
    double b = 0.585;

    // Define symbol
    casadi::SX x_sx = casadi::SX::sym("x", 4 + ARM_Q);
    // x, y, z, yaw, q_1, ... , q_7
    casadi::SX xdot_sx = casadi::SX::sym("xdot", 4 + ARM_Q);
    // dx, dy, dz, dyaw, dq_1, ... , dq_7
    casadi::SX u_sx = casadi::SX::sym("u", 2 + ARM_Q);
    // v_l, v_r, dq_1, dq_2, ... , dq_7
    casadi::SX z_sx = casadi::SX::sym("z", 7);
    // x, y, z, theta, manipulability, x_base, y_base
    casadi::SX p_sx = casadi::SX::sym("p", 9);
    // x, y, z, q_w, q_x, q_y, q_z, x_base, y_base
    casadi::SX multiplier = casadi::SX::sym("multiplier", 11 + ARM_Q);

    // Calculate xdot
    Eigen::MatrixXs mat_eigen;
    mat_eigen.setZero(4 + ARM_Q, 2 + ARM_Q);
    mat_eigen.block<ARM_Q, ARM_Q>(4, 2).setIdentity();
    mat_eigen.block<1, 2>(0, 0) << wheel_radius / 2., wheel_radius / 2.;
    mat_eigen.block<1, 2>(3, 0) << -wheel_radius / wheel_distance,
        wheel_radius / wheel_distance;
    mat_eigen.block<3, 2>(0, 0) =
        Eigen::AngleAxiss(x_sx.get_elements()[3], Eigen::Vector3s(0, 0, 1))
            .toRotationMatrix() *
        mat_eigen.block<3, 2>(0, 0);
    Eigen::MatrixXs u_eigen =
        Eigen::Map<Eigen::MatrixXs>(u_sx.get_elements().data(), n_joints, 1);
    Eigen::MatrixXs xdot_eigen = mat_eigen * u_eigen;

    // Calculate z
    casadi::Function fk_pos_ee =
        robot_model.forward_kinematics("position", "flange");
    Eigen::Quaternions quat = Eigen::Quaternions(
        Eigen::AngleAxiss(x_sx.get_elements()[3], Eigen::Vector3s(0, 0, 1)));
    std::vector<casadi::SXElem> base_rot(
        {quat.x(), quat.y(), quat.z(), quat.w()});
    std::vector<casadi::SXElem> x_reorder = x_sx.get_elements();
    x_reorder.erase(x_reorder.begin() + 3);
    x_reorder.insert(x_reorder.begin() + 3, base_rot.begin(), base_rot.end());
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
    casadi::SXElem rot_err = 3. - sum((rot_sx * rot_ref_sx).get_elements());

    std::vector<casadi::SXElem> q(x_reorder.end() - ARM_Q, x_reorder.end());
    casadi::SX jac_q = jacobian(pos_err_sx, casadi::SX(q));
    casadi::SXElem manipulability =
        1. / sqrt(det(mtimes(jac_q, jac_q.T()))).get_elements()[0];

    std::vector<casadi::SXElem> base_err =
        (x_sx(casadi::Slice(0, 2), 0) - p_sx(casadi::Slice(7, 9), 0))
            .get_elements();

    // Concatenate xdot, z as f_expl
    std::vector<casadi::SXElem> f_expl(
        xdot_eigen.data(),
        xdot_eigen.data() + xdot_eigen.rows() * xdot_eigen.cols());
    f_expl.insert(f_expl.end(), pos_err_vector.begin(), pos_err_vector.end());
    f_expl.push_back(rot_err);
    f_expl.push_back(manipulability);
    f_expl.insert(f_expl.end(), base_err.begin(), base_err.end());

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
    // Set functions for collision avoidance constraints
    // ---------------------------------------------------------------------
    Eigen::Vector2d origin, map_size;
    origin << -1., -1.5;
    map_size << 3., 3.;
    double resolution = 0.01;
    fiesta::ESDFMap esdf_map(origin, resolution, map_size);

    // set occupancy for some positions
    Eigen::Vector2d pos;
    std::vector<std::pair<double, double>> vp;
    for (double y = -1.5; y <= -0.5; y += 0.01)
        vp.push_back(std::make_pair(0.9, y));
    for (double y = 0.5; y <= 1.5; y += 0.01)
        vp.push_back(std::make_pair(0.9, y));
    for (double l = 0; l <= 1; l += 0.01)
        vp.push_back(std::make_pair(0.9 - 0.5 * sqrt(3) * l, 0.5 - 0.5 * l));

    // insert
    for (auto iter = vp.begin(); iter != vp.end(); iter++) {
        pos << iter->first, iter->second;
        esdf_map.SetOccupancy(pos, 1);
        esdf_map.UpdateESDF();
    }

    std::vector<std::vector<double>> grid;
    std::vector<double> xgrid, ygrid;
    for (int i = 0; i < esdf_map.grid_size_(0); ++i)
        xgrid.push_back((i + 0.5) * esdf_map.resolution_ + esdf_map.origin_(0));
    for (int i = 0; i < esdf_map.grid_size_(1); ++i)
        ygrid.push_back((i + 0.5) * esdf_map.resolution_ + esdf_map.origin_(1));
    grid.push_back(xgrid);
    grid.push_back(ygrid);

    std::vector<double> values;
    for (int idx = 0; idx < esdf_map.grid_total_size_; idx++) {
        Eigen::Vector2i vox(idx % esdf_map.grid_size_(0),
                            idx / esdf_map.grid_size_(0));
        double dist = esdf_map.GetDistance(vox);
        values.push_back(dist);
    }
    casadi::Function esdf_fun =
        casadi::interpolant("esdf", "linear", grid, values);

    // Define symbol
    casadi::MX x_mx = casadi::MX::sym("x", 4 + ARM_Q);
    casadi::MX u_mx = casadi::MX::sym("u", 2 + ARM_Q);
    casadi::MX z_mx = casadi::MX::sym("z", 7);
    casadi::MX p_mx = casadi::MX::sym("p", 9);
    casadi::MX lam_h = casadi::MX::sym("lam", 5);

    casadi::MX c_c = x_mx(casadi::Slice(0, 2), 0);
    casadi::MX c_fl = casadi::MX::vertcat(casadi::MXVector{
        x_mx(0, 0) + a / 2 * cos(x_mx(3, 0)) - b / 2 * sin(x_mx(3, 0)),
        x_mx(1, 0) + a / 2 * sin(x_mx(3, 0)) + b / 2 * cos(x_mx(3, 0))});
    casadi::MX c_fr = casadi::MX::vertcat(casadi::MXVector{
        x_mx(0, 0) + a / 2 * cos(x_mx(3, 0)) + b / 2 * sin(x_mx(3, 0)),
        x_mx(1, 0) + a / 2 * sin(x_mx(3, 0)) - b / 2 * cos(x_mx(3, 0))});
    casadi::MX c_rl = casadi::MX::vertcat(casadi::MXVector{
        x_mx(0, 0) - a / 2 * cos(x_mx(3, 0)) - b / 2 * sin(x_mx(3, 0)),
        x_mx(1, 0) - a / 2 * sin(x_mx(3, 0)) + b / 2 * cos(x_mx(3, 0))});
    casadi::MX c_rr = casadi::MX::vertcat(casadi::MXVector{
        x_mx(0, 0) - a / 2 * cos(x_mx(3, 0)) + b / 2 * sin(x_mx(3, 0)),
        x_mx(1, 0) - a / 2 * sin(x_mx(3, 0)) - b / 2 * cos(x_mx(3, 0))});

    casadi::MX h_mx = casadi::MX::vertcat(casadi::MXVector{
        esdf_fun(c_c)[0] - b / 2, esdf_fun(c_fl)[0], esdf_fun(c_fr)[0],
        esdf_fun(c_rl)[0], esdf_fun(c_rr)[0]});

    casadi::MX h_jac_x = jacobian(h_mx, x_mx);
    casadi::MX h_jac_u = jacobian(h_mx, u_mx);
    casadi::MX h_jac_uxt =
        casadi::MX::vertcat(casadi::MXVector{h_jac_u.T(), h_jac_x.T()});
    casadi::MX h_jac_zt = jacobian(h_mx, z_mx).T();
    casadi::MX u_x_mx = casadi::MX::vertcat(casadi::MXVector{u_mx, x_mx});
    casadi::MX adj_ux = jtimes(h_mx, u_x_mx, lam_h, true);
    casadi::MX hess_ux = jacobian(adj_ux, u_x_mx);
    casadi::MX adj_z = jtimes(h_mx, z_mx, lam_h, true);
    casadi::MX hess_z = jacobian(adj_z, z_mx);

    // Set functions
    casadi::Function nl_constr_h_fun_jac(
        robot_model.name + "_constr_h_fun_jac_uxt_zt",
        casadi::MXVector{x_mx, u_mx, z_mx, p_mx},
        casadi::MXVector{h_mx, h_jac_uxt, h_jac_zt});
    casadi::Function nl_constr_h_fun(robot_model.name + "_constr_h_fun",
                                     casadi::MXVector{x_mx, u_mx, z_mx, p_mx},
                                     casadi::MXVector{h_mx});
    casadi::Function nl_constr_h_fun_jac_hess(
        robot_model.name + "_constr_h_fun_jac_uxt_zt_hess",
        casadi::MXVector{x_mx, u_mx, lam_h, z_mx, p_mx},
        casadi::MXVector{h_mx, h_jac_uxt, hess_ux, h_jac_zt, hess_z});

    // ---------------------------------------------------------------------
    // Evaluate a kinematics or dynamics function
    // ---------------------------------------------------------------------
    // Test a function with numerical values
    std::vector<double> x_vec = {0.23, 0.354, -0.52, 1.43, 0.243, 1.32,
                                 0.32, 1.386, 3.29,  2.10, 0.42};
    std::vector<double> xdot_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> u_vec = {54.23, -11.12, 2.43, 0.41, 1.37,
                                 1.72,  0.62,   3.12, 0.128};
    std::vector<double> z_vec = {0, 0, 0, 0, 0, 0, 0};
    std::vector<double> p_vec = {0, 0, 0, 0.7071, 0, 0.7071, 0, 0, 0};
    // Evaluate the function with a casadi::DMVector containing q_vec as input
    casadi::DM impl_res =
        impl_dae_fun(casadi::DMVector{x_vec, xdot_vec, u_vec, z_vec, p_vec})[0];
    std::cout << "Func result: " << impl_res << std::endl;

    // Calculate by RBDL
    std::vector<double> rbdl_res;
    std::shared_ptr<robot_dynamics::RobotDynamics> robot_model_;
    robot_model_.reset(
        new robot_dynamics::RobotDynamics(urdf_filename, {"flange"}));
    robot_model_->Reset();
    Eigen::VectorXd joint(ARM_Q);
    joint << x_vec[4], x_vec[5], x_vec[6], x_vec[7], x_vec[8], x_vec[9],
        x_vec[10];
    robot_model_->SetJointPos(joint);
    robot_model_->SetTorsoPosOri(
        Eigen::Vector3d({x_vec[0], x_vec[1], x_vec[2]}),
        Eigen::Quaterniond(
            Eigen::AngleAxisd(x_vec[3], Eigen::Vector3d(0, 0, 1))));

    Eigen::MatrixXd joint_mat_;
    joint_mat_.setZero(4 + ARM_Q, 2 + ARM_Q);
    joint_mat_.block<ARM_Q, ARM_Q>(4, 2).setIdentity();
    joint_mat_.block<1, 2>(0, 0) << wheel_radius / 2., wheel_radius / 2.;
    joint_mat_.block<1, 2>(3, 0) << -wheel_radius / wheel_distance,
        wheel_radius / wheel_distance;
    joint_mat_.block<3, 2>(0, 0) =
        Eigen::AngleAxisd(x_vec[3], Eigen::Vector3d(0, 0, 1))
            .toRotationMatrix() *
        joint_mat_.block<3, 2>(0, 0);

    Eigen::MatrixXd qdot_rbdl =
        joint_mat_ * Eigen::Map<Eigen::MatrixXd>(u_vec.data(), u_vec.size(), 1);
    for (int i = 0; i < 4 + ARM_Q; i++)
        rbdl_res.push_back(-1. * qdot_rbdl.coeff(i, 0));

    Eigen::Matrix3d ee_rot = robot_model_->EndEffectorOri(0).toRotationMatrix();
    Eigen::Matrix3d ee_rot_ref =
        Eigen::Quaterniond(p_vec[3], p_vec[4], p_vec[5], p_vec[6])
            .toRotationMatrix();
    for (int i = 0; i < 3; i++)
        rbdl_res.push_back(-1. *
                           (robot_model_->EndEffectorPos(0)[i] - p_vec[i]));

    rbdl_res.push_back(-1. * (3. - (ee_rot.transpose() * ee_rot_ref).trace()));

    Eigen::MatrixXd ee_jac =
        robot_model_->EndEffectorJacobian(0).block<3, 6>(3, 6);
    rbdl_res.push_back(-1. / sqrt((ee_jac * ee_jac.transpose()).determinant()));

    rbdl_res.push_back(-1. * (x_vec[0] - p_vec[7]));
    rbdl_res.push_back(-1. * (x_vec[1] - p_vec[8]));

    std::cout << "RBDL result: " << casadi::DM(rbdl_res) << std::endl;

    // Evaluate the esdf map
    std::ofstream file;
    file.open("map.csv");
    for (double x = -1.5; x <= 2.5; x += 0.1)
        for (double y = -2; y <= 2; y += 0.1) {
            x_vec = {x,    y,     -0.52, 1.43, 0.243, 1.32,
                     0.32, 1.386, 3.29,  2.10, 0.42};
            file << x << ", " << y << ", ";
            std::vector<double> h_fun = nl_constr_h_fun_jac(casadi::DMVector{
                x_vec, u_vec, z_vec, p_vec})[0]
                                            .get_elements();
            for (int i = 0; i < h_fun.size(); i++) file << h_fun[i] << ", ";
            std::vector<double> h_jac = nl_constr_h_fun_jac(casadi::DMVector{
                x_vec, u_vec, z_vec, p_vec})[1]
                                            .get_elements();
            for (int i = 0; i < h_jac.size() - 1; i++) file << h_jac[i] << ", ";
            file << h_jac[h_jac.size() - 1] << std::endl;
        }
    file.close();

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
    mecali::generate_code(nl_constr_h_fun, nl_constr_h_fun.name());
    mecali::generate_code(nl_constr_h_fun_jac, nl_constr_h_fun_jac.name());
    mecali::generate_code(nl_constr_h_fun_jac_hess,
                          nl_constr_h_fun_jac_hess.name());
}
