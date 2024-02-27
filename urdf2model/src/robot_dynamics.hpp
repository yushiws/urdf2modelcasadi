#ifndef ROBOT_DYNAMICS_HPP_
#define ROBOT_DYNAMICS_HPP_

#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <string>

namespace RigidBodyDynamics {

class Model;

}

namespace Eigen {

// avoid compliation error on std::vector<Eigen::Vector3d>
class Vector3d_t : public Vector3d {
 public:
    Vector3d_t(const Vector3d &v) : Vector3d(v) {}
    Vector3d_t(Vector3d &&v) : Vector3d(v) {}

    template <typename OtherDerived>
    Vector3d_t(const DenseBase<OtherDerived> &other) {
        Base::_set(other);
    }
};

class MatrixXd_t : public MatrixXd {
 public:
    MatrixXd_t(const MatrixXd &m) : MatrixXd(m) {}

    MatrixXd_t(MatrixXd &&m) : MatrixXd(m) {}

    template <typename OtherDerived>
    MatrixXd_t(const DenseBase<OtherDerived> &other) {
        Base::_set(other);
    }
};

using Vector6d = Matrix<double, 6, 1>;

/**
 * @brief VecToSkewSymmetricMat
 * Convert a vector v to a matrix S that satisfies
 * cross(v, q) = S * q for all 3-d vector q
 * @param v
 * @return converted skew symmetric matrix
 */
inline Matrix3d VecToSkewSymmetricMat(const Vector3d &v) {
    Matrix3d S;
    S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return S;
}

}  // namespace Eigen

namespace robot_dynamics {

/**
 * @brief The RobotDynamics class
 * This class is a wrapper for RBDL. Note that in RBDL>=3.0.0.,
 * naming a link as "base_link" in urdf will cause RBDL to fix it to the ground.
 * So take care about urdf link name.
 *
 * In this class, we use "base link" or "torso" to refer to the floating base
 * of the robot.
 *
 * In this class, we use following notations for frames of reference:
 * Fixed frame is the stationary inertial frame of reference.
 * Torso frame is the frame attached to torso.
 * Centroidal frame is a frame whose origin is identical with robot's
 *      centroid and orientation identical with fixed frame.
 *
 * RBDL does not directly handle floating-base robots.
 * Instead, it adds a "floating base joint" between the floating base
 * and the fixed base. Such joint consists of a 3-d prismatic joint
 * (enables free translation) and a spherical joint (enables free rotation).
 * The existance of floating base joint cause the generalized coordinates
 * of RBDL robots to differ a bit from that of floating base robots.
 * Specifically, the first six elements of the generalized velocity become
 * [lin_vel_in_fixed_frame, rot_vel_in_torso_frame], while in typical
 * floating base robot that is spatial velocity.
 */
class RobotDynamics {
 public:
    RobotDynamics(const std::string &urdf_path,
                  const std::vector<std::string> &end_effector_names);

    /**
     * @brief GetLinkNameList
     * Get a list of link names, in RBDL order.
     * @return a vector of link names
     */
    std::vector<std::string> GetLinkNameList() const;

    /**
     * @brief NumOfJoint
     * @return number of one-dof joints in this robot.
     */
    int NumOfJoint() const { return qdot_size_ - 6; }

    /**
     * @brief Reset
     * Indicate that the robot configuration is changed and
     * re-calculation is required.
     */
    void Reset() {
        base_pos_ori_loaded_ = false;
        joint_pos_loaded_ = false;
        base_vel_loaded_ = false;
        joint_vel_loaded_ = false;
        base_acc_loaded_ = false;
        joint_acc_loaded_ = false;
        kinematics_calculated_ = false;
        ee_pos_ori_calculated_ = false;
        jacobian_calculated_ = false;
        kin_vel_calculated_ = false;
        dj_dq_calculated_ = false;
        com_state_calculated_ = false;
        dynamics_calculated_ = false;
        cen_dyn_calculated_ = false;
        wrench_calculated_ = false;
    }

    /**
     * @brief SetTorsoPosOri
     * Set robot's base link position and orientation,
     * both are measured in fixed frame.
     * @param pos base position
     * @param ori base orientation
     */
    void SetTorsoPosOri(const Eigen::Vector3d &pos,
                        const Eigen::Quaterniond &ori) {
        q_.head<3>() = pos;
        q_.segment<3>(3) = ori.coeffs().head<3>();
        q_(q_size_ - 1) = ori.w();
        base_pos_ori_loaded_ = true;
    }

    /**
     * @brief SetTorsoVel
     * Set base link's linear and angular velocity of the robot,
     * Both are measured in fixed frame. However, angular velocity
     * is expressed in torso frame, i.e.,
     * ang_vel = R * ang_vel_in_fixed_frame.
     * @param lin_vel base linear velocity in fixed frame
     * @param ang_vel base angular velocity EXPRESSED IN TORSO FRAME
     */
    void SetTorsoVel(const Eigen::Vector3d &lin_vel,
                     const Eigen::Vector3d &ang_vel) {
        qdot_.head<3>() = lin_vel;
        qdot_.segment<3>(3) = ang_vel;
        base_vel_loaded_ = true;
    }

    /**
     * @brief SetJointPos
     * Set joint position of the robot. Currently, only
     * one-dof joint is supported. The joint order may differ
     * from that in the urdf file. Use GetLinkNameList() to get
     * RBDL's order of links.
     * @param jpos a vector containing all joint positions.
     */
    void SetJointPos(const Eigen::VectorXd &jpos) {
        q_.segment(6, q_size_ - 6 - 1) = jpos;
        joint_pos_loaded_ = true;
    }

    /**
     * @brief SetJointVel
     * @param jvel a vector containing all joint velocities.
     */
    void SetJointVel(const Eigen::VectorXd &jvel) {
        qdot_.tail(qdot_size_ - 6) = jvel;
        joint_vel_loaded_ = true;
    }

    /**
     * @brief SetQdot
     * Directly set the generalized velocity of the model.
     * Generalized velocity contains both torso velocity (linear & angular)
     * and joint velocity. It is made up by the following form:
     * qdot = [torso_lin_vel, torso_rot_vel_in_torso_frame, joint_vel]
     * Since RBDL use different form of spatial velocity,
     * ONLY USE THIS if you know qdot's form very clearly!
     * @param qdot generalized velocity
     */
    void SetQdot(const Eigen::VectorXd &qdot) {
        qdot_ = qdot;
        base_vel_loaded_ = true;
        joint_vel_loaded_ = true;
    }

    /**
     * @brief SetTorsoAcc
     * Set base link's linear and angular accelerations.
     * Both accelerations are measured in fixed frame.
     * However, angular acceleration is expressed in torso frame,
     * i.e., ang_acc = R * ang_acc_in_fixed_frame.
     * @param lin_acc base link's linear acceleration.
     * @param ang_acc base link's rotational acc EXPRESSED IN TORSO FRAME
     */
    void SetTorsoAcc(const Eigen::VectorXd &lin_acc,
                     const Eigen::VectorXd &ang_acc) {
        qddot_.head<3>() = lin_acc;
        qddot_.segment<3>(3) = ang_acc;
        base_acc_loaded_ = true;
    }

    /**
     * @brief SetJointAcc
     * @param jacc joint acceleration.
     */
    void SetJointAcc(const Eigen::VectorXd &jacc) {
        qddot_.tail(qdot_size_ - 6) = jacc;
        joint_acc_loaded_ = true;
    }

    /**
     * @brief SetQddot
     * Directly set the derivative of generalized velocity.
     * Since RBDL use different form of spatial velocity,
     * ONLY USE THIS if you know qddot's form very clearly!
     * @param qddot the derivative of generalized velocity.
     */
    void SetQddot(const Eigen::VectorXd &qddot) {
        qddot_ = qddot;
        base_acc_loaded_ = true;
        joint_acc_loaded_ = true;
    }

    const Eigen::Vector3d TorsoPos() const { return q_.head<3>(); }

    const Eigen::Quaterniond TorsoOri() const {
        return Eigen::Quaterniond(q_(q_size_ - 1), q_(3), q_(4), q_(5));
    }

    Eigen::VectorXd JointPos() const { return q_.segment(6, q_size_ - 7); }

    /**
     * @brief Qdot
     * Use this in case you don't understand the form of qdot!
     */
    const Eigen::VectorXd &Qdot() const { return qdot_; }

    /**
     * @brief Qddot
     * Use this in case you don't understand the form of qddot!
     */
    const Eigen::VectorXd &Qddot() const { return qddot_; }

    /**
     * @brief TotalMass
     * Get the robot's total mass.
     * @return total mass
     */
    double TotalMass() const { return total_mass_; }

    /**
     * @brief EndEffectorPos
     * Get end effector's position in fixed frame.
     * @param ee_id end effector's id.
     * @return End effector's position in fixed frame.
     */
    const Eigen::Vector3d &EndEffectorPos(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!ee_pos_ori_calculated_)
            ComputeEEPosOri();
        return ee_pos_[ee_id];
    }

    const Eigen::Vector3d &EndEffectorPosConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!ee_pos_ori_calculated_)
            NotComputedError("end effector position & orientation");
        return ee_pos_[ee_id];
    }

    /**
     * @brief EndEffectorOri
     * Get end effector's orientation in fixed frame.
     * @param ee_id end effector's id.
     * @return End effector's orientation in fixed frame.
     */
    const Eigen::Quaterniond &EndEffectorOri(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!ee_pos_ori_calculated_)
            ComputeEEPosOri();
        return ee_ori_[ee_id];
    }

    const Eigen::Quaterniond &EndEffectorOriConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!ee_pos_ori_calculated_)
            NotComputedError("end effector position & orientation");
        return ee_ori_[ee_id];
    }

    /**
     * @brief EndEffectorJacobian
     * Get end effector's spatial jacobian. The spatial jacobian
     * is a 6 x dof matrix that satisfies:
     * v_spatial = J * qdot.
     * Note that v_spatial = [rot_vel, lin_vel] follows the form of the
     * commonly used spatial vector, and is measured in fixed frame.
     * @param ee_id end effector's id
     * @return Spatial jacobian.
     */
    const Eigen::MatrixXd &EndEffectorJacobian(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            ComputeJacobian();
        return ee_jacobian_[ee_id];
    }

    const Eigen::MatrixXd &EndEffectorJacobianConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            NotComputedError("jacobian");
        return ee_jacobian_[ee_id];
    }

    /**
     * @brief EndEffectorAngVelJacob
     * @param ee_id end effector's id
     * @return angular velocity jacobian
     */
    Eigen::MatrixXd EndEffectorAngVelJacob(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            ComputeJacobian();
        return ee_jacobian_[ee_id].topRows<3>();
    }

    Eigen::MatrixXd EndEffectorAngVelJacobConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            NotComputedError("jacobian");
        return ee_jacobian_[ee_id].topRows<3>();
    }

    /**
     * @brief EndEffectorLinVelJacob
     * @param ee_id end effector's id
     * @return linear velocity jacobian
     */
    Eigen::MatrixXd EndEffectorLinVelJacob(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            ComputeJacobian();
        return ee_jacobian_[ee_id].bottomRows<3>();
    }

    Eigen::MatrixXd EndEffectorLinVelJacobConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!jacobian_calculated_)
            NotComputedError("jacobian");
        return ee_jacobian_[ee_id].bottomRows<3>();
    }

    /**
     * @brief EndEffectorJdotQdot
     * Get the result of d(Spatial Jacobian)/dt * qdot assuming
     * current generalized position and velocity.
     * This term is useful as acceleration of the end effector
     * can be calculated by:
     * spatial_acc = d(spatial_vel)/dt = d(J * qdot)/dt
     *      = JdotQdot + J * qddot
     * @param ee_id end effector's id
     * @return d(Spatial Jacobian)/dt * qdot of the end effector
     */
    const Eigen::Vector6d &EndEffectorJdotQdot(int ee_id) {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!dj_dq_calculated_)
            ComputeJdotQdot();
        return ee_dj_dq_[ee_id];
    }

    const Eigen::Vector6d &EndEffectorJdotQdotConst(int ee_id) const {
        if (ee_id < 0 || ee_id >= ee_cnt_)
            OutOfRangeError(ee_id);
        if (!dj_dq_calculated_)
            NotComputedError("dj_dq");
        return ee_dj_dq_[ee_id];
    }

    /**
     * @brief CoMPos
     * Get center of mass position in fixed frame.
     * @return center of mass position
     */
    const Eigen::Vector3d &CoMPos() {
        if (!com_state_calculated_)
            ComputeCoMState();
        return com_pos_;
    }

    const Eigen::Vector3d &CoMPosConst() const {
        if (!com_state_calculated_)
            NotComputedError("CoM state");
        return com_pos_;
    }

    /**
     * @brief CoMVel
     * Get center of mass velocity in fixed frame.
     * @return center of mass linear velocity
     */
    const Eigen::Vector3d &CoMVel() {
        if (!com_state_calculated_)
            ComputeCoMState();
        return com_vel_;
    }

    const Eigen::Vector3d &CoMVelConst() const {
        if (!com_state_calculated_)
            NotComputedError("CoM state");
        return com_vel_;
    }

    /**
     * @brief LinearMomentum
     * Get the total momentum of the robot expressed in fixed frame.
     * Should equal to com_vel * total_mass
     * @return momentum
     */
    const Eigen::Vector3d &LinearMomentum() {
        if (!com_state_calculated_)
            ComputeCoMState();
        return lin_momentum_;
    }

    const Eigen::Vector3d &LinearMomentumConst() const {
        if (!com_state_calculated_)
            NotComputedError("CoM state");
        return lin_momentum_;
    }

    /**
     * @brief AngularMomentum
     * Get the total angular momentum of the robot expressed in fixed frame.
     * @return angular momentum
     */
    const Eigen::Vector3d &AngularMomentum() {
        if (!com_state_calculated_)
            ComputeCoMState();
        return ang_momentum_;
    }

    const Eigen::Vector3d &AngularMomentumConst() const {
        if (!com_state_calculated_)
            NotComputedError("CoM state");
        return ang_momentum_;
    }

    /**
     * @brief MassMatrix
     * Get the robot's mass matrix H. The mass matrix satisfies:
     * H * qddot + C + G = [f_ext, torq_ext, joint_torq]
     * In the equation above, f_ext is expressed in fixed frame,
     * torq_ext is expressed in torso frame.
     * C denotes the generalized Coriolis force and
     * G denotes the generalized gravity, which can be aquired by
     * CoriolisForce() and GeneralizedGravity(), respectively.
     * @return
     */
    const Eigen::MatrixXd &MassMatrix() {
        if (!dynamics_calculated_)
            ComputeDynamics();
        return mass_matrix_;
    }

    const Eigen::MatrixXd &MassMatrixConst() const {
        if (!dynamics_calculated_)
            NotComputedError("dynamics");
        return mass_matrix_;
    }

    /**
     * @brief CoriolisForce
     * Get the robot's generalized Coriolis force.
     * Since the robot has rotating joints
     * (and the robot itself may also be rotating),
     * it takes all generalized joints to apply forces to merely
     * maintain their velocity. The force required for maintaining
     * the velocity of joints is called generalized Coriolis force.
     * @return generalized Coriolis force
     */
    const Eigen::VectorXd &CoriolisForce() {
        if (!dynamics_calculated_)
            ComputeDynamics();
        return cori_;
    }

    const Eigen::VectorXd &CoriolisForceConst() const {
        if (!dynamics_calculated_)
            NotComputedError("dynamics");
        return cori_;
    }

    /**
     * @brief GeneralizedGravity
     * Get the robot's generalized gravity.
     * As the robot is affected by gravity, an extra force
     * must be applied to cancel it. The generalized force required
     * to cancel gravity effect is called generalized gravity.
     * Strictly speaking, this term should be called
     * negative generalized gravity, for it in fact is in the opposite
     * direction of gravity.
     * @return generalized gravity
     */
    const Eigen::VectorXd &GeneralizedGravity() {
        if (!dynamics_calculated_)
            ComputeDynamics();
        return grav_;
    }

    const Eigen::VectorXd &GeneralizedGravityConst() const {
        if (!dynamics_calculated_)
            NotComputedError("dynamics");
        return grav_;
    }

    /**
     * @brief CentroidMomentumMat
     * Get the robot's centroidal momentum matrix A_g.
     * A_g is a 6 x dof matrix which is the "jacobian"
     * of spatial momentum w.r.t. generalized velocity,
     * i.e., A_g satisfies the following equation:
     * h_g = A_g * qdot for all qdot.
     * Here, h_g = [ang_momentum, lin_momentum] is expressed
     * at centroid frame.
     * This method implements the algorithm given by
     * "Improved Computation of the Humanoid Centroidal Dynamics
     * and Application for Whole-Body Control".
     * @see{https://doi.org/10.1142/S0219843615500395}
     * @return centroidal momentum matrix
     */
    const Eigen::MatrixXd &CentroidMomentumMat() {
        if (!cen_dyn_calculated_)
            ComputeCentroidDynamics();
        return cmm_;
    }

    const Eigen::MatrixXd &CentroidMomentumMatConst() const {
        if (!cen_dyn_calculated_)
            NotComputedError("centroid dynamics");
        return cmm_;
    }

    /**
     * @brief AGdotQdot
     * Get the result of d(A_g)/dt * qdot assuming
     * current generalized position and velocity.
     * @return d(A_g)/dt * qdot
     */
    const Eigen::Vector6d &AGdotQdot() {
        if (!cen_dyn_calculated_)
            ComputeCentroidDynamics();
        return dcmm_dq_;
    }

    const Eigen::Vector6d &AGdotQdotConst() const {
        if (!cen_dyn_calculated_)
            NotComputedError("centroid dynamics");
        return dcmm_dq_;
    }

    /**
     * @brief CentroidalWrench
     * Get the external wrench (= [torq, force]) measured
     * in centroidal frame, given generalized acceleration.
     * @return external wrench
     */
    const Eigen::Vector6d &CentroidalWrench() {
        if (!wrench_calculated_)
            ComputeWrench();
        return fg_;
    }

    const Eigen::Vector6d &CentroidalWrenchConst() const {
        if (!wrench_calculated_)
            NotComputedError("wrench");
        return fg_;
    }

 protected:
    void ComputeKinematics();

    void ComputeEEPosOri();

    void ComputeJacobian();

    void ComputeVelKin();

    void ComputeJdotQdot();

    void ComputeCoMState();

    void ComputeDynamics();

    void ComputeCentroidDynamics();

    void ComputeWrench();

    void OutOfRangeError(int idx) const __attribute__((__noreturn__));

    void NotComputedError(const char *item) const __attribute__((__noreturn__));

    // RBDL interface
    std::shared_ptr<RigidBodyDynamics::Model> model_;

    // constants
    std::vector<unsigned int> ee_ids;
    int ee_cnt_;
    int qdot_size_;
    // Floating base robots have a sperical joint (between torso and ground).
    // RBDL use quaternion for sperical joint orientation,
    // so q_size_ = qdot_size_ + 1.
    int q_size_;

    // computation state indicator
    bool base_pos_ori_loaded_;
    bool joint_pos_loaded_;
    bool base_vel_loaded_;
    bool joint_vel_loaded_;
    bool base_acc_loaded_;
    bool joint_acc_loaded_;
    bool kinematics_calculated_;
    bool ee_pos_ori_calculated_;
    bool jacobian_calculated_;
    bool kin_vel_calculated_;
    bool dj_dq_calculated_;
    bool com_state_calculated_;
    bool dynamics_calculated_;
    bool cen_dyn_calculated_;
    bool wrench_calculated_;

    // variables and computation results
    // generalized states and accelerations
    Eigen::VectorXd q_;
    Eigen::VectorXd qdot_;
    Eigen::VectorXd qddot_;

    // end effector positions/orientations
    std::vector<Eigen::Vector3d_t> ee_pos_;
    std::vector<Eigen::Quaterniond> ee_ori_;

    // jacobian of a given end effector, a 6 * qdot_size_ matrix
    // first 3 rows: angular vel jacobian
    // last 3 rows: linear vel jacobian
    std::vector<Eigen::MatrixXd_t> ee_jacobian_;
    // dJ * dq of a given end effector
    std::vector<Eigen::Vector6d> ee_dj_dq_;

    // center of mass point state
    Eigen::Vector3d com_pos_;
    Eigen::Vector3d com_vel_;
    double total_mass_;
    Eigen::Vector3d lin_momentum_;  // momentum
    Eigen::Vector3d ang_momentum_;  // centroidal angular momentum

    // dynamics
    Eigen::MatrixXd mass_matrix_;
    Eigen::VectorXd cori_;
    Eigen::VectorXd grav_;

    // centroid dynamics
    // hg = CMM * qdot
    // fg = CMM * qddot + CMM_dot * q_dot
    Eigen::MatrixXd cmm_;      // Centroidal momentum matrix
    Eigen::Vector6d dcmm_dq_;  // CMM_dot * q_dot

    // external wrench on centroid
    // fg_ = [torq_ext, f_ext]^T
    Eigen::Vector6d fg_;  // net external wrench
};

}  // namespace robot_dynamics

#endif  // ROBOT_DYNAMICS_HPP_
