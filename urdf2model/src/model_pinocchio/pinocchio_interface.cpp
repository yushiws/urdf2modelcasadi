#include "pinocchio_interface.h"

#define DoP 4 // degree of parallelism

pinocchio::Model model;

// initialize your own size
// data for the NMPC controller
pinocchio::Data data[DoP] = {pinocchio::Data(model),pinocchio::Data(model),pinocchio::Data(model),pinocchio::Data(model)};

// data for the simulation model
pinocchio::Data sim_data(model);

void robot_init(std::string filename)
{
    pinocchio::urdf::buildModel(filename,model);

    // Set the gravity applied to the model
    model.gravity.linear(pinocchio::Model::gravity981);     //model.gravity.setZero(), model.gravity.linear( Eigen::Vector3d(0,0,-9.8));

    // init for NMPC
    for(int i=0;i<DoP;i++)
    {
        data[i] = pinocchio::Data(model);
    }
    // init for Simulation
    sim_data = pinocchio::Data(model);

    casadi::SX x = casadi::SX::sym("x");
    casadi::SX y = casadi::SX::sym("y");
    casadi::Function f("f", {x, y}, {2*x, x/y});
    std::vector<casadi::DM> f_arg = {3,4};
    std::cout << "f_kin: " << f(f_arg) << std::endl;

}

void qdd_cal(double *q, double *qd, double *qdd, double *tau, int parIdx)
{
    Eigen::VectorXd q_Eigen   = Eigen::Map<Eigen::VectorXd>(q, model.nv);
    Eigen::VectorXd qd_Eigen  = Eigen::Map<Eigen::VectorXd>(qd,model.nv);
    Eigen::VectorXd qdd_Eigen = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd tau_Eigen = Eigen::Map<Eigen::VectorXd>(tau,model.nv);

    qdd_Eigen = pinocchio::aba(model,data[parIdx-1],q_Eigen,qd_Eigen,tau_Eigen);

    // to double
    Eigen::Map<Eigen::VectorXd>(qdd,model.nv) = qdd_Eigen;

//    std::cout << "qdd = " << qdd_Eigen << std::endl;
}

int get_ndof() {return model.nv;}
int get_nq() {return model.nq;}
