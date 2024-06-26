#include <casadi/casadi.hpp>
#include "model_interface.hpp"
using namespace std;
int main()
{
  // Example with UR10 URDF.

  // ---------------------------------------------------------------------
  // Create a model based on a URDF file
  // ---------------------------------------------------------------------
  std::string urdf_filename = "../urdf2model/models/atlas/urdf/atlas.urdf";
  // Instantiate a Serial_Robot object called robot_model
  mecali::Serial_Robot robot_model;
  // Define (optinal) gravity vector to be used
  Eigen::Vector3d gravity_vector(0, 0, -9.81);
  // Create the model based on a URDF file
  robot_model.import_floating_base_model(urdf_filename, gravity_vector, true, true);

  // Print some information related to the imported model (boundaries, frames, DoF, etc)
  robot_model.print_model_data();

  // ---------------------------------------------------------------------
  // Set functions for robot dynamics and kinematics
  // ---------------------------------------------------------------------
  // Set function for forward dynamics
  //   casadi::Function fwd_dynamics = robot_model.forward_dynamics();
  // Set function for inverse dynamics
  // casadi::Function inv_dynamics = robot_model.inverse_dynamics();
  // Set function for forward kinematics
  // std::vector<std::string> required_Frames = {"Actuator1", "Actuator2", "Actuator3", "Actuator4", "Actuator5", "Actuator6", "Actuator7", "EndEffector" };

  // casadi::Function fkpos_ee = robot_model.forward_kinematics("position", "EndEffector");
  // casadi::Function fkrot_ee = robot_model.forward_kinematics("rotation", "EndEffector");
  // casadi::Function fk_ee    = robot_model.forward_kinematics("transformation", "EndEffector");
  casadi::Function fk = robot_model.forward_kinematics();
  casadi::Function fd = robot_model.forward_dynamics();
  casadi::Function id = robot_model.inverse_dynamics();

  casadi::Function J_fd = robot_model.forward_dynamics_derivatives("jacobian");
  casadi::Function J_id = robot_model.inverse_dynamics_derivatives("jacobian");

  robot_model.generate_json("atlas.json");

  // ---------------------------------------------------------------------
  // Generate (or save) a function
  // ---------------------------------------------------------------------
  // Code-generate or save a function
  // If you use options, you can set if you want to C-code-generate the function, or just save it as "second_function.casadi" (which can be loaded afterwards using casadi::Function::load("second_function.casadi"))
  mecali::Dictionary codegen_options;
  codegen_options["c"] = false;
  codegen_options["save"] = true;
  // mecali::generate_code(fkpos_ee, "kinova_fkpos_ee", codegen_options);
  // mecali::generate_code(fkrot_ee, "kinova_fkrot_ee", codegen_options);
  // mecali::generate_code(fk_ee, "kinova_fk_ee", codegen_options);
  mecali::generate_code(fk, "atlas_fk", codegen_options);
  mecali::generate_code(fd, "atlas_fd", codegen_options);
  mecali::generate_code(id, "atlas_id", codegen_options);

  mecali::generate_code(J_fd, "atlas_J_fd", codegen_options);
  mecali::generate_code(J_id, "atlas_J_id", codegen_options);
}
