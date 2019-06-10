#pragma once

#include <map>
#include "eigen-eigen-323c052e1731/Eigen/Dense"


typedef std::map<std::string, std::string> Dict_SS;
typedef std::map<std::string, int> Dict_SI;

class AbstractTopology
{
public:

    std::string prefix ;
    double t ;
    int n ;
    int nc ;
    int nrows ;
    int ncols ;

    Eigen::VectorXd rows;
    Eigen::VectorXd jac_rows;
    Eigen::VectorXd jac_cols;

    Eigen::VectorXd pos_eq;
    Eigen::VectorXd vel_eq;
    Eigen::VectorXd acc_eq;
    std::vector<Eigen::MatrixXd> jac_eq;
    
    Dict_SI indicies_map;
    
    Eigen::VectorXd q ;
    Eigen::VectorXd qd;
    Eigen::VectorXd q0;

public:

    virtual void initialize();
    virtual void assemble(Dict_SI& indicies_map, Dict_SS& interface_map, int rows_offset);
    virtual void set_initial_states();
    virtual void eval_constants();

    virtual void eval_pos_eq();
    virtual void eval_vel_eq();
    virtual void eval_acc_eq();
    virtual void eval_jac_eq();

    virtual void set_gen_coordinates(Eigen::VectorXd& q);
    virtual void set_gen_velocities(Eigen::VectorXd& qd);
    virtual void set_gen_accelerations(Eigen::VectorXd& qdd);

private:
    virtual void set_mapping(Dict_SI& indicies_map, Dict_SS& interface_map);

};


