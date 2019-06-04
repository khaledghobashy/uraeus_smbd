
#include <iostream>
#include </home/khaledghobashy/Documents/eigen-eigen-323c052e1731/Eigen/Dense>

#include "AbstractClasses.hpp"


class Topology : public AbstractTopology
{
public:
    int ground ;
    int rbs_crank ;
    int rbs_conct ;
    int rbs_rockr ;

public:
    Eigen::Vector3d R_ground ;
    Eigen::Vector4d P_ground ;
    Eigen::Vector3d R_rbs_crank ;
    Eigen::Vector4d P_rbs_crank ;
    Eigen::Vector3d R_rbs_conct ;
    Eigen::Vector4d P_rbs_conct ;
    Eigen::Vector3d R_rbs_rockr ;
    Eigen::Vector4d P_rbs_rockr ;

public:
    Eigen::Vector3d Rd_ground ;
    Eigen::Vector4d Pd_ground ;
    Eigen::Vector3d Rd_rbs_crank ;
    Eigen::Vector4d Pd_rbs_crank ;
    Eigen::Vector3d Rd_rbs_conct ;
    Eigen::Vector4d Pd_rbs_conct ;
    Eigen::Vector3d Rd_rbs_rockr ;
    Eigen::Vector4d Pd_rbs_rockr ;

public:
    Eigen::Vector3d Rdd_ground ;
    Eigen::Vector4d Pdd_ground ;
    Eigen::Vector3d Rdd_rbs_crank ;
    Eigen::Vector4d Pdd_rbs_crank ;
    Eigen::Vector3d Rdd_rbs_conct ;
    Eigen::Vector4d Pdd_rbs_conct ;
    Eigen::Vector3d Rdd_rbs_rockr ;
    Eigen::Vector4d Pdd_rbs_rockr ;

public:    
    Eigen::Vector4d Pg_ground ;
    double m_ground;
    Eigen::Matrix<double, 3, 3> Jbar_ground ;
    Eigen::Matrix3d Mbar_ground_jcs_a ;
    Eigen::Matrix3d Mbar_rbs_crank_jcs_a ;
    Eigen::Vector3d ubar_ground_jcs_a ;
    Eigen::Vector3d ubar_rbs_crank_jcs_a ;
    Eigen::Matrix3d Mbar_ground_jcs_a ;
    Eigen::Matrix3d Mbar_rbs_crank_jcs_a ;
    Eigen::Vector3d F_rbs_crank_gravity ;
    Eigen::Vector3d ubar_rbs_crank_mcs_abs ;
    Eigen::Vector3d ubar_ground_mcs_abs ;
    Eigen::Matrix3d Mbar_rbs_crank_jcs_b ;
    Eigen::Matrix3d Mbar_rbs_conct_jcs_b ;
    Eigen::Vector3d ubar_rbs_crank_jcs_b ;
    Eigen::Vector3d ubar_rbs_conct_jcs_b ;
    Eigen::Vector3d F_rbs_conct_gravity ;
    Eigen::Matrix3d Mbar_rbs_conct_jcs_c ;
    Eigen::Matrix3d Mbar_rbs_rockr_jcs_c ;
    Eigen::Vector3d ubar_rbs_conct_jcs_c ;
    Eigen::Vector3d ubar_rbs_rockr_jcs_c ;
    Eigen::Vector3d F_rbs_rockr_gravity ;
    Eigen::Matrix3d Mbar_rbs_rockr_jcs_d ;
    Eigen::Matrix3d Mbar_ground_jcs_d ;
    Eigen::Vector3d ubar_rbs_rockr_jcs_d ;
    Eigen::Vector3d ubar_ground_jcs_d ;

public:

    Topology(std::string prefix);

    void initialize();
    void assemble(Dict_SI, Dict_SS, int);
    void set_initial_states();
    void eval_constants();

    void eval_pos_equations();
    void eval_vel_equations();
    void eval_acc_equations();
    void eval_jac_equations();

    void set_gen_coordinates(Eigen::VectorXd);
    void set_gen_velocities(Eigen::VectorXd);
    void set_gen_accelerations(Eigen::VectorXd);

private:
    void set_mapping(Dict_SI, Dict_SS);


};
