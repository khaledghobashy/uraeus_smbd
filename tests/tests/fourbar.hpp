
#include <iostream>
#include <map>
#include </home/khaledghobashy/Documents/eigen-eigen-323c052e1731/Eigen/Dense>
#include </home/khaledghobashy/Documents/eigen-eigen-323c052e1731/Eigen/Eigen>


typedef std::map<std::string, std::string> Dict_SS;
typedef std::map<std::string, int> Dict_SI;

class Configuration
{

public:
    Eigen::VectorXd q;
    Eigen::VectorXd qd;

public:
    Eigen::Vector3d R_ground ;
    Eigen::Vector4d P_ground ;
    Eigen::Vector3d Rd_ground ;
    Eigen::Vector4d Pd_ground ;
    Eigen::Vector3d Rdd_ground ;
    Eigen::Vector4d Pdd_ground ;
    Eigen::Vector3d R_rbs_crank ;
    Eigen::Vector4d P_rbs_crank ;
    Eigen::Vector3d Rd_rbs_crank ;
    Eigen::Vector4d Pd_rbs_crank ;
    Eigen::Vector3d Rdd_rbs_crank ;
    Eigen::Vector4d Pdd_rbs_crank ;
    double m_rbs_crank ;
    Eigen::Matrix<double, 3, 3> Jbar_rbs_crank ;
    Eigen::Vector3d R_rbs_conct ;
    Eigen::Vector4d P_rbs_conct ;
    Eigen::Vector3d Rd_rbs_conct ;
    Eigen::Vector4d Pd_rbs_conct ;
    Eigen::Vector3d Rdd_rbs_conct ;
    Eigen::Vector4d Pdd_rbs_conct ;
    double m_rbs_conct ;
    Eigen::Matrix<double, 3, 3> Jbar_rbs_conct ;
    Eigen::Vector3d R_rbs_rockr ;
    Eigen::Vector4d P_rbs_rockr ;
    Eigen::Vector3d Rd_rbs_rockr ;
    Eigen::Vector4d Pd_rbs_rockr ;
    Eigen::Vector3d Rdd_rbs_rockr ;
    Eigen::Vector4d Pdd_rbs_rockr ;
    double m_rbs_rockr ;
    Eigen::Matrix<double, 3, 3> Jbar_rbs_rockr ;
    Eigen::Vector3d ax1_jcs_a ;
    Eigen::Vector3d pt1_jcs_a ;
    Eigen::Vector3d ax1_jcs_b ;
    Eigen::Vector3d pt1_jcs_b ;
    Eigen::Vector3d ax1_jcs_c ;
    Eigen::Vector3d ax2_jcs_c ;
    Eigen::Vector3d pt1_jcs_c ;
    Eigen::Vector3d ax1_jcs_d ;
    Eigen::Vector3d pt1_jcs_d ;

public:
    void set_inital_configuration();

};



class Topology
{

public:
    const int n = 28;
    const int nc = 27;
    const int nrows = 14;
    const int ncols = 2*4;

    std::string prefix;
    double t = 0;

    Eigen::VectorXd q0;

    Eigen::VectorXd pos_eq;
    Eigen::VectorXd vel_eq;
    Eigen::VectorXd acc_eq;

    Eigen::VectorXd rows = Eigen::VectorXd::LinSpaced(this->nrows, 0, this->nrows-1);

    Eigen::VectorXd jac_rows;
    Eigen::VectorXd jac_cols;

    Dict_SI indicies_map;

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
    double m_ground ;
    Eigen::Matrix<double, 3, 3> Jbar_ground ;
    Eigen::Matrix3d Mbar_ground_jcs_a ;
    Eigen::Matrix3d Mbar_rbs_crank_jcs_a ;
    Eigen::Vector3d ubar_ground_jcs_a ;
    Eigen::Vector3d ubar_rbs_crank_jcs_a ;
    Eigen::Vector3d F_rbs_crank_gravity ;
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

    Configuration config;

    void initialize();
    void assemble(Dict_SI &indicies_map, Dict_SS &interface_map, int rows_offset);
    void set_initial_states();
    void eval_constants();

    void eval_pos_eq();
    void eval_vel_eq();
    void eval_acc_eq();
    void eval_jac_eq();

    void set_gen_coordinates(Eigen::VectorXd &q);
    void set_gen_velocities(Eigen::VectorXd &qd);
    void set_gen_accelerations(Eigen::VectorXd &qdd);

private:
    void set_mapping(Dict_SI &indicies_map, Dict_SS &interface_map);


};
