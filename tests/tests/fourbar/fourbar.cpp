
#include "fourbar.hpp"


Topology::Topology(std::string prefix)
{
    this-> prefix = prefix;

    this-> q0.resize(this-> n);

    this-> pos_eq.resize(this-> nc);
    this-> vel_eq.resize(this-> nc);
    this-> acc_eq.resize(this-> nc);

    this-> indicies_map[prefix + "ground"] = 0;
    this-> indicies_map[prefix + "rbs_crank"] = 1;
    this-> indicies_map[prefix + "rbs_conct"] = 2;
    this-> indicies_map[prefix + "rbs_rockr"] = 3;
};



void Topology::initialize()
{
    Dict_SS interface_map;
    this->t = 0;
    this->assemble(this->indicies_map, interface_map, 0);
    this->set_initial_states();
    this->eval_constants();

};


void Topology::assemble(Dict_SI &indicies_map, Dict_SS &interface_map, int rows_offset)
{
    this-> set_mapping(indicies_map, interface_map);
    this-> rows += (rows_offset * Eigen::VectorXd::Ones(this ->rows.size()) );

    this-> jac_rows.resize(43);
    this-> jac_rows << 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 11, 12, 13;
    this-> jac_rows += (rows_offset * Eigen::VectorXd::Ones(this ->jac_rows.size()) );

    this-> jac_cols.resize(43);
    this-> jac_cols << 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_crank*2, 
        this-> rbs_crank*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_crank*2, 
        this-> rbs_crank*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_crank*2, 
        this-> rbs_crank*2+1, 
        this-> rbs_crank*2, 
        this-> rbs_crank*2+1, 
        this-> rbs_conct*2, 
        this-> rbs_conct*2+1, 
        this-> rbs_conct*2, 
        this-> rbs_conct*2+1, 
        this-> rbs_rockr*2, 
        this-> rbs_rockr*2+1, 
        this-> rbs_conct*2, 
        this-> rbs_conct*2+1, 
        this-> rbs_rockr*2, 
        this-> rbs_rockr*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_rockr*2, 
        this-> rbs_rockr*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_rockr*2, 
        this-> rbs_rockr*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_rockr*2, 
        this-> rbs_rockr*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> ground*2, 
        this-> ground*2+1, 
        this-> rbs_crank*2+1, 
        this-> rbs_conct*2+1, 
        this-> rbs_rockr*2+1;
};


void Topology::set_initial_states()
{
    this-> set_gen_coordinates(this-> config.q);
    this-> set_gen_velocities(this-> config.qd);
    this-> q0 = this-> config.q;
};


void Topology::set_mapping(Dict_SI &indicies_map, Dict_SS &interface_map)
{
    std::string p = this-> prefix;

    this-> ground = indicies_map[p+"ground"];
    this-> rbs_crank = indicies_map[p+"rbs_crank"];
    this-> rbs_conct = indicies_map[p+"rbs_conct"];
    this-> rbs_rockr = indicies_map[p+"rbs_rockr"];

    
};




void Configuration::set_inital_configuration()
{
    this-> q.resize(28);
    this-> q << 
        this-> R_ground, 
        this-> P_ground, 
        this-> R_rbs_crank, 
        this-> P_rbs_crank, 
        this-> R_rbs_conct, 
        this-> P_rbs_conct, 
        this-> R_rbs_rockr, 
        this-> P_rbs_rockr;

    this-> qd.resize(28);
    this-> qd << 
        this-> Rd_ground, 
        this-> Pd_ground, 
        this-> Rd_rbs_crank, 
        this-> Pd_rbs_crank, 
        this-> Rd_rbs_conct, 
        this-> Pd_rbs_conct, 
        this-> Rd_rbs_rockr, 
        this-> Pd_rbs_rockr;
};

void Topology::set_gen_coordinates(Eigen::VectorXd &q)
{
    this-> R_ground << q.block(0,0, 3,1) ;
    this-> P_ground << q.block(3,0, 4,1) ;
    this-> R_rbs_crank << q.block(7,0, 3,1) ;
    this-> P_rbs_crank << q.block(10,0, 4,1) ;
    this-> R_rbs_conct << q.block(14,0, 3,1) ;
    this-> P_rbs_conct << q.block(17,0, 4,1) ;
    this-> R_rbs_rockr << q.block(21,0, 3,1) ;
    this-> P_rbs_rockr << q.block(24,0, 4,1) ;
};

void Topology::set_gen_velocities(Eigen::VectorXd &qd)
{
    this-> Rd_ground << qd.block(0,0, 3,1) ;
    this-> Pd_ground << qd.block(3,0, 4,1) ;
    this-> Rd_rbs_crank << qd.block(7,0, 3,1) ;
    this-> Pd_rbs_crank << qd.block(10,0, 4,1) ;
    this-> Rd_rbs_conct << qd.block(14,0, 3,1) ;
    this-> Pd_rbs_conct << qd.block(17,0, 4,1) ;
    this-> Rd_rbs_rockr << qd.block(21,0, 3,1) ;
    this-> Pd_rbs_rockr << qd.block(24,0, 4,1) ;
};

void Topology::set_gen_accelerations(Eigen::VectorXd &qdd)
{
    this-> Rdd_ground << qdd.block(0,0, 3,1) ;
    this-> Pdd_ground << qdd.block(3,0, 4,1) ;
    this-> Rdd_rbs_crank << qdd.block(7,0, 3,1) ;
    this-> Pdd_rbs_crank << qdd.block(10,0, 4,1) ;
    this-> Rdd_rbs_conct << qdd.block(14,0, 3,1) ;
    this-> Pdd_rbs_conct << qdd.block(17,0, 4,1) ;
    this-> Rdd_rbs_rockr << qdd.block(21,0, 3,1) ;
    this-> Pdd_rbs_rockr << qdd.block(24,0, 4,1) ;
};



void Topology::eval_constants()
{
    auto &config = this-> config;

    this-> Pg_ground << 1, 0, 0, 0 ;
    this-> m_ground = 1.0 ;
    this-> Jbar_ground << 1, 0, 0, 0, 1, 0, 0, 0, 1 ;
    this-> F_rbs_crank_gravity << 0, 0, -9810.0*config.m_rbs_crank ;
    this-> F_rbs_conct_gravity << 0, 0, -9810.0*config.m_rbs_conct ;
    this-> F_rbs_rockr_gravity << 0, 0, -9810.0*config.m_rbs_rockr ;

    this-> Mbar_ground_jcs_a << A(config.P_ground).transpose() * triad(config.ax1_jcs_a) ;
    this-> Mbar_rbs_crank_jcs_a << A(config.P_rbs_crank).transpose() * triad(config.ax1_jcs_a) ;
    this-> ubar_ground_jcs_a << (A(config.P_ground).transpose() * config.pt1_jcs_a + -1 * A(config.P_ground).transpose() * config.R_ground) ;
    this-> ubar_rbs_crank_jcs_a << (A(config.P_rbs_crank).transpose() * config.pt1_jcs_a + -1 * A(config.P_rbs_crank).transpose() * config.R_rbs_crank) ;
    this-> Mbar_rbs_crank_jcs_b << A(config.P_rbs_crank).transpose() * triad(config.ax1_jcs_b) ;
    this-> Mbar_rbs_conct_jcs_b << A(config.P_rbs_conct).transpose() * triad(config.ax1_jcs_b) ;
    this-> ubar_rbs_crank_jcs_b << (A(config.P_rbs_crank).transpose() * config.pt1_jcs_b + -1 * A(config.P_rbs_crank).transpose() * config.R_rbs_crank) ;
    this-> ubar_rbs_conct_jcs_b << (A(config.P_rbs_conct).transpose() * config.pt1_jcs_b + -1 * A(config.P_rbs_conct).transpose() * config.R_rbs_conct) ;
    this-> Mbar_rbs_conct_jcs_c << A(config.P_rbs_conct).transpose() * triad(config.ax1_jcs_c) ;
    this-> Mbar_rbs_rockr_jcs_c << A(config.P_rbs_rockr).transpose() * triad(config.ax2_jcs_c, triad(config.ax1_jcs_c).block(0,1, 3,1)) ;
    this-> ubar_rbs_conct_jcs_c << (A(config.P_rbs_conct).transpose() * config.pt1_jcs_c + -1 * A(config.P_rbs_conct).transpose() * config.R_rbs_conct) ;
    this-> ubar_rbs_rockr_jcs_c << (A(config.P_rbs_rockr).transpose() * config.pt1_jcs_c + -1 * A(config.P_rbs_rockr).transpose() * config.R_rbs_rockr) ;
    this-> Mbar_rbs_rockr_jcs_d << A(config.P_rbs_rockr).transpose() * triad(config.ax1_jcs_d) ;
    this-> Mbar_ground_jcs_d << A(config.P_ground).transpose() * triad(config.ax1_jcs_d) ;
    this-> ubar_rbs_rockr_jcs_d << (A(config.P_rbs_rockr).transpose() * config.pt1_jcs_d + -1 * A(config.P_rbs_rockr).transpose() * config.R_rbs_rockr) ;
    this-> ubar_ground_jcs_d << (A(config.P_ground).transpose() * config.pt1_jcs_d + -1 * A(config.P_ground).transpose() * config.R_ground) ;
};

void Topology::eval_pos_eq()
{
    auto &config = this-> config;
    auto &t = this-> t;

    Eigen::Vector3d x0 = this-> R_ground ;
    Eigen::Vector3d x1 = this-> R_rbs_crank ;
    Eigen::Vector4d x2 = this-> P_ground ;
    Eigen::Matrix<double, 3, 3> x3 = A(x2) ;
    Eigen::Vector4d x4 = this-> P_rbs_crank ;
    Eigen::Matrix<double, 3, 3> x5 = A(x4) ;
    Eigen::Matrix<double, 3, 3> x6 = x3.transpose() ;
    Eigen::Vector3d x7 = this-> Mbar_rbs_crank_jcs_a.col(2) ;
    Eigen::Vector3d x8 = this-> R_rbs_conct ;
    Eigen::Vector4d x9 = this-> P_rbs_conct ;
    Eigen::Matrix<double, 3, 3> x10 = A(x9) ;
    Eigen::Vector3d x11 = this-> R_rbs_rockr ;
    Eigen::Vector4d x12 = this-> P_rbs_rockr ;
    Eigen::Matrix<double, 3, 3> x13 = A(x12) ;
    Eigen::Matrix<double, 3, 3> x14 = x13.transpose() ;
    Eigen::Vector3d x15 = this-> Mbar_ground_jcs_d.col(2) ;
    Eigen::Matrix<double, 1, 1> x16 = -1 * Eigen::MatrixXd::Identity(1, 1) ;

    this-> pos_eq << 
        (x0 + -1 * x1 + x3 * this-> ubar_ground_jcs_a + -1 * x5 * this-> ubar_rbs_crank_jcs_a),
        this-> Mbar_ground_jcs_a.col(0).transpose() * x6 * x5 * x7,
        this-> Mbar_ground_jcs_a.col(1).transpose() * x6 * x5 * x7,
        (x1 + -1 * x8 + x5 * this-> ubar_rbs_crank_jcs_b + -1 * x10 * this-> ubar_rbs_conct_jcs_b),
        (x8 + -1 * x11 + x10 * this-> ubar_rbs_conct_jcs_c + -1 * x13 * this-> ubar_rbs_rockr_jcs_c),
        this-> Mbar_rbs_conct_jcs_c.col(0).transpose() * x10.transpose() * x13 * this-> Mbar_rbs_rockr_jcs_c.col(0),
        (x11 + -1 * x0 + x13 * this-> ubar_rbs_rockr_jcs_d + -1 * x3 * this-> ubar_ground_jcs_d),
        this-> Mbar_rbs_rockr_jcs_d.col(0).transpose() * x14 * x3 * x15,
        this-> Mbar_rbs_rockr_jcs_d.col(1).transpose() * x14 * x3 * x15,
        x0,
        (x2 + -1 * this-> Pg_ground),
        (x16 + x4.transpose() * x4),
        (x16 + x9.transpose() * x9),
        (x16 + x12.transpose() * x12);
};

void Topology::eval_vel_eq()
{
    auto &config = this-> config;
    auto &t = this-> t;

    Eigen::Vector3d v0 = Eigen::MatrixXd::Zero(3, 1) ;
    Eigen::Matrix<double, 1, 1> v1 = Eigen::MatrixXd::Zero(1, 1) ;

    this-> vel_eq << 
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v0,
        v1,
        v1,
        v0,
        Eigen::MatrixXd::Zero(4, 1),
        v1,
        v1,
        v1;
};

void Topology::eval_acc_eq()
{
    auto &config = this-> config;
    auto &t = this-> t;

    Eigen::Vector4d a0 = this-> Pd_ground ;
    Eigen::Vector4d a1 = this-> Pd_rbs_crank ;
    Eigen::Vector3d a2 = this-> Mbar_ground_jcs_a.col(0) ;
    Eigen::Vector4d a3 = this-> P_ground ;
    Eigen::Matrix<double, 3, 3> a4 = A(a3).transpose() ;
    Eigen::Vector3d a5 = this-> Mbar_rbs_crank_jcs_a.col(2) ;
    Eigen::Matrix<double, 3, 4> a6 = B(a1, a5) ;
    Eigen::Matrix<double, 1, 3> a7 = a5.transpose() ;
    Eigen::Vector4d a8 = this-> P_rbs_crank ;
    Eigen::Matrix<double, 3, 3> a9 = A(a8).transpose() ;
    Eigen::Matrix<double, 1, 4> a10 = a0.transpose() ;
    Eigen::Matrix<double, 3, 4> a11 = B(a8, a5) ;
    Eigen::Vector3d a12 = this-> Mbar_ground_jcs_a.col(1) ;
    Eigen::Vector4d a13 = this-> Pd_rbs_conct ;
    Eigen::Vector4d a14 = this-> Pd_rbs_rockr ;
    Eigen::Vector3d a15 = this-> Mbar_rbs_conct_jcs_c.col(0) ;
    Eigen::Vector4d a16 = this-> P_rbs_conct ;
    Eigen::Vector3d a17 = this-> Mbar_rbs_rockr_jcs_c.col(0) ;
    Eigen::Vector4d a18 = this-> P_rbs_rockr ;
    Eigen::Matrix<double, 3, 3> a19 = A(a18).transpose() ;
    Eigen::Matrix<double, 1, 4> a20 = a13.transpose() ;
    Eigen::Vector3d a21 = this-> Mbar_rbs_rockr_jcs_d.col(0) ;
    Eigen::Vector3d a22 = this-> Mbar_ground_jcs_d.col(2) ;
    Eigen::Matrix<double, 3, 4> a23 = B(a0, a22) ;
    Eigen::Matrix<double, 1, 3> a24 = a22.transpose() ;
    Eigen::Matrix<double, 1, 4> a25 = a14.transpose() ;
    Eigen::Matrix<double, 3, 4> a26 = B(a3, a22) ;
    Eigen::Vector3d a27 = this-> Mbar_rbs_rockr_jcs_d.col(1) ;

    this-> acc_eq << 
        (B(a0, this-> ubar_ground_jcs_a) * a0 + -1 * B(a1, this-> ubar_rbs_crank_jcs_a) * a1),
        (a2.transpose() * a4 * a6 * a1 + a7 * a9 * B(a0, a2) * a0 + 2 * a10 * B(a3, a2).transpose() * a11 * a1),
        (a12.transpose() * a4 * a6 * a1 + a7 * a9 * B(a0, a12) * a0 + 2 * a10 * B(a3, a12).transpose() * a11 * a1),
        (B(a1, this-> ubar_rbs_crank_jcs_b) * a1 + -1 * B(a13, this-> ubar_rbs_conct_jcs_b) * a13),
        (B(a13, this-> ubar_rbs_conct_jcs_c) * a13 + -1 * B(a14, this-> ubar_rbs_rockr_jcs_c) * a14),
        (a15.transpose() * A(a16).transpose() * B(a14, a17) * a14 + a17.transpose() * a19 * B(a13, a15) * a13 + 2 * a20 * B(a16, a15).transpose() * B(a18, a17) * a14),
        (B(a14, this-> ubar_rbs_rockr_jcs_d) * a14 + -1 * B(a0, this-> ubar_ground_jcs_d) * a0),
        (a21.transpose() * a19 * a23 * a0 + a24 * a4 * B(a14, a21) * a14 + 2 * a25 * B(a18, a21).transpose() * a26 * a0),
        (a27.transpose() * a19 * a23 * a0 + a24 * a4 * B(a14, a27) * a14 + 2 * a25 * B(a18, a27).transpose() * a26 * a0),
        Eigen::MatrixXd::Zero(3, 1),
        Eigen::MatrixXd::Zero(4, 1),
        2 * a1.transpose() * a1,
        2 * a20 * a13,
        2 * a25 * a14;
};

void Topology::eval_jac_eq()
{
    auto &config = this-> config;
    auto &t = this-> t;

    Eigen::Matrix<double, 3, 3> j0 = Eigen::MatrixXd::Identity(3, 3) ;
    Eigen::Vector4d j1 = this-> P_ground ;
    Eigen::Matrix<double, 1, 3> j2 = Eigen::MatrixXd::Zero(1, 3) ;
    Eigen::Vector3d j3 = this-> Mbar_rbs_crank_jcs_a.col(2) ;
    Eigen::Matrix<double, 1, 3> j4 = j3.transpose() ;
    Eigen::Vector4d j5 = this-> P_rbs_crank ;
    Eigen::Matrix<double, 3, 3> j6 = A(j5).transpose() ;
    Eigen::Vector3d j7 = this-> Mbar_ground_jcs_a.col(0) ;
    Eigen::Vector3d j8 = this-> Mbar_ground_jcs_a.col(1) ;
    Eigen::Matrix<double, 3, 3> j9 = -1 * j0 ;
    Eigen::Matrix<double, 3, 3> j10 = A(j1).transpose() ;
    Eigen::Matrix<double, 3, 4> j11 = B(j5, j3) ;
    Eigen::Vector4d j12 = this-> P_rbs_conct ;
    Eigen::Vector3d j13 = this-> Mbar_rbs_rockr_jcs_c.col(0) ;
    Eigen::Vector4d j14 = this-> P_rbs_rockr ;
    Eigen::Matrix<double, 3, 3> j15 = A(j14).transpose() ;
    Eigen::Vector3d j16 = this-> Mbar_rbs_conct_jcs_c.col(0) ;
    Eigen::Vector3d j17 = this-> Mbar_ground_jcs_d.col(2) ;
    Eigen::Matrix<double, 1, 3> j18 = j17.transpose() ;
    Eigen::Vector3d j19 = this-> Mbar_rbs_rockr_jcs_d.col(0) ;
    Eigen::Vector3d j20 = this-> Mbar_rbs_rockr_jcs_d.col(1) ;
    Eigen::Matrix<double, 3, 4> j21 = B(j1, j17) ;

    this-> jac_eq << 
        j0,
        B(j1, this-> ubar_ground_jcs_a),
        j9,
        -1 * B(j5, this-> ubar_rbs_crank_jcs_a),
        j2,
        j4 * j6 * B(j1, j7),
        j2,
        j7.transpose() * j10 * j11,
        j2,
        j4 * j6 * B(j1, j8),
        j2,
        j8.transpose() * j10 * j11,
        j0,
        B(j5, this-> ubar_rbs_crank_jcs_b),
        j9,
        -1 * B(j12, this-> ubar_rbs_conct_jcs_b),
        j0,
        B(j12, this-> ubar_rbs_conct_jcs_c),
        j9,
        -1 * B(j14, this-> ubar_rbs_rockr_jcs_c),
        j2,
        j13.transpose() * j15 * B(j12, j16),
        j2,
        j16.transpose() * A(j12).transpose() * B(j14, j13),
        j9,
        -1 * B(j1, this-> ubar_ground_jcs_d),
        j0,
        B(j14, this-> ubar_rbs_rockr_jcs_d),
        j2,
        j19.transpose() * j15 * j21,
        j2,
        j18 * j10 * B(j14, j19),
        j2,
        j20.transpose() * j15 * j21,
        j2,
        j18 * j10 * B(j14, j20),
        j0,
        Eigen::MatrixXd::Zero(3, 4),
        Eigen::MatrixXd::Zero(4, 3),
        Eigen::MatrixXd::Identity(4, 4),
        2 * j5.transpose(),
        2 * j12.transpose(),
        2 * j14.transpose();
};


