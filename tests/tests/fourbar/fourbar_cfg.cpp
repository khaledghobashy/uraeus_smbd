
#include "fourbar_cfg.hpp"


void Configuration::assemble()
{
    auto &config = this->config;

    config.ax1_jcs_a << this-> vcs_x ;
    config.pt1_jcs_a << this-> hps_a ;
    config.ax1_jcs_b << this-> vcs_z ;
    config.pt1_jcs_b << this-> hps_b ;
    config.ax1_jcs_c << oriented({this-> hps_b, this-> hps_c}) ;
    config.ax2_jcs_c << oriented({this-> hps_c, this-> hps_b}) ;
    config.pt1_jcs_c << this-> hps_c ;
    config.ax1_jcs_d << this-> vcs_y ;
    config.pt1_jcs_d << this-> hps_d ;
};


