
#include "geometries.hpp"
#include "spatial_algebra.hpp"
#include "euler_parameters.hpp"


cylinder_geometry::cylinder_geometry(Eigen::Vector3d& p1, Eigen::Vector3d& p2, double& ro, double ri)
{   
    Eigen::Vector3d axis = p2 - p1 ;
    auto frame = triad(axis);
    
    const double l = (p2 - p1).norm();
    const double vol = (22/7)*(std::pow(ro, 2) - std::pow(ri,2)) * l * 1e-3;

    double m = 7.9/vol ;
    const double Jzz = (m/2)*(std::pow(ro,2) + std::pow(ri,2));
    const double Jyy = (m/12)*(3*(std::pow(ro,2) + std::pow(ri,2)) + std::pow(l,2));
    const double Jxx = Jyy ;

    Eigen::Vector4d P = A2P(frame);
    Eigen::DiagonalMatrix<double, 3> J(Jxx, Jyy, Jzz);

    this-> R = centered({p1, p2});
    this-> P << 1, 0, 0, 0 ;
    this-> J = A(P).transpose() * J * A(P) ;
    this-> m = m ;

};
