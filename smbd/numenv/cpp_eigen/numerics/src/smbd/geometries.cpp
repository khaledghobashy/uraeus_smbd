
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


triangular_prism::triangular_prism(Eigen::Vector3d& p1, Eigen::Vector3d& p2, Eigen::Vector3d& p3, double thickness)
{
    
    double l1, l2, l3, pr, theta, height, area, volume, density;

    Eigen::Vector3d v1 = p1 - p2 ;
    Eigen::Vector3d v2 = p1 - p3 ;
    Eigen::Vector3d v3 = p2 - p3 ;

    l1 = v1.norm();
    l2 = v2.norm();
    l3 = v3.norm();

    pr = (l1 + l2 + l3) / 2 ;

    double cos_theta = v1.transpose() * v2;
    cos_theta = cos_theta / (l1*l2) ;
    theta = std::acos(cos_theta);
    height = l2 * std::sin(theta);
    area = std::sqrt(pr*(pr - l1)*(pr-l2)*(pr-l3)) ;
    volume = area * height ;
    density = 7.9 ;

    Eigen::Vector3d v = oriented({p1, p2, p3});
    Eigen::Matrix3d frame = triad(v, v1) ;

    double a = v2.transpose() * v1 ;

    double Ixc = (l1*std::pow(height,3)) / 36 ;
    double Iyc = ((std::pow(l1,3)*height)-(std::pow(l1,2)*height*a)+(l1*height*std::pow(a,2))) / 36 ;
    double Izc = ((std::pow(l1,3)*height)-(std::pow(l1,2)*height*a)+(l1*height*std::pow(a,2))+(l1*std::pow(height,3))) / 36 ;

    double mass = density * volume * 1e-3;
    Eigen::Vector4d P = A2P(frame);
    Eigen::DiagonalMatrix<double, 3> J(Ixc, Iyc, Izc);
    J = mass*J;
    
    this-> R = centered({p1, p2, p3});
    this-> P << 1, 0, 0, 0 ;
    this-> J = A(P).transpose() * J * A(P) ;
    this-> m = mass ;

};

