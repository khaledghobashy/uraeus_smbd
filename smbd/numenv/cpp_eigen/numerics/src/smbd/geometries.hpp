#pragma once

#include <eigen/Eigen/Dense>
#include <vector>
#include <numeric>

class cylinder_geometry
{
public:
    Eigen::Vector3d R ;
    Eigen::Vector4d P ;
    Eigen::Matrix3d J ;
    Eigen::Matrix3d I ;
    double m;

    cylinder_geometry(Eigen::Vector3d& p1, Eigen::Vector3d& p2, double& ro, double ri = 0);
};



class triangular_prism
{
public:
    Eigen::Vector3d R ;
    Eigen::Vector4d P ;
    Eigen::Matrix3d J ;
    Eigen::Matrix3d I ;
    double m;

    triangular_prism(Eigen::Vector3d& p1, Eigen::Vector3d& p2, Eigen::Vector3d& p3, double thickness);
};
