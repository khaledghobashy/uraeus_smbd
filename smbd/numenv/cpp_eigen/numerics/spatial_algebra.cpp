#include </home/khaledghobashy/Documents/eigen-eigen-323c052e1731/Eigen/Dense>
#include <iostream>
#include <vector>
#include <numeric>
#include "spatial_algebra.hpp"


Eigen::Matrix3d skew(Eigen::Vector3d &v)
{
    auto &x = v(0);
    auto &y = v(1);
    auto &z = v(2);

    Eigen::Matrix3d mat;
    mat <<
         0, -z,  y,
         z,  0, -x,
        -y,  x,  0;
    
    return mat;
};


Eigen::Vector3d orthogonal_vector(Eigen::Vector3d &v1)
{
    Eigen::VectorXd dummy = Eigen::VectorXd::Ones(3,1);
    Eigen::VectorXd::Index max_index;
    Eigen::VectorXd::Index i = v1.maxCoeff(&max_index);
    dummy(i) = 0;

    Eigen::Vector3d v = (skew(v1) * dummy).normalized();
    return v;

};


Eigen::Matrix3d triad(Eigen::Vector3d &v1, Eigen::Vector3d &v2)
{
    Eigen::Vector3d k = v1.normalized();
    Eigen::Vector3d i = v2.normalized();
    Eigen::Vector3d j = k.cross(i);

    Eigen::Matrix3d mat;
    mat.col(0) = i;
    mat.col(1) = j;
    mat.col(2) = k;

    return mat;
};

template <typename T>
Eigen::Matrix3d triad(Eigen::Vector3d &v1, T const &v2)
{
    Eigen::Vector3d v2_ = v2;
    return triad(v1, v2_);

};

Eigen::Matrix3d triad(Eigen::Vector3d &v1)
{
    Eigen::Vector3d k = v1.normalized();
    Eigen::Vector3d i = orthogonal_vector(k).normalized();
    Eigen::Vector3d j = k.cross(i);

    Eigen::Matrix3d mat;
    mat.col(0) = i;
    mat.col(1) = j;
    mat.col(2) = k;

    return mat;

};

Eigen::Matrix3d triad(Eigen::Vector3d &v1, Col const &v2)
{
    Eigen::Vector3d k = v1.normalized();
    Eigen::Vector3d i = orthogonal_vector(k).normalized();
    Eigen::Vector3d j = k.cross(i);

    Eigen::Matrix3d mat;
    mat.col(0) = i;
    mat.col(1) = j;
    mat.col(2) = k;

    return mat;
};

Eigen::Vector3d mirrored(Eigen::Vector3d &v)
{
    auto v2 = v;
    v2(1) = -1*v(1);
    return v2;
};


Eigen::Vector3d centered(std::vector<Eigen::Vector3d> const &args)
{
    Eigen::Vector3d sum;
    sum << Eigen::VectorXd::Zero(3);
    for (auto x : args) sum += x;
    sum = sum/args.size();
    return sum;
};

Eigen::Vector3d oriented(std::vector<Eigen::Vector3d> const &args)
{
    Eigen::Vector3d v;

    if (args.size() == 2)
    {
        v = args[1] - args[0];
    }
    else
    {
        Eigen::Vector3d v1, v2;
        v1 = args[1] - args[0];
        v2 = args[2] - args[0];
        v  = v2.cross(v1);
    };

    return v;
};



/* int main()
{
    Eigen::Vector3d v1, v2, R;
    v1 << 1,1,1;
    v2 << 1,2,3;

    std::cout << "Skew : " << skew(v1) << "\n\n";
    std::cout << "Orthogonal : " << orthogonal_vector(v1) << "\n\n";
    std::cout << "Triad : " << triad(v1) << "\n\n";
    std::cout << "Triad : " << triad(v1, v2) << "\n\n";

    std::cout << "V1 : " << v1 << "\n";
    std::cout << "V2 : " << v2 << "\n\n";

    std::cout << "Centered : " << centered({v1, v2}) << "\n\n";
    std::cout << "Oriented : " << oriented({v1, v2}) << "\n\n";

    R << oriented({v1, v2});
    std::cout << "R << Oriented : " << oriented({v1, v2}) << "\n\n";

}
 */