
#include "euler_parameters.hpp"


Eigen::Vector4d A2P(Eigen::Matrix3d& dcm)
{
    double e0s, e1s, e2s, e3s, emax, e0, e1, e2, e3 ;
    Eigen::Vector4d P;
    const double trace = dcm.trace();

    e0s = (trace + 1) / 4 ;
    e1s = (2*dcm(0,0) - trace + 1) / 4 ;
    e2s = (2*dcm(1,1) - trace + 1) / 4 ;
    e3s = (2*dcm(2,2) - trace + 1) / 4 ;

    emax = std::max({e0s, e1s, e2s, e3s});

    if (e0s == emax)
    {
        e0 = std::abs(std::sqrt(e0s)) ;
        e1 = (dcm(2,1) - dcm(1,2)) / (4*e0) ;
        e2 = (dcm(0,2) - dcm(2,0)) / (4*e0) ;
        e3 = (dcm(0,1) - dcm(1,0)) / (4*e0) ;
    }    
    else if (e1s == emax)
    {
        e1 = std::abs(std::sqrt(e1s)) ;
        e0 = (dcm(2,1) - dcm(1,2)) / (4*e1) ;
        e2 = (dcm(0,1) + dcm(1,0)) / (4*e1) ;
        e3 = (dcm(0,2) + dcm(2,0)) / (4*e1) ;
    
    }
    else if (e2s == emax)
    {
        e2 = std::abs(std::sqrt(e2s)) ;
        e0 = (dcm(0,2) - dcm(2,0)) / (4*e2) ;
        e1 = (dcm(0,1) + dcm(1,0)) / (4*e2) ;
        e3 = (dcm(2,1) + dcm(1,2)) / (4*e2) ;
    
    }
    else if (e3s == emax)
    {
        e3 = std::abs(std::sqrt(e3s)) ;
        e0 = (dcm(0,1) - dcm(1,0)) / (4*e3) ;
        e1 = (dcm(0,2) + dcm(2,0)) / (4*e3) ;
        e2 = (dcm(2,1) + dcm(1,2)) / (4*e3) ;
    };

    P << e0, e1, e2, e3 ;
    return P ;
};



Eigen::Matrix3d A(const Eigen::Vector4d& P)
{
    double e0 = P(0,0);
    double e1 = P(1,0);
    double e2 = P(2,0);
    double e3 = P(3,0);

    Eigen::Matrix3d A;

    /**
     *  
    result[0,0] = (e0**2+e1**2-e2**2-e3**2)
    result[0,1] = 2*((e1*e2)-(e0*e3))              
    result[0,2] = 2*((e1*e3)+(e0*e2))
    
    result[1,0] = 2*((e1*e2)+(e0*e3))
    result[1,1] = e0**2-e1**2+e2**2-e3**2
    result[1,2] = 2*((e2*e3)-(e0*e1))
    
    result[2,0] = 2*((e1*e3)-(e0*e2))
    result[2,1] = 2*((e2*e3)+(e0*e1))
    result[2,2] = e0**2-e1**2-e2**2+e3**2
     * 
     */

    A(0,0) = pow(e0,2) + pow(e1,2) - pow(e2,2) - pow(e3,2);
    A(0,1) = 2*((e1*e2) - (e0*e3));
    A(0,2) = 2*((e1*e3) + (e0*e2));
    
    A(1,0) = 2*((e1*e2) + (e0*e3));
    A(1,1) = pow(e0,2) - pow(e1,2) + pow(e2,2) - pow(e3,2);
    A(1,2) = 2*((e2*e3) - (e0*e1));
    
    A(2,0) = 2*((e1*e3) - (e0*e2));
    A(2,1) = 2*((e2*e3) + (e0*e1));
    A(2,2) = pow(e0,2) - pow(e1,2) - pow(e2,2) + pow(e3,2);

    return A;
};


Eigen::Matrix<double, 3, 4> B(const Eigen::Vector4d& P, const Eigen::Vector3d& u)
{
    Eigen::Matrix<double, 3, 4> mat;

    double e0 = P(0,0);
    double e1 = P(1,0);
    double e2 = P(2,0);
    double e3 = P(3,0);
    
    double ux = u(0,0);
    double uy = u(1,0);
    double uz = u(2,0);
    
    mat(0,0) = 2*e0*ux + 2*e2*uz - 2*e3*uy ;
    mat(0,1) = 2*e1*ux + 2*e2*uy + 2*e3*uz ;
    mat(0,2) = 2*e0*uz + 2*e1*uy - 2*e2*ux ;
    mat(0,3) = -2*e0*uy + 2*e1*uz - 2*e3*ux ;
    
    mat(1,0) = 2*e0*uy - 2*e1*uz + 2*e3*ux ;
    mat(1,1) = -2*e0*uz - 2*e1*uy + 2*e2*ux ;
    mat(1,2) = 2*e1*ux + 2*e2*uy + 2*e3*uz ;
    mat(1,3) = 2*e0*ux + 2*e2*uz - 2*e3*uy ;
    
    mat(2,0) = 2*e0*uz + 2*e1*uy - 2*e2*ux ;
    mat(2,1) = 2*e0*uy - 2*e1*uz + 2*e3*ux ;
    mat(2,2) = -2*e0*ux - 2*e2*uz + 2*e3*uy ;
    mat(2,3) = 2*e1*ux + 2*e2*uy + 2*e3*uz ;

    return mat;

};


