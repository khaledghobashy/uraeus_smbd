#include <vector>
#include <iostream>

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"
#include "eigen-eigen-323c052e1731/Eigen/Sparse"


typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseBlock;
typedef std::vector<int> Indicies;
typedef std::vector<Eigen::MatrixXd> DataBlocks;

void SparseAssembler(SparseBlock& mat, Indicies& rows, Indicies& cols, DataBlocks& data);

template <class T>
class Solver
{
public:
    T model_;
    SparseBlock Jacobian;
    std::vector<int> jac_rows;
    std::vector<int> jac_cols;

public:
    
    Solver(T& model)
    {
        this-> model_ = model;
        model.initialize();
        model.set_gen_coordinates(model.q0);
        Jacobian.resize(model.nc, model.n);

        for (size_t i = 0; i < model.nc; i++)
        {
            jac_rows.push_back(int(model.jac_rows(i)));
            jac_cols.push_back(int(model.jac_cols(i)));        
        }

    };

    Eigen::VectorXd eval_pos_eq();
    Eigen::VectorXd eval_vel_eq();
    Eigen::VectorXd eval_acc_eq();
    SparseBlock eval_jac_eq();

    void set_gen_coordinates(Eigen::VectorXd &q);
    void set_gen_velocities(Eigen::VectorXd &qd);
    void set_gen_accelerations(Eigen::VectorXd &qdd);
};


/** 
 * CLASS METHODS IMPLEMENTATION
 * ============================
*/

template<class T>
void Solver<T>::set_gen_coordinates(Eigen::VectorXd &q)
{   
    this-> model_.set_gen_coordinates(q);
};

template<class T>
void Solver<T>::set_gen_velocities(Eigen::VectorXd &qd)
{   
    this-> model_.set_gen_velocities(qd);
};

template<class T>
void Solver<T>::set_gen_accelerations(Eigen::VectorXd &qdd)
{   
    this-> model_.set_gen_accelerations(qdd);
};


template<class T>
Eigen::VectorXd Solver<T>::eval_pos_eq()
{   
    std::cout << "Pos_Eq Size = " << this-> model_.pos_eq.size() << "\n";
    this-> model_.eval_pos_eq();
    return this-> model_.pos_eq;
};

template<class T>
Eigen::VectorXd Solver<T>::eval_vel_eq()
{   
    this-> model_.eval_vel_eq();
    return this-> model_.vel_eq;
};

template<class T>
Eigen::VectorXd Solver<T>::eval_acc_eq()
{   
    this-> model_.eval_acc_eq();
    return this-> model_.acc_eq;
};


template<class T>
SparseBlock Solver<T>::eval_jac_eq()
{   
    this-> model_.eval_jac_eq();
    SparseAssembler(this-> Jacobian, this-> jac_rows, this-> jac_cols, this-> model_.jac_eq);
    return this-> Jacobian;
};



