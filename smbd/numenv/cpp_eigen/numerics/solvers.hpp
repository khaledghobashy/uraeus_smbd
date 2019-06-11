#include <vector>
#include <iostream>

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"
#include "eigen-eigen-323c052e1731/Eigen/Sparse"
#include "eigen-eigen-323c052e1731/Eigen/SparseLU"


typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseBlock;
typedef std::vector<int> Indicies;
typedef std::vector<Eigen::MatrixXd> DataBlocks;

// Declaring helper functions.
void SparseAssembler(SparseBlock& mat, Indicies& rows, Indicies& cols, DataBlocks& data);


// Declaring and Implementing the Solver class as a template class.
// Type T should be any Topology class type.
template <class T>
class Solver
{
public:
    T* model_ptr; // Intial pointer to the model object.
    SparseBlock Jacobian;
    std::vector<int> jac_rows;
    std::vector<int> jac_cols;

    Eigen::VectorXd q;

public:

    Solver(T& model)
    {
        this-> model_ptr = &model;
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

public:
    void NewtonRaphson(Eigen::VectorXd &guess);
};


/** 
 * CLASS METHODS IMPLEMENTATION
 * ============================
*/

template<class T>
void Solver<T>::set_gen_coordinates(Eigen::VectorXd &q)
{   
    auto& model = *this-> model_ptr;
    model.set_gen_coordinates(q);
};

template<class T>
void Solver<T>::set_gen_velocities(Eigen::VectorXd &qd)
{   
    auto& model = *this-> model_ptr;
    model.set_gen_velocities(qd);
};

template<class T>
void Solver<T>::set_gen_accelerations(Eigen::VectorXd &qdd)
{   
    auto& model = *this-> model_ptr;
    model.set_gen_accelerations(qdd);
};


template<class T>
Eigen::VectorXd Solver<T>::eval_pos_eq()
{   
    auto& model = *this-> model_ptr;
    model.eval_pos_eq();
    return model.pos_eq;
};

template<class T>
Eigen::VectorXd Solver<T>::eval_vel_eq()
{   
    auto& model = *this-> model_ptr;
    model.eval_vel_eq();
    return model.vel_eq;
};

template<class T>
Eigen::VectorXd Solver<T>::eval_acc_eq()
{   
    auto& model = *this-> model_ptr;
    model.eval_acc_eq();
    return model.acc_eq;
};


template<class T>
SparseBlock Solver<T>::eval_jac_eq()
{   
    auto& model = *this-> model_ptr;
    model.eval_jac_eq();
    SparseAssembler(this-> Jacobian, this-> jac_rows, this-> jac_cols, model.jac_eq);
    return this-> Jacobian;
};


template<class T>
void Solver<T>::NewtonRaphson(Eigen::VectorXd &guess)
{
    auto& model = *this-> model_ptr;
    
    // Creating a SparseSolver object.
    Eigen::SparseLU<SparseBlock> SparseSolver;

    this-> set_gen_coordinates(guess);
    auto b = this-> eval_pos_eq();
    auto A = this-> eval_jac_eq();
    SparseSolver.compute(A);
    Eigen::VectorXd error = SparseSolver.solve(-b);

    int itr = 0;
    while (error.norm() >= 1e-5)
    {
        guess += error;
        this-> set_gen_coordinates(guess);
        b = this-> eval_pos_eq();
        error = SparseSolver.solve(-b);

        if (itr%5 == 0 && itr!=0)
        {
            A = this-> eval_jac_eq();
            SparseSolver.compute(A);
            error = SparseSolver.solve(-b);
        };

        if (itr>50)
        {
            break;
        };

        itr++;
    };
    
    this-> q = guess;
};

