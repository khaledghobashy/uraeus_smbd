#include <vector>
#include <iostream>
#include "helpers.hpp"


// Declaring and Implementing the Solver class as a template class.
// Type T should be any Topology class type.
template <class T>
class Solver
{
public:
    T* model_ptr; // Intial pointer to the model object.

    double t;
    double step_size;
    Eigen::VectorXd time_array;

    std::vector<Eigen::VectorXd> pos_history;
    std::vector<Eigen::VectorXd> vel_history;
    std::vector<Eigen::VectorXd> acc_history;

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

        for (size_t i = 0; i < model.jac_rows.size(); i++)
        {
            jac_rows.push_back(int(model.jac_rows(i)));
            jac_cols.push_back(int(model.jac_cols(i)));        
        }

    };

    void set_time(double& t);
    void set_time_array(const double& duration, const double& spacing);

    Eigen::VectorXd eval_pos_eq();
    Eigen::VectorXd eval_vel_eq();
    Eigen::VectorXd eval_acc_eq();
    SparseBlock eval_jac_eq();

    void set_gen_coordinates(Eigen::VectorXd& q);
    void set_gen_velocities(Eigen::VectorXd& qd);
    void set_gen_accelerations(Eigen::VectorXd& qdd);

public:
    void NewtonRaphson(Eigen::VectorXd &guess);
    void Solve();
};


/** 
 * CLASS METHODS IMPLEMENTATION
 * ============================
*/

template<class T>
void Solver<T>::set_time(double& t)
{
    this-> model_ptr-> t = t;
};

template<class T>
void Solver<T>::set_time_array(const double& duration, const double& spacing)
{
    if (duration > spacing)
    {   
        double steps = duration/spacing;
        this-> time_array = Eigen::VectorXd::LinSpaced(steps, 0, duration);
        this-> step_size = spacing;
    }
    else if (duration < spacing)
    {
        this-> time_array = Eigen::VectorXd::LinSpaced(spacing, 0, duration);
        this-> step_size = duration/spacing;
    }
};

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
    //std::cout << "Declaring SparseLU" << "\n";
    Eigen::SparseLU<SparseBlock> SparseSolver;

    //std::cout << "Setting Guess" << "\n";
    this-> set_gen_coordinates(guess);
    //std::cout << "Evaluating Pos_Eq " << "\n";
    auto b = this-> eval_pos_eq();
    //std::cout << "Evaluating Jac_Eq " << "\n";
    auto A = this-> eval_jac_eq();
    //std::cout << "Computing Matrix A " << "\n";
    SparseSolver.compute(A);
    std::cout << "Solver Status : " << SparseSolver.info() << "\n";
    /* SparseSolver.analyzePattern(A);
    if(SparseSolver.info() == Eigen::Success){std::cout << "Done Analyze" << "\n";};
    std::cout << SparseSolver.info() << "\n";
    SparseSolver.factorize(A); */

    //std::cout << "Solving for Vector b " << "\n";
    Eigen::VectorXd error = SparseSolver.solve(-b);

    //std::cout << "Entring While Loop " << "\n";
    int itr = 0;
    while (error.norm() >= 1e-5)
    {
        std::cout << "Error e = " << error.norm() << "\n";
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


template<class T>
void Solver<T>::Solve()
{
    std::cout << "Starting Solver ..." << "\n";
    Eigen::SparseLU<SparseBlock> SparseSolver;
    auto& time_array = this-> time_array;
    auto& dt = this-> step_size;
    double t = 0;
    auto samples = time_array.size();

    std::cout << "Setting Initial Position History" << "\n";
    this-> pos_history.emplace_back(this-> model_ptr-> q0);

    Eigen::VectorXd vi;
    Eigen::VectorXd ai;
    Eigen::VectorXd guess;

    this-> pos_history.reserve(samples);
    this-> vel_history.reserve(samples);
    this-> acc_history.reserve(samples);

    std::cout << "Computing Jacobian" << "\n";
    auto A = this-> eval_jac_eq();
    SparseSolver.compute(A);

    std::cout << "Solving for Velocity" << "\n";
    Eigen::VectorXd v0 = SparseSolver.solve(-this-> eval_vel_eq());
    std::cout << "Setting Generalized Velocities" << "\n";
    this-> set_gen_velocities(v0);
    std::cout << "Storing Generalized Velocities" << "\n";
    this-> vel_history.emplace_back(v0);
    
    std::cout << "Solving for Accelerations" << "\n";
    Eigen::VectorXd a0 = SparseSolver.solve(-this-> eval_acc_eq());
    this-> acc_history.emplace_back(a0);

    std::cout << "\nRunning System Kinematic Analysis: " << "\n";
    for (size_t i = 0; i < samples; i++)
    {
        t = time_array(i);
        this-> set_time(t);

        guess = this-> pos_history[i]
              + this-> vel_history[i] * dt
              + 0.5 * this-> acc_history[i] * pow(dt,2);

        this-> NewtonRaphson(guess);
        this-> pos_history.emplace_back(this-> q);

        A = this-> eval_jac_eq();
        SparseSolver.compute(A);

        vi = SparseSolver.solve(-this-> eval_vel_eq());
        this-> set_gen_velocities(vi);
        this-> vel_history.emplace_back(vi);

        ai = SparseSolver.solve(-this-> eval_acc_eq());
        this-> acc_history.emplace_back(ai);

    }
    
};
