#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <map>

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

private:
    std::map<int, std::vector<Eigen::VectorXd>*> results;
    std::map<int, std::string> results_names;

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

        results[0] =  &this-> pos_history;
        results_names[0] = "Positions";

        results[1] =  &this-> vel_history;
        results_names[1] = "Velocities";

        results[2] =  &this-> acc_history;
        results_names[2] = "Accelerations";
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

    void ExportResultsCSV(std::string location = "", int id = 0);
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

    //std::cout << "Solving for Vector b " << "\n";
    Eigen::VectorXd error = SparseSolver.solve(-b);

    //std::cout << "Entring While Loop " << "\n";
    int itr = 0;
    while (error.norm() >= 1e-5)
    {
        //std::cout << "Error e = " << error.norm() << "\n";
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
            std::cout << "ITERATIONS EXCEEDED!! => " << itr << "\n" ;
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

    //std::cout << "Setting Initial Position History" << "\n";
    this-> pos_history.emplace_back(this-> model_ptr-> q0);

    Eigen::VectorXd v0, vi, a0, ai, guess;

    this-> pos_history.reserve(samples);
    this-> vel_history.reserve(samples);
    this-> acc_history.reserve(samples);

    //std::cout << "Computing Jacobian" << "\n";
    auto A = this-> eval_jac_eq();
    SparseSolver.compute(A);

    //std::cout << "Solving for Velocity" << "\n";
    v0 = SparseSolver.solve(-this-> eval_vel_eq());
    //std::cout << "Setting Generalized Velocities" << "\n";
    this-> set_gen_velocities(v0);
    //std::cout << "Storing Generalized Velocities" << "\n";
    this-> vel_history.emplace_back(v0);
    
    //std::cout << "Solving for Accelerations" << "\n";
    a0 = SparseSolver.solve(-this-> eval_acc_eq());
    this-> acc_history.emplace_back(a0);

    //std::cout << "\nRunning System Kinematic Analysis: " << "\n";
    const int barWidth = 50;
    for (size_t i = 1; i < samples; i++)
    {
        std::cout << "[";
        int pos = barWidth * i/samples ;
        for (int p = 0; p < barWidth; ++p) 
        {
            if (p < pos) std::cout << "=" ;
            else if (p == pos) std::cout << ">" ;
            else std::cout << " " ;
        }
        std::cout << "] " << i << " \r" ;
        std::cout.flush() ;

        //std::cout << "Simulation Steps : " << i << " \r" ;
        //std::cout.flush() ;

        t = time_array(i) ;
        this-> set_time(t) ;

        guess = this-> pos_history[i-1]
              + this-> vel_history[i-1] * dt
              + 0.5 * this-> acc_history[i-1] * pow(dt, 2);

        this-> NewtonRaphson(guess);
        this-> pos_history.emplace_back(this-> q);

        A = this-> eval_jac_eq();
        SparseSolver.compute(A);

        vi = SparseSolver.solve(-this-> eval_vel_eq());
        this-> set_gen_velocities(vi);
        this-> vel_history.emplace_back(vi);

        ai = SparseSolver.solve(-this-> eval_acc_eq());
        this-> acc_history.emplace_back(ai);
    };
    std::cout << "\n";
};


const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", ",");

template<class T>
void Solver<T>::ExportResultsCSV(std::string location, int id)
{
    // declaring and initializing the needed variables
    auto& name = this-> results_names[id];
    auto& results = *(this-> results[id]);
    auto& model = *this-> model_ptr;
    std::ofstream results_file;

    std::map<int, std::string> ordered_indicies;
    for (auto x : model.indicies_map)
    {
        ordered_indicies[x.second] = x.first;
    };

    // Creating the system indicies string to be used as the fisrt line
    // in the .csv file
    std::string indicies = "";
    std::vector<std::string> coordinates{"x", "y", "z", "e0", "e1", "e2", "e3"};
    for (auto x : ordered_indicies)
    {
        auto body_name = x.second;
        for (auto& coordinate : coordinates)
        {
          indicies += body_name + "." + coordinate + "," ;
        };
    };

    // Opening the file as a .csv file.
    results_file.open (location + name + ".csv");
    
    // Inserting the first line to be the indicies of the system.
    results_file << indicies + "time\n";

    // Looping over the results and writing each line to the .csv file.
    int i = 0;
    for (auto x : results)
    {
        results_file << x.transpose().format(CSVFormat) ;
        results_file << std::to_string(this-> time_array(i)) + "\n";
        i += 1;
    };

    results_file.close();
    std::cout << "\n" << name << " results saved as : " << location + name + ".csv" << "\n";
    
};

