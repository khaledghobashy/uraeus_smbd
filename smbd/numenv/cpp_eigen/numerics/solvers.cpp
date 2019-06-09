#include "solvers.hpp"

SparseBlock SparseAssembler(const int mrows, const int ncols, Indicies& rows, Indicies& cols, DataBlocks& data)
{
    //std::cout << "Declaring Local Variables" << "\n";

    SparseBlock mat(mrows, ncols);

    int row_counter = 0;
    int col_counter = 0;
    int prev_rows_size = 0;
    int prev_cols_size = 0;
    int flat_count = 0;
    const int nnz = rows.size();
    int vi, vj, m, n;
    double value;

    //std::cout << "Starting Main for loop" << "\n";
    for (size_t v = 0; v < nnz; v++)
    {
        //std::cout << "loop v = " << v << "\n";
        vi = rows[v];
        vj = cols[v];
        //std::cout << "vi = " << vi << "\n";
        //std::cout << "vj = " << vj << "\n";

        if (vi != row_counter)
        {
            //std::cout << "if vi != row_counter " << "\n";
            row_counter += 1;
            prev_rows_size += m;
            prev_cols_size  = 0;
            //std::cout << "row_counter = " << row_counter << "\n";
            //std::cout << "prev_rows_size = " << prev_rows_size << "\n";
            //std::cout << "prev_cols_size = " << prev_cols_size << "\n";
            //std::cout << "Exit if" << "\n";
        }

        m = data[v].rows();
        n = data[v].cols();
        //std::cout << "m = " << m << "\n";
        //std::cout << "n = " << n << "\n";

        if (n == 3)
        {
            //std::cout << "if n == 3" << "\n";
            prev_cols_size = 7*(vj/2);
            //std::cout << "prev_cols_size = " << prev_cols_size << "\n";
        }
        else
        {
            //std::cout << "if n!= 3" << "\n";
            prev_cols_size = 7*(vj/2) + 3;
            //std::cout << "prev_cols_size = " << prev_cols_size << "\n";
        }
        
        //std::cout << "Entring for (size_t i = 0; i < m; i++)" << "\n";
        for (size_t i = 0; i < m; i++)
        {
            //std::cout << "i = " << i << "\n";
            //std::cout << "Entring for (size_t j = 0; j < n; j++)" << "\n\n";
            for (size_t j = 0; j < n; j++)
            {
                //std::cout << "j = " << j << "\n";
                if (std::abs(data[v](i,j))>1/1000)
                {
                    //std::cout << "if (std::abs(data[v](i,j))>1/1000) = " << std::abs(data[v](i,j)) << "\n";
                    //std::cout << "Inserting Data " << "\n";
                    //std::cout << "mat(" << prev_rows_size+i << "," << prev_cols_size+j << ") = " << data[v](i,j) << "\n";
                    mat.insert(prev_rows_size+i, prev_cols_size+j) = data[v](i,j);
                    //std::cout << "Done" << "\n";
                }
                
            }
            
        }
        
    }
    return mat;
    
};

