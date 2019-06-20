
#include "utilities.hpp"

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");

void ExportResultsCSV(const std::string& name, const std::vector<Eigen::VectorXd>& results)
{
    std::ofstream results_file;
    results_file.open (name + ".csv");
    for (auto x : results) {results_file << x.transpose().format(CSVFormat) ;};
    results_file.close();
    std::cout << "Results Saved as : " << name + ".csv" << "\n";
};

