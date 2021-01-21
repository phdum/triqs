#include <triqs/stat/mean_error.hpp>
#include <mpi/mpi.hpp>

using namespace triqs::stat;

int main() {
  mpi::communicator world;
  int rank = world.rank();
  // Define linear data spread over the different mpi threads:
  // thread 0: {1,2,3,4}; thread 1 {5,6,7,8}, etc.
  std::vector<double> data{4. * rank + 1, 4. * rank + 2, 4. * rank + 3, 4. * rank + 4};

  auto [ave, err] = mean_and_err(data);

  std::cout << "Average: " << ave << std::endl; 
  // Output: (1.0 + 4.0 * world.size()) / 2.0
  std::cout << "Standard Error: " << err << std::endl; 
  // Output: sqrt((world.size() + 1.) / 12.)
} 

