all_tests()
add_subdirectory(multivar)
add_subdirectory(functions)
add_subdirectory(meshes)

add_cpp_test(mpi_gf)
set(TEST_MPI_NUMPROC 2)
add_cpp_test(mpi_gf)
set(TEST_MPI_NUMPROC 3)
add_cpp_test(mpi_gf)
set(TEST_MPI_NUMPROC 4)
add_cpp_test(mpi_gf)
