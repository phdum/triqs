from cpp2py.wrap_generator import *
import re

module = module_(full_name = "triqs.gf.meshes", doc = "All the meshes", app_name="triqs")

module.add_imports("triqs.lattice")

module.add_include("<triqs/gfs.hpp>")

module.add_include("<cpp2py/converters/string.hpp>")
module.add_include("<cpp2py/converters/vector.hpp>")
module.add_include("<cpp2py/converters/function.hpp>")
module.add_include("<cpp2py/converters/optional.hpp>")

module.add_include("<triqs/cpp2py_converters.hpp>")

module.add_using("namespace triqs::arrays")
module.add_using("namespace triqs::mesh")
module.add_preamble("""
""")

## --------------------------------------------------
##                    WARNING
# The names of the Meshes MUST be MeshXXX, with XXX = ImFreq, ...
# where XXX is the name appearing in hdf5 in multivar mesh (gf/h5.hpp, gf_h5_name trait).
## --------------------------------------------------


########################
##   enums
########################

module.add_enum(c_name = "statistic_enum",
                c_namespace = "triqs::mesh",
                values = ["Fermion","Boson"])

module.add_enum(c_name = "triqs::mesh::imfreq::option",
         c_namespace = "",
         values = ["imfreq::option::all_frequencies","imfreq::option::positive_frequencies_only"])

########################
##   Mesh generic
########################

def make_mesh(py_type, c_tag, index_type='long'):

    m = class_( py_type = py_type,
            c_type = "%s"%c_tag,
            c_type_absolute = "triqs::mesh::%s"%c_tag,
            hdf5 = True,
            serializable= "tuple",
            is_printable= True,
            comparisons = "== !="
           )

    m.add_method("long index_to_linear(%s i)"%index_type, doc = "index -> linear index")
    m.add_len(calling_pattern = "int result = self_c.size()", doc = "Size of the mesh")
    m.add_iterator()
    m.add_method("PyObject * values()",
                 calling_pattern = """
                    static auto cls = pyref::get_class("triqs.gf", "MeshValueGenerator", /* raise_exception */ true);
                    pyref args = PyTuple_Pack(1, self);
                    auto result = PyObject_CallObject(cls, args);
                 """, doc = "A numpy array of all the values of the mesh points")
    
    m.add_method_copy()
    m.add_method_copy_from()

    return m

########################
##   MeshImFreq
########################

m = make_mesh( py_type = "MeshImFreq", c_tag = "imfreq")
m.add_constructor(signature = "(double beta, statistic_enum S, int n_max=1025)")
m.add_method("""int last_index()""")
m.add_method("""int first_index()""")
m.add_method("""bool positive_only()""")
m.add_method("""void set_tail_fit_parameters(double tail_fraction, int n_tail_max = 30, std::optional<int> expansion_order = {})""")
m.add_property(name = "beta",
               getter = cfunction(calling_pattern="double result = self_c.domain().beta",
               signature = "double()",
               doc = "Inverse temperature"))
m.add_property(name = "statistic",
               getter = cfunction(calling_pattern="statistic_enum result = self_c.domain().statistic", signature = "statistic_enum()"),
               doc = "Statistic")
m.add_call(signature = "dcomplex (long n)", calling_pattern = " auto result = dcomplex{0, (2*n + self_c.domain().statistic)*M_PI/self_c.domain().beta}", doc = "")

module.add_class(m)

########################
##   MeshImTime
########################

m = make_mesh(py_type = "MeshImTime", c_tag = "imtime")
m.add_constructor(signature = "(double beta, statistic_enum S, int n_max)")
m.add_property(name = "beta",
               getter = cfunction(calling_pattern="double result = self_c.domain().beta",
               signature = "double()",
               doc = "Inverse temperature"))
m.add_property(name = "statistic",
               getter = cfunction(calling_pattern="statistic_enum result = self_c.domain().statistic", signature = "statistic_enum()"),
               doc = "Statistic")

module.add_class(m)

########################
##   MeshLegendre
########################


# the domain
dom = class_( py_type = "GfLegendreDomain",
        c_type = "legendre_domain",
        c_type_absolute = "triqs::mesh::legendre_domain",
        serializable= "tuple",
       )
dom.add_constructor(signature = "(double beta, statistic_enum S, int n_max)")
module.add_class(dom)

# the mesh
m = make_mesh( py_type = "MeshLegendre", c_tag = "triqs::mesh::legendre")
m.add_constructor(signature = "(double beta, statistic_enum S, int n_max=1025)")
m.add_property(name = "beta",
               getter = cfunction(calling_pattern="double result = self_c.domain().beta",
               signature = "double()",
               doc = "Inverse temperature"))
m.add_property(name = "statistic",
               getter = cfunction(calling_pattern="statistic_enum result = self_c.domain().statistic", signature = "statistic_enum()"),
               doc = "Statistic")

module.add_class(m)

########################
##   MeshReFreq
########################

m = make_mesh(py_type = "MeshReFreq", c_tag = "refreq")
m.add_constructor(signature = "(double omega_min, double omega_max, int n_max)")

m.add_property(name = "omega_min",
               getter = cfunction(calling_pattern="double result = self_c.x_min()",
               signature = "double()",
               doc = "Inverse temperature"))

m.add_property(name = "omega_max",
               getter = cfunction(calling_pattern="double result = self_c.x_max()",
               signature = "double()",
               doc = "Inverse temperature"))

m.add_property(name = "delta",
               getter = cfunction(calling_pattern="double result = self_c.delta()",
               signature = "double()",
               doc = "The mesh-spacing"))

module.add_class(m)

########################
##   MeshReTime
########################

m = make_mesh(py_type = "MeshReTime", c_tag = "retime")
m.add_constructor(signature = "(double t_min, double t_max, int n_max)")

m.add_property(name = "t_min",
               getter = cfunction(calling_pattern="double result = self_c.x_min()",
               signature = "double()",
               doc = "Inverse temperature"))

m.add_property(name = "t_max",
               getter = cfunction(calling_pattern="double result = self_c.x_max()",
               signature = "double()",
               doc = "Inverse temperature"))

m.add_property(name = "delta",
               getter = cfunction(calling_pattern="double result = self_c.delta()",
               signature = "double()",
               doc = "The mesh-spacing"))

module.add_class(m)

########################
##   MeshBrZone
########################

m = make_mesh( py_type = "MeshBrZone", c_tag = "brzone", index_type = 'std::array<long,3>' )
m.add_constructor(signature = "(triqs::lattice::brillouin_zone b, int n_k)")
m.add_constructor(signature = "(triqs::lattice::brillouin_zone b, matrix_view<int> periodization_matrix)")
m.add_method(name="locate_neighbours", signature="std::array<long,3> locate_neighbours(triqs::arrays::vector<double> x)")

m.add_property(name = "linear_dims",
               getter = cfunction(calling_pattern="std::array<long,3> result = self_c.get_dimensions()",
               signature = "std::array<long,3>()",
               doc = "Linear dimensions"))
m.add_property(name = "domain",
               getter = cfunction(calling_pattern="brillouin_zone result = self_c.domain()",
               signature = "brillouin_zone()",
               doc = "Domain"))

module.add_class(m)

########################
##   MeshCycLat
########################

m = make_mesh( py_type = "MeshCycLat", c_tag = "cyclat", index_type = 'std::array<long,3>' )
m.add_constructor(signature = "(int L1, int L2, int L3)")
m.add_constructor(signature = "(triqs::lattice::bravais_lattice b, matrix_view<int> periodization_matrix)")
m.add_constructor(signature = "(triqs::lattice::bravais_lattice b, int L)")
m.add_method(name="locate_neighbours", signature="std::array<long,3> locate_neighbours(triqs::arrays::vector<double> x)")

m.add_property(name = "linear_dims",
               getter = cfunction(calling_pattern="std::array<long,3> result = self_c.get_dimensions()",
               signature = "std::array<long,3>()",
               doc = "Linear dimensions"))
m.add_property(name = "domain",
               getter = cfunction(calling_pattern="bravais_lattice result = self_c.domain()",
               signature = "bravais_lattice()",
               doc = "Domain"))

module.add_class(m)

############################
##   Mesh Factory Functions
############################

# ---------------------- make_adjoint_mesh --------------------
module.add_function("brzone triqs::gfs::make_adjoint_mesh(cyclat m)", doc = "Create the adjoint k-mesh")
module.add_function("cyclat triqs::gfs::make_adjoint_mesh(brzone m)", doc = "Create the adjoint r-mesh")
module.add_function("imfreq triqs::gfs::make_adjoint_mesh(imtime m, int n_iw = -1)", doc = "Create the adjoint iw-mesh")
module.add_function("imtime triqs::gfs::make_adjoint_mesh(imfreq m, int n_tau = -1)", doc = "Create the adjoint tau-mesh")
module.add_function("refreq triqs::gfs::make_adjoint_mesh(retime m, bool shift_half_bin = false)", doc = "Create the adjoint w-mesh")
module.add_function("retime triqs::gfs::make_adjoint_mesh(refreq m, bool shift_half_bin = false)", doc = "Create the adjoint t-mesh")

##   Code generation
module.generate_code()
