##@namespace produtil.mpi_impl.no_mpi
# Stub funcitons to allow produtil.mpi_impl  to run when MPI is unavailable.
#
# This module is part of the produtil.mpi_impl package.  It underlies
# the produtil.run.openmp, produtil.run.mpirun , and
# produtil.run.mpiserial functions, providing the implementation
# needed to run when MPI is unavailable.


def openmp(arg,threads):
    """!Does nothing.  This implementation does not support OpenMP.

    @param arg An produtil.prog.Runner or
    produtil.mpiprog.MPIRanksBase object tree
    @param threads the number of threads, or threads per rank, an
    integer"""
def mpirunner(arg,**kwargs):
    """!Raises an exception to indicate MPI is not supported
    @param arg,kwargs Ignored."""
    raise MPIDisabled('This job cannot run MPI programs.')
def can_run_mpi():
    """!Returns False to indicate MPI is not supported."""
    return False
def bigexe_prepend(arg,**kwargs): 
    """!Returns an empty list to indicate that nothing is needed to
    run large serial programs on compute nodes.
    @param arg,kwargs Ignored."""
    return []

