##@namespace produtil.mpi_impl.mpiexec 
# Adds MPICH or MVAPICH2 support to produtil.run
#
# This module is part of the mpi_impl package -- see produtil.mpi_impl
# for details.  This implements the Hydra MPI wrapper and MPICH MPI
# implementation with Intel OpenMP, but may work for other MPI
# implementations that use the "mpiexec" command and OpenMP
# implementations that use the KMP_NUM_THREADS or OMP_NUM_THREADS
# environment variables.
#
# @warning This module assumes the TOTAL_TASKS environment variable is
# set to the maximum number of MPI ranks the program has available to
# it.  That is used when the mpirunner is called with the
# allranks=True option.

import os
import produtil.fileop,produtil.prog,produtil.mpiprog

from .mpi_impl_base import MPIMixed,CMDFGen

##@var mpiexec_path
# Path to the mpiexec program
mpiexec_path=produtil.fileop.find_exe('mpiexec',raise_missing=False)

def openmp(arg,threads):
    """!Adds OpenMP support to the provided object

    @param arg An produtil.prog.Runner or
    produtil.mpiprog.MPIRanksBase object tree
    @param threads the number of threads, or threads per rank, an
    integer"""
    if threads is not None:
        return arg.env(OMP_NUM_THREADS=threads,KMP_NUM_THREADS=threads,
                       KMP_AFFINITY='scatter')
    else:
        return arg

def detect():
    """!Detects whether the MPICH mpi implementation is available by
    looking for the mpiexec program in $PATH."""
    return mpiexec_path is not None

def can_run_mpi():
    """!Does this module represent an MPI implementation? Returns True."""
    return True

def bigexe_prepend(arg,**kwargs): 
    """!Returns an array of arguments to prepend before a serial
    command to execute in order to run it on compute nodes instead of
    the batch node.  
    @returns an empty list
    @param arg The produtil.prog.Runner to run on compute nodes
    @param kwargs Ignored."""
    return []

def mpirunner(arg,allranks=False,**kwargs):
    """!Turns a produtil.mpiprog.MPIRanksBase tree into a produtil.prog.Runner
    @param arg a tree of produtil.mpiprog.MPIRanksBase objects
    @param allranks if True, and only one rank is requested by arg, then
      all MPI ranks will be used
    @param kwargs passed to produtil.mpi_impl.mpi_impl_base.CMDFGen
      when mpiserial is in use.
    @returns a produtil.prog.Runner that will run the selected MPI program
    @warning Assumes the TOTAL_TASKS environment variable is set
      if allranks=True"""
    assert(isinstance(arg,produtil.mpiprog.MPIRanksBase))
    (serial,parallel)=arg.check_serial()
    if serial and parallel:
        raise MPIMixed('Cannot mix serial and parallel MPI ranks in the '
                       'same MPI program.')
    if arg.nranks()==1 and allranks:
        arglist=[ a for a in arg.to_arglist(
                pre=[mpiexec_path],
                before=['-np',os.environ['TOTAL_TASKS']],
                between=[':'])]
        return produtil.prog.Runner(arglist)
    elif allranks:
        raise MPIAllRanksError(
            "When using allranks=True, you must provide an mpi program "
            "specification with only one MPI rank (to be duplicated across "
            "all ranks).")
    elif serial:
        lines=[a for a in arg.to_arglist(to_shell=True,expand=True)]
        if produtil.fileop.find_exe('mpiserial') is None:
            raise MPISerialMissing(
                'Attempting to run a serial program via mpiexec but the '
                'mpiserial program is not in your $PATH.')
        return produtil.prog.Runner(
            [mpiexec_path,'-np','%s'%(arg.nranks()),'mpiserial'],
            prerun=CMDFGen('serialcmdf',lines,**kwargs))
    else:
        arglist=[ a for a in arg.to_arglist(
                pre=[mpiexec_path],   # command is mpiexec
                before=['-np','%(n)d'], # pass env, number of procs is kw['n']
                between=[':']) ] # separate commands with ':'
        return produtil.prog.Runner(arglist)

