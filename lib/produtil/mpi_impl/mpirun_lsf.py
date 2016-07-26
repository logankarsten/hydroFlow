##@namespace produtil.mpi_impl.mpirun_lsf
# Adds LSF+IBMPE support to produtil.run
#
# This module is part of the produtil.mpi_impl package.  It underlies
# the produtil.run.openmp, produtil.run.mpirun , and
# produtil.run.mpiserial functions, providing the implementation
# needed to run with LSF combined with the IBMPE MPI implementation.
# It may work with other MPI implementations connected to LSF, as long
# as they use mpirun.lsf to launch MPI programs.
#
# @note Unlike other MPI implementations, LSF does not allow changing of the
#  number of MPI ranks used when running an MPI program.  You can only run
#  on all provided ranks, or one rank.  Hence the TOTAL_TASKS variable used
#  elsewhere in produtil, is ignored here.
import os, socket, logging
import produtil.fileop,produtil.prog,produtil.mpiprog

from .mpi_impl_base import MPIMixed,CMDFGen

##@var mpirun_lsf_path
# Path to the mpirun.lsf program, or None if it isn't found.
mpirun_lsf_path=produtil.fileop.find_exe('mpirun.lsf',raise_missing=False)

def openmp(arg,threads):
    """!Adds OpenMP support to the provided object

    @param arg An produtil.prog.Runner or
    produtil.mpiprog.MPIRanksBase object tree
    @param threads the number of threads, or threads per rank, an
    integer"""
    if threads is not None:
        return arg.env(OMP_NUM_THREADS=threads)
    else:
        return arg

def detect():
    """!Determines if LSF+IBMPE should be used to run MPI programs by
    looking for the mpirun.lsf program in $PATH"""
    return mpirun_lsf_path is not None

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

def mpirunner(arg,allranks=False,logger=None,**kwargs):
    """!Turns a produtil.mpiprog.MPIRanksBase tree into a produtil.prog.Runner
    @param arg a tree of produtil.mpiprog.MPIRanksBase objects
    @param allranks if True, and only one rank is requested by arg, then
      all MPI ranks will be used
    @param logger a logging.Logger for log messages
    @param kwargs passed to produtil.mpi_impl.mpi_impl_base.CMDFGen
      when mpiserial is in use.
    @returns a produtil.prog.Runner that will run the selected MPI program
    @note LSF does not support modifying the number of MPI ranks
      to use when running a program.  You can only use all provided
      ranks, or one rank."""
    if logger is None:
        logger=logging.getLogger('mpirun_lsf')
    assert(isinstance(arg,produtil.mpiprog.MPIRanksBase))
    (serial,parallel)=arg.check_serial()
    if serial and parallel:
        raise MPIMixed(
            'Cannot mix serial and parallel MPI ranks in the same '
            'MPI program.')
    if arg.nranks()==1 and allranks:
        arglist=[ a for a in arg.to_arglist(
                pre=[mpirun_lsf_path],
                before=[],
                between=[])]
        return produtil.prog.Runner(arglist)
    elif arg.nranks()==1:
        # Hack to get LSF to run only one MPI rank.  Tested on NCEP
        # WCOSS supercomputer and approved by its IBM support staff.
        host=socket.gethostname()
        runner=produtil.prog.Runner(
            [ a for a in arg.to_arglist(
                    pre=[mpirun_lsf_path],
                    before=[], between=[]) ])
        runner=runner.env(
            LSB_PJL_TASK_GEOMETRY="{(0)}",LSB_HOSTS=host,
            LSB_MCPU_HOSTS=host+" 1", LSB_DJOB_NUMPROC='1',
            LSB_MAX_NUM_PROCESSORS='1',MP_TASK_AFFINITY='core')
        if logger is not None:
            logger.info(
                'Using a hack to work around an LSF bug and run a one core '
                'program: '+repr(runner))
        return runner
    elif allranks:
        raise MPIAllRanksError(
            "When using allranks=True, you must provide an mpi program "
            "specification with only one MPI rank (to be duplicated "
            "across all ranks).")
    elif serial:
        lines=[a for a in arg.to_arglist(to_shell=True,expand=True)]
        if produtil.fileop.find_exe('mpiserial') is None:
            raise MPISerialMissing(
                'Attempting to run a serial program via mpirun.lsf but '
                'the mpiserial program is not in your $PATH.')
        return produtil.prog.Runner([mpirun_lsf_path,'mpiserial'],
                                    prerun=CMDFGen('serialcmdf',lines,
                                                   cmd_envar='MP_CMDFILE',
                                                   model_envar='MP_PGMMODEL',
                                                   **kwargs))
    else:        
        lines=[a for a in arg.to_arglist(to_shell=True,expand=True)]
        return produtil.prog.Runner([mpirun_lsf_path],
                                    prerun=CMDFGen('mpirun_lsf_cmdf',lines,
                                                   cmd_envar='MP_CMDFILE',
                                                   model_envar='MP_PGMMODEL',
                                                   **kwargs))

