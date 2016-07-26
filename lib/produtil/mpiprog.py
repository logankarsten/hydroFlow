"""!Object structure for describing MPI programs.

Do not load this module directly.  It is meant to be loaded only by
the produtil.run module.

This module handles execution of MPI programs, and execution of
groups of non-MPI programs through an MPI interface (which requires
all sorts of tricks).  This module is also the interface to the
various produtil.mpi_impl.* modules that generate the shell command to
run MPI programs.  This module is built on top of the produtil.prog
module and uses it to run the MPI-launching program for your local
cluster (mpiexec, mpirun, poe, etc.)  

In addition, this module contains code to simplify adding new MPI
implementations to the produtil.mpi_impl subpackage.  High-level code,
such as the HWRF scripts, use the produtil.run module to generate
object trees of MPIRanksBase objects.  The produtil.mpi_impl
subpackages then implement an mpirunner function that turns those into
a produtil.prog.Runner to be directly executed.  The MPIRanksBase
object, and its subclasses, implement a few utilites to automate that
for you:

* to_arglist --- converts the MPI ranks to an mpi launcher command
                 as a produtil.prog.Runner, or to an array of strings
                 for a command file.

* nranks     --- calculates the number of requested MPI ranks 

* expand_iter --- iterates over groups of identical MPI ranks

* check_serial --- tells whether this program is running MPI programs,
                 or running serial programs as if they were MPI
                 (or both, which most MPI implementations don't support)

For MPI implementations that require a command file, see the
produtil.mpi_impl.mpi_impl_base CMDFGen class to have the
produtil.prog module automatically write the command file before
executing the program.  The produtil.mpi_impl.mpirun_lsf shows an
example of how to use it.

See the produtil.run module for full documentation."""

##@var __all__
# Ensure nothing is loaded by "from produtil.mpiprog import *"
__all__=[]

import produtil.prog
from produtil.prog import ProgSyntaxError

class MPIProgSyntaxError(ProgSyntaxError): 
    """!Base class of syntax errors in MPI program specifications"""
class ComplexProgInput(MPIProgSyntaxError): 
    """!Raised when something that cannot be expressed as a pure MPI
    rank is given as a pure MPI rank."""
class NotMPIProg(ProgSyntaxError): 
    """!Raised when an MPI program was expected but something else was
    given."""
class NotSerialProg(ProgSyntaxError): 
    """!Raised when a serial program was expected, but something else
    was given."""
class InputsNotStrings(ProgSyntaxError): 
    """!Raised when the validation scripts were expecting string
    arguments or string executable names, but something else was
    found."""

########################################################################

class MPIRanksBase(object):
    """!This is the abstract superclass of all classes that represent
    one or more MPI ranks, including MPI ranks that are actually
    serial programs.  Subclasses of MPIRanksBase allow an MPI program
    to be represented as a tree of MPIRanksBase objects, in such a way
    that they can be easily converted to a produtil.prog.Runner object
    for execution.  The actual conversion to a Runner is done in the
    produtil.mpi_impl package (see produtil/mpi_impl/__init__.py)"""

    def to_arglist(self,to_shell=False,expand=False,shell_validate=None,
                   pre=[],before=[],between=[],after=[],post=[],extra={}):
        """!This is the underlying implementation of most of the
        mpi_impl modules, and hence make_runner as well.  It converts
        this group of MPI ranks into a set of arguments suitable for
        sending to a Runner object or for writing to a command file.
        This is done by iterating over either all ranks (if
        expand=True) or groups of repeated ranks (if expand=False),
        converting their arguments to a list.  It prepends an
        executable, and can insert other arguments in specified
        locations (given in the pre, before, between, after, and post
        arguments).  It can also use the to_shell argument to convert
        programs to POSIX sh commands, and it performs simple string
        interpolation via the "extra" hash.

        If to_shell=False then the executable and arguments are
        inserted directly to the output list.  Otherwise (when
        to_shell=True) the to_shell subroutine is called on the
        MPIRank object to produce a single argument that contains a
        shell command.  That single argument is then used in place of
        the executable and arguments.  Note that may raise
        NotValidPosixSh (or a subclass thereof) if the command cannot
        be expressed as a shell command.  In addition, if
        shell_validate is not None, then it is called on each
        post-conversion shell argument, and the return value is used
        instead.

        You can specify additional argument lists to be inserted in
        certain locations.  Each argument in those lists will be
        processed through the % operator, specifying "extra" as the
        keyword list with two new keywords added: nworld is the number
        of ranks in the MPI program, and "n" is the number in the
        current group of repeated ranks if expand=False (n=1 if
        expand=True).  Those argument lists are: pre, before, between,
        after and post.

        @param to_shell If True, convert executable and arguments to
          a POSIX sh command instead of inserting them directly.
        @param expand If True, groups of repeated ranks are expanded.
        @param shell_validate A function to convert each argument to
          some "shell-acceptable" version.
        @param pre Inserted before everything else.  This is where you
          would put the "mpiexec" and any global settings.
        @param before Inserted before each rank (if expand=True) or group
          (if expand=False)
        @param between Inserted between each rank (if expand=True) or group
          (if expand=False)
        @param after Inserted after each rank (if expand=True) or group (if
          expand=False)
        @param post Appended at the end of the list of arguments.
        @param extra used for string expansion"""
        kw=dict(extra)
        kw['nworld']=self.nranks()
        for x in pre: yield x%kw
        first=True
        for rank,count in self.expand_iter(bool(expand)):
            assert(isinstance(rank,MPIRanksBase))
            assert(isinstance(count,int))
            if not count>0:
                continue
            if first:
                first=False
            else:
                for x in between: yield x%kw
            kw['n']=count
            for x in before: yield x%kw
            if to_shell:
                if shell_validate is not None:
                    yield shell_validate(rank.to_shell())
                else:
                    yield rank.to_shell()
            else:
                for arg in rank.args(): yield arg
            for x in after: yield x%kw
        for x in post: yield x%kw
    def make_runners_immutable(self):
        """!Returns a copy of this object where all child
        produtil.prog.Runner objects have been replaced with
        produtil.prog.ImmutableRunner objects."""
    def get_logger(self):
        """!Returns a logger.Logger object for this MPIRanksBase or one
        from its child MPIRanksBase objects (if it has any).  If no
        logger is found, None is returned."""
        return None
    def check_serial(self):
        """!Returns a tuple (s,p) where s=True if there are serial
        ranks in this part of the MPI program, and p=True if there are
        parallel ranks.  Note that it is possible that both could be
        True, which is an error.  It is also possible that neither are
        True if there are zero ranks."""
        return (False,False)
    def nranks(self):
        """!Returns the number of ranks in this part of the MPI
        program."""
        return 0
    def ranks(self):
        """!Iterates over all MPIRank objects in this part of the MPI
        program."""
        pass
    def ngroups(self):
        """!Returns the number of groups of repeated MPI ranks in the
        MPI program."""
        return 0
    def groups(self):
        """!Iterates over all groups of repeating MPI ranks in the MPI
        program returning tuples (r,c) containing a rank r and the
        count (number) of that rank c."""
        pass
    def __eq__(self,other):
        """!Returns True if this part of
        the MPI program is the same as another part.  
        @param other the other object"""
        return NotImplemented
    def __mul__(self,factor):
        """!Returns a new set of MPI ranks that consist of this group
        of ranks repeated "factor" times.
        @param factor how many times to duplicate"""
        return NotImplemented
    def __rmul__(self,other):
        """!Returns a new set of MPI ranks that consist of this group
        of ranks repeated "factor" times.
        @param other how many times to duplicate"""
        return NotImplemented
    def __add__(self,other):
        """!Returns a new set of MPI ranks that consist of this set of
        ranks with the "other" set appended.
        @param other the data to append"""
        return NotImplemented
    def __radd__(self,other):
        """!Returns a new set of MPI ranks that consist of the "other"
        set of ranks with this set appended.
        @param other the data to prepend"""
        return NotImplemented
    def isplainexe(self):
        """!Determines if this set of MPI ranks can be represented by a
        single serial executable with a single set of arguments run
        without MPI.  Returns false by default: this function can only
        return true for MPISerial."""
        return False
    def to_shell(self):
        """!Returns a POSIX sh command that will execute the serial
        program, if possible, or raise a subclass of NotValidPosixSh
        otherwise.  Works only on single MPI ranks that are actually
        MPI wrappers around a serial program (ie.: from mpiserial)."""
        raise NotSerialProg('This is an MPI program, so it cannot be represented as a non-MPI POSIX sh command.')
    def expand_iter(self,expand):
        """!This is a wrapper around ranks() and groups() which will
        call self.groups() if expand=False.  If expand=True, this will
        call ranks() returning a tuple (rank,1) for each rank.
        @param expand If True, expand groups of identical ranks into one
          rank of each member"""
        if expand:
            for rank in self.ranks():
                yield (rank,1)
        else:
            for rank,count in self.groups():
                yield (rank,count)
    def __repr__(self):
        """!Returns a string representation of this object intended for debugging."""
        raise NotImplementedError('This class did not implement __repr__.')

########################################################################

class MPIRanksSPMD(MPIRanksBase):
    """!Represents one MPI program duplicated across many ranks."""
    def __init__(self,mpirank,count):
        """!MPIRanksSPMD constructor
        @param mpirank the program to run
        @param count how many times to run it"""
        if not isinstance(mpirank,MPIRank):
            raise MPIProgSyntaxError('Input to MPIRanksSPMD must be an MPIRank.')
        self._mpirank=mpirank
        self._count=int(count)
    def make_runners_immutable(self):
        """!Returns a new MPIRanksSPMD with an immutable version of
        self._mpirank."""
        return MPIRanksSPMD(self._mpirank.make_runners_immutable(),self._count)
    def __repr__(self):
        """!Returns "X*N" where X is the MPI program and N is the
        number of ranks."""
        return '%s*%d'%(repr(self._mpirank),int(self._count))
    def ngroups(self):
        """!Returns 1 or 0: 1 if there are ranks and 0 if there are none."""
        if self._count>0:
            return 1
        else:
            return 0
    def groups(self):
        """!Yields a tuple (X,N) where X is the mpi program and N is
        the number of ranks."""
        yield self._mpirank,self._count
    def copy(self):
        """!Returns a deep copy of self."""
        return MPIRanksSPMD(self._mpirank.copy(),self._count)
    def ranks(self):
        """!Iterates over MPI ranks within self."""
        if self._count>0:
            for i in xrange(self._count):
                yield self._mpirank
    def nranks(self):
        """!Returns the number of ranks this program requests."""
        if self._count>0:
            return self._count
        else:
            return 0
    def __mul__(self,factor):
        """!Multiply the number of requested ranks by some factor."""
        if not isinstance(factor,int):
            return NotImplemented
        return MPIRanksSPMD(self._mpirank.copy(),count*factor)
    def __rmul__(self,factor):
        """!Multiply the number of requested ranks by some factor."""
        if not isinstance(factor,int):
            return NotImplemented
        return MPIRanksSPMD(self._mpirank.copy(),count*factor)
    def __add__(self,other):
        """!Add some new ranks to self.  If they are not identical to
        the MPI program presently requested, this returns a new
        MPIRanksMPMD."""
        copy=True
        ocount=other.nranks()
        for mpirank,count in other.groups():
            if not mpirank==self._mpirank:
                copy=False
                break
        if copy:
            return MPIRanksSPMD(self._mpirank.copy(),self.nranks()+ocount)
        else:
            return MPIRanksMPMD([self.copy(),other.copy()])
    def check_serial(self):
        """!Checks to see if this program contains serial (non-MPI) or
        MPI components.
        @returns a tuple (serial,parallel) where serial is True if
        there are serial components, and parallel is True if there are
        parallel components.  If there are no components, returns 
        (False,False)"""
        if self._count>0:
            return self._mpirank.check_serial()
        else:
            return (False,False)
    def get_logger(self):
        """!Returns my MPI program's logger."""
        return self._mpirank.get_logger()

########################################################################

class MPIRanksMPMD(MPIRanksBase):
    """!Represents a group of MPI programs, each of which have some
    number of ranks assigned."""
    def __init__(self,args):
        """!MPIRanksMPMD constructor
        @param args an array of MPIRanksBase to execute."""
        self._el=list(args)
        self._ngcache=None
        self._nrcache=None
    def make_runners_immutable(self):
        """!Tells each containing element to make its
        produtil.prog.Runners into produtil.prog.ImmutableRunners so
        that changes to them will not change the original."""
        return MPIRanksMPMD([el.make_runners_immutable() for el in self._el])
    def __repr__(self):
        """!Returns a pythonic description of this object."""
        reprs=[]
        for el in self._el:
            if el.nranks()>0:
                reprs.append(repr(el))
        return ' + '.join(reprs)
    def ngroups(self):
        """!How many groups of identical repeated ranks are in this
        MPMD program?"""
        if self._ngcache is None:
            ng=0
            for g in self._el:
                ng+=g.ngroups()
            self._ngcache=ng
        return self._ngcache
    def nranks(self):
        """!How many ranks does this program request?"""
        if self._nrcache is None:
            nr=0
            for g in self._el:
                nr+=g.nranks()
            self._nrcache=nr
        return self._nrcache
    def groups(self):
        """!Iterates over tuples (rank,count) of groups of identical
        ranks."""
        for groups in self._el:
            for rank,count in groups.groups():
                yield rank,count
    def ranks(self):
        """!Iterates over groups of repeated ranks returning the number
        of ranks each requests."""
        for ranks in self._el:
            for rank in ranks.ranks():
                yield rank
    def __add__(self,other):
        """!Adds more ranks to this program.
        @param other an MPIRanksMPMD or MPIRanksSPMD to add"""
        if isinstance(other,MPIRanksMPMD):
            return MPIRanksMPMD(self._el+other._el)
        elif isinstance(other,MPIRank) or isinstance(other,MPIRanksBase):
            return MPIRanksMPMD(self._el+[other])
        return NotImplemented
    def __radd__(self,other):
        """!Prepends more ranks to this program.
        @param other an MPIRanksMPMD or MPIRanksSPMD to prepend"""
        if isinstance(other,MPIRanksMPMD):
            return MPIRanksMPMD(other._el+self._el)
        elif isinstance(other,MPIRank) or isinstance(other,MPIRanksBase):
            return MPIRanksMPMD([other]+self._el)
        return NotImplemented
    def __mul__(self,factor):
        """!Duplicates this MPMD program "factor" times.
        @param factor how many times to duplicate this program."""
        if isinstance(factor,int):
            return MPIRanksMPMD(self._el*factor)
        return NotImplemented
    def __rmul__(self,factor):
        """!Duplicates this MPMD program "factor" times.
        @param factor how many times to duplicate this program."""
        if isinstance(factor,int):
            return MPIRanksMPMD(factor*self._el)
    def check_serial(self):
        """!Checks to see if this program contains serial (non-MPI) or
        MPI components.
        @returns a tuple (serial,parallel) where serial is True if
        there are serial components, and parallel is True if there are
        parallel components."""
        serial=False
        parallel=False
        for el in self._el:
            (s,p)=el.check_serial()
            serial = (serial or s)
            parallel = (parallel or p)
        return (serial,parallel)
    def get_logger(self):
        """!Returns a logging.Logger for the first rank that has one."""
        for el in self._el:
            logger=el.get_logger()
            if logger is not None:
                return logger
        return None

########################################################################

class MPIRank(MPIRanksBase):
    """!Represents a single MPI rank."""
    def __init__(self,arg,logger=None):
        """!MPIRank constructor.
        @param arg What program to run.  Can be a produtil.prog.Runner,
         or some way of creating one, such as a program name or list
         of program+arguments.
        @param logger a logging.Logger for log messages or None to
         have no logger."""
        self._logger=logger
        if isinstance(arg,MPIRank):
            if self._logger is None:
                self._logger=arg._logger
            self._args=list(arg._args)
        elif isinstance(arg,produtil.prog.Runner):
            if arg.isplainexe():
                self._args=[x for x in arg.args()]
            else:
                raise ComplexProgInput(
                    'Tried to convert a Runner to an MPIRank directly, when '
                    'the Runner had more than an executable and arguments.  '
                    'Use mpiserial instead.')
        elif isinstance(arg,basestring):
            self._args=[arg]
        elif isinstance(arg,list) or isinstance(arg,tuple):
            self._args=[x for x in arg]
        else:
            raise MPIProgSyntaxError(
                'Input to MPIRank.__init__ must be a string, a list of '
                'strings, or a Runner that contains only the executable '
                'and its arguments.')
        self.validate()
    def to_shell(self):
        """!Return a POSIX sh representation of this MPI rank, if
        possible."""
        return ' '.join([produtil.prog.shbackslash(x) for x in self._args])
    def __getitem__(self,args):
        """!Adds arguments to this MPI rank's program."""
        c=self.copy()
        if isinstance(args,basestring):
            c._args.append(args)
        else:
            c._args.extend(args)
        return c
    def __repr__(self):
        """!Returns a Pythonic representation of this object for
        debugging."""
        s='mpi(%s)'%(repr(self._args[0]))
        if len(self._args)>1:
            s=s+'['+','.join([repr(x) for x in self._args[1:]])+']'
        return s
    def get_logger(self):
        """!Returns a logging.Logger for this object, or None."""
        return logger
    def validate(self,more=None):
        """!Checks to see if this MPIRank is valid, or has errors.
        @param more Arguments to the executable to validate.
        @returns None if there are no errors, or raises a descriptive
          exception."""
        for x in self.args():
            if not isinstance(x,basestring):
                raise InputsNotStrings(
                    'Executable and arguments must be strings.')
        if more is not None and len(more)>0:
            for x in more:
                if not isinstance(x,basestring):
                    raise InputsNotStrings(
                        'Executable and arguments must be strings.')
    def args(self):
        """!Iterates over the executable arguments."""
        for arg in self._args: yield arg
    def copy(self):
        """!Return a copy of self.  This is a deep copy except for the
        logger which whose reference is copied."""
        return MPIRank(self)
    def nranks(self):
        """!Returns 1: the number of MPI ranks."""
        return 1
    def ngroups(self):
        """!Returns 1: the number of groups of identical ranks."""
        return 1
    def ranks(self): 
        """!Yields self once: all MPI ranks."""
        yield self
    def groups(self): 
        """!Yields (self,1): all groups of identical ranks and the
        number per group."""
        yield (self,1)
    def __add__(self,other):
        """!Creates an MPIRanksSPMD or MPIRanksMPMD with this MPIRank
        and the other ranks.
        @param other The other ranks."""
        if not isinstance(other,MPIRank):
            return NotImplemented
        elif other==self:
            return MPIRanksSPMD(self.copy(),2)
        else:
            return MPIRanksMPMD([self,other])
    def __mul__(self,factor):
        """!Creates an MPIRanksSPMD with this MPIRank duplicated factor times.
        @param factor the number of times to duplicate"""
        if isinstance(factor,int):
            return MPIRanksSPMD(self,factor)
        return NotImplemented
    def __rmul__(self,factor):
        """!Creates an MPIRanksSPMD with this MPIRank duplicated factor times.
        @param factor the number of times to duplicate"""
        if isinstance(factor,int):
            return MPIRanksSPMD(self,factor)
        return NotImplemented
    def __eq__(self,other):
        """!Returns True if this MPIRank is equal to the other object."""
        return self._args==other._args
    def check_serial(self): 
        """!Returns (False,True): this is a pure parallel program."""
        return (False,True)

########################################################################

class MPISerial(MPIRank):
    """!Represents a single rank of an MPI program that is actually
    running a serial program.  This is supported directly by some
    MPI implementations while others require kludges to work properly."""
    def __init__(self,runner,logger=None):
        """!MPISerial constructor."""
        self._runner=runner
        self._logger=logger
    def make_runners_immutable(self):
        """!Creates a version of self with a produtil.prog.ImmutableRunner child."""
        if not isinstance(self._runner,produtil.prog.ImmutableRunner):
            return MPISerial(produtil.prog.ImmutableRunner(self._runner),self._logger)
        else:
            return self
    def copy(self):
        """!Duplicates self."""
        return MPISerial(self._runner,self._logger)
    def __repr__(self):
        """!Returns a pythonic string representation of self for debugging."""
        return 'mpiserial(%s)'%(repr(self._runner),)
    def args(self):
        """!Iterates over command arguments of the child serial program."""
        for arg in self._runner.args():
            yield arg
    def get_logger(self):
        """!Returns my logging.Logger that I use for log messages."""
        if self._logger is not None:
            return self._logger
        return self._runner.get_logger()
    def validate(self): 
        """!Does nothing."""
    def __eq__(self,other):
        """!Returns True if other is an MPISerial with the same Runner,
        False otherwise.
        @param other the other object to compare against."""
        return isinstance(other,MPISerial) and other._runner==self._runner
    def check_serial(self): 
        """!Returns (True,False) because this is a serial program
        (True,) and not a parallel program (,False)."""
        return (True,False)
    def isplainexe(self):
        """!Returns True if the child serial program is a plain
        executable, False otherwise.  See
        produtil.prog.Runner.isplainexe() for details."""
        return self._runner.isplainexe()
    def to_shell(self):
        """!Returns a POSIX sh version of the child serial program."""
        return self._runner.to_shell()
