#! /bin/env python

# Top level launching program for running WRF-Hydro model runs and
# testing.

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory

import sys, os

def setPythonPath():
    # Function to parse current working path and append proper paths to 
    # Python path for use of modules/libraries with this package.

    cwd = os.getcwd()
    
    # Point to 
    pathTmp = os.path.join(cwd,'../lib/produtil')
    sys.path.insert(0,pathTmp)
    pathTmp = os.path.join(cwd,'../lib/hydroFlowUtil')
    sys.path.insert(0,pathTmp)    
    pathTmp = os.path.join(cwd,'../lib/hydroFlowCycle')
    sys.path.insert(0,pathTmp)
    pathTmp = os.path.join(cwd,'../lib/hydroFlowTest')
    sys.path.insert(0,pathTmp)
    
def error_default(msg):
    print msg
    sys.exit(2)
    
def main():
    # Top level module to run the following opions:
    # 1) Run model for a given configuration.
    # 2) Test new model code against benchmark code through
    #    regression testing using a given configuration.

    args = sys.argv[1:]
    
    print 'alksdf'
    if len(args) < 4:
        error_default("ERROR: Insufficient number of arguments passed to hydroFlow")
        raise
    
    if args[1] == "RUN":
        if len(args) != 4:
            error_default("ERROR: Expected four arguments for model run, got" + len(args))
    elif args[1] == "TEST":
        if len(args) != 2:
            error_default("ERROR: Expected two arguments for code testing, got" + len(args))
    
if __name__ == '__main__':
    try:
        setPythonPath()
    except:
        error_default("ERROR in launching hydroFlow")
        
    try:
        main()
    except:
        error_default("ERROR in launching hydroFlow")