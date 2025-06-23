## In code documentation

Code should be documented according to
https://peps.python.org/pep-0257/ This inline documentation can be
translated to web pages as described below (@avihgna to add directions
for generating documentation).

## Mains of primary modules

Each of the primary modules should have a main that runs in one of two modes:
(See kmeans.py for an example)

1) Test mode: specified by the command line parameter --test. This is
a mode in which the main tests the correctness of the code in the
module. The test should be completely autonomous, the only parameter
that needs to be fiven is --test and the test determined whether the
code passed or failed the test -- no human judgement needed.

2) non-test mode = modular mode: in this mode the main reads input
files that are produced by the modules that precede it and outputs
files that are read by the modules that follow it. This allows running
each main module in isolation, without incurring the cost of running
the preceding modules. It is also advantageoud because the input to
the module is fixed and does not change from run to run.

# Separation of code and data

It is in general a bad idea to store data files / images on
github. Those files tend to be large and diff does not compress
them. We therefor store large files either locally on the computer or
on a shared google drive. @shivany to explain how to do this in a way
that would allow the software to reach the datafiles from any
computer.

# git branching

In general, each programmer works on their own branch. However
branches other than the master should be short-living so that the code
does not diverge. As a rule of thumb, merge your branch with main at
least once a week, once merged, delete the branch (not just on your
computer but upstream on github).


