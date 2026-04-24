# Notes

## Lattice generator

Implemented a Python-ized version of the original that looks like it is working properly. Can't directly compare to previous implementation because this one includes some randomness, so every lattice is different.

## Path length for disconnected graphs

How to handle graphs that aren't fully connected?

For path length, my implementation works the same as using NetworkX, as long as the graph is connected.

From Springer Encyclopedia of Systems Biology (emphasis mine): "In case of an unconnected graph, the characteristic path length is infinite, as the number of edges between two unconnected vertices is considered infinite. In this case, formula (1) is often modified to **sum over just all connected vertex pairs**. Alternatively, other measures such as the harmonic mean or the average inverse path length can be used." So maybe using the LCC only is okay. This could change depending on thresholding strategies for FC.
Link: https://link.springer.com/rwe/10.1007/978-1-4419-9863-7_1460

Could also look into using global efficiency, but this not the same thing.

Original code just replaces infite distances with 0, which artificially deflates the path length, but maybe that's just how it has to be?

## Thresholding options
* Absolute threshold - varies for every SPI/scale
* Percentile/proportional thresholds - scale agnostic, but inter- and intra-subject effects can cause problems. Maybe it doesn't matter for this since I'm averaging over subjects and I don't care about individiaul differences?
    * This seems to not affect the path length too much, which is probably good
    * Original clustering is ~0.23, proportional thresholding *increases* it to $[0.27, 0.37]$.
    * Original paper used this method, I think at 9.2%
* Maximum/minimum spanning tree (makes it too sparse) but it stays fully connected
* Consensus thresholding: Eliminate edges that do not have strength of at least 𝝆 in at least X% of subjects

## Plan
* Figure out how to run MATLAB code to test it -- DONE
    * Save all matrices to `.mat` using `scipy.io.savemat` and make text file with paths
    * Write script to run `small_world_propensity.mat`
    * Write batch script to pass in matrix, run matlab file, save result to file
        * Make sure to include subject ID in output somehow for debugging
* Iterate over various proportional thresholds for just covariance FC
    * Show how clustering changes
    * Show how path length changes
    * Show how SWP changes
        * Make sure that it's producing reasonable values
    * Also do modularity
    * Maybe also do global efficiency?
    * Other graph metrics of interest?
* If there's time, repeat for other FC measures

## Thurs morning plan
1. Implement consensus thresholding
2. Iterate over proportional thresholds [5, 50]