# groupfinder_masking
A repo for checking how masking might change groupfinder results. 
We can compare deep, north and south WAVES regions as well as GAMA to see if masking density has an proportional affect.

## Qs to answer in the case of a WAVES like simulation

* What fraction of groups are missing members from masking total?
* What fraction of groups are missing members per group N? - Remove groups missing all members.
* What fraction of groups loose a meaningful amount of stellar mass? 
* What fraction of groups loose a meaningful amount of stellar mass per N? - Heatmap, N, stellar mass missing, density.
* What fraction of the groups loose a meaningful amount of r50 and gapper vel dispersion ?
* When applying nessie, how are the calculated r50 and gapper vel disperson affected?
* How does this propagate to a mass A - like HMF?

* I need to now start thinking about my errors. This needs to be in the mask fraction perhaps?
- I could also just say this mask is basically bright GAIA (not true for waves-s where there is a big GC), and pick different non galatic sky patches to mask out. Could also randomly drop that GC in somewhere. Also need to consider if the ghostmasks can be done as well.

* I also need to compare WAVESwideN/S, WAVESdeep, and GAMA.

* Is a correction to dynamical or stellar mass possible? How consistent are these results?
- Would this mean fitting a function to counts vs gapper vel?

[test](plots/masked_galaxy_stats.png)
