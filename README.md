# groupfinder_masking
A repo for checking how masking might change groupfinder results. 
We can compare deep, north and south WAVES regions as well as GAMA to see if masking density has an proportional affect.

## Qs to answer in the case of a WAVES like simulation

* What fraction of groups are missing members from masking total?
* What fraction of groups are missing members per group N? - Remove groups missing all members.
* What fraction of groups loose a meaningful amount of stellar mass? 
* What fraction of groups loose a meaningful amount of stellar mass per N? - Heatmap, N, stellar mass missing, density.
* What fraction of the groups loose a meaningful amount of r50 and gapper vel dispersion per N?
* When applying nessie, how do the S scores compare between a masked and an unmasked catalog?

* Does masking have the same effect as dropping sources at random from the catalog on each of these meterics, to the same given global completeness?
- If so, does a global completeness correction work better when using nessie?
