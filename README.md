# groupfinder_masking
A repo for checking how masking might change groupfinder results. 

## Intro
We take WAVES Wide as our example region, using the sharks mock catalog and the WAVES input catalog starmask. We aim to explore how a survey starmask may effect resulting group properties, and if biasing can occur. 

![Alt text](plots/masked_galaxy_stats.png)

This plot shows how many groups are partially masked, and the extent of the masking. Bottom 2 panels don't include full masked groups.

The WAVES Wide mask reduces the rectangular footprint by approx 7%. This means there should be 7% fewer galaxies and groups of a given multiplicity (N) in the sample, if they are uniformly distributed. It is an unresolved question if galaxy groups, which are not uniform in their size, do not experience any second order affects from a starmask being applied. The plot above shows that groups with more members more likely to be masked, but less likely to be mostly masked.

This plot shows the difference in N in masked and unmasked groups.

![Alt text](plots/group_multiplicity_bootstrap.png)

First we examine the effects of the masking on the groups.
This plot is a stacked density plot of 50 random groups in a given N bin. It shows that for groups with a larger N. This plot, as well as the plot ab. This for groups ids given in the sharks simulation.

![Alt text](plots/stacked_star_masks_by_group_multiplicity.png)

And in terms of the galaxies masked by the sources in the plot above, you get this. Same set up.

![Alt text](plots/stacked_masked_galaxy_density_by_group_multiplicity.png)

They key take aways from these plots are 
1) Group abundance per N scales generally with footprint area.
2) Bigger groups seem likelier to retain similar properties to their unmasked counterpart. This is because a group with 10 members that looses 7% of its area will on average have 9 members. This 9 member group will still have mostly the same r50, and $\sigma_{gapper}$ as the unmasked group.
3) This means masking causes lower multiplicity groups to be systematically noiser than their counterparts at higher multiplicites. The question is how this error compares to errors in the estimators for derived group properties in used in real groupfinders. 
4) It is unclear if this effect scales down into the group catalog level, when built with a group finder.

To examine this further, we look at the abundance of derived properties from masked and unmasked groups. In particular, we look at r50 and $\sigma_{gapper}$ as these are use in many mass estimates. This was not extended to the SHMR or LHMR, as you can see in the first plot that larger groups dont loose that much stellar mass. 

Firstly $\sigma_{gapper}$:

![Alt text](plots/id_fof_group_velocity_dispersion_gap_histograms_log_counts_area_corrected.png)

Plot of $\sigma_{gapper}$ abundances. Errors from bootstrapping. Area difference between masked and unmasked is 7.2%. Seems to be systematically higher at at higher velocity dispersions. 

Now r50:

![Alt text](plots/id_fof_group_r50_histograms_log_counts_area_corrected.png)

Plot of R50 abundances. Errors from bootstrapping. Area difference between masked and unmasked is 7.2%. As expected, we see r50 go systematically higher at higher values. Although for bins of r50 with less than 100 members the errors increase too much to be meaningful.

And finally we check that this propigates down into the dynamical mass, which it does:

![Alt text](plots/id_fof_dynam_mass_no_lum_corr_area_corrected.png)

Same as above plots, this time dynamical mass. Still biased at high mass end!

Seems to me like this would cause systematics in any HMF, where your off on the higher mass end by the fraction you upweight your area correction with. Having a look how much group properties changed vs original unmasked group multiplicity looks like this:

![Alt text](plots/unmasked_n_vs_props.png)

Which again shows that groups with larger N have largely unchanged properties. As you can see in the first figure they are also vanishingly unlikely to be full masked.

But, we still need to forward model this using a real groupfinder. 
So, lets do so with these plots again:

![Alt text](plots/Nessie_group_velocity_dispersion_gap_histograms_log_counts_area_corrected.png)

![Alt text](plots/Nessie_group_r50_histograms_log_counts_area_corrected.png)

![Alt text](plots/Nessie_dynam_mass_no_lum_corr_area_corrected.png)

And would you look at that, this whole effect seems to go away. Now, to see why, lets take a look how well nessie recovers its groups from the masked to the unmasked version. Here, we run nessie on the masked and unmasked catalog. We then mask the unmask group ids, to find a S score for these two groups. This is what we get:

![Alt text](plots/nessie_masked_vs_unmasked_S_score.png)

So, as this S score scales ^(1/4) with efficency and purity, what we see is the groups are about 99 percent the same between the group finder run on the masked catalog, and the groupfinder run on the unmasked catalog which then has masked galaxies removed afterwards. So, these two catalogs are basically identical. So, the systamtic offset goes away, and gets washed out by how noisy groupfinding is anyway. 

Now, as a next step i can find out what makes the group finders noisy in this way (id guess from shattering fof chains, and large groups being poor anyway) but should probably stop here to see how interesting others find it.


## Questions

* Does group edge, like in gama, get rid of any of these issues?
* Why do groupfinders not seem to experience any issues? (see line above)