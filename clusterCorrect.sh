home=/media/vym/Data/YaleBD
analysisDir=${home}/HLM_analysis/Despike_analysis/Amygdala_nucl_analysis_block/Output_data/right_HO_WH_amyg*diag*neg-pos_conn_block_NoOut
brainmask=${home}/Templates/MNI152_T1_2mm_brain_mask.nii.gz
statsMap=${analysisDir}/F_maps.nii.gz
residMap=${analysisDir}/resid_sd_mlmTest.nii.gz

# Make directory 
if [ ! -d ${analysisDir}/ClusterCorr ]; then
	mkdir ${analysisDir}/ClusterCorr
fi
cd ${analysisDir}/ClusterCorr

# Use Clustsim to look at minimum cluster size needed to keep FWE under 0.05
3dFWHMx -mask ${brainmask} -detrend -ACF -input ${residMap} > ACFval.txt


# Run 3dClustSim - be sure to update the smoothing parameters from the above command
3dClustSim -mask ${brainmask} -acf `tail -1 ACFval.txt | cut -d ' ' -f 1-6` -iter 10000 -seed 0 > ./cluster_simtables.txt

# Minimum cluster sizes - Set option -NN 1 (faces touching) and 2-sided thresholds (since working with t-maps)
# FWE Alpha 0.05, pthr = 0.001
minClust=`head -14 cluster_simtables.txt | tail -1 | cut -d ' ' -f 9-10` 
minClust=`printf "%.0f" $minClust`
### Apply cluster correction 

# Break  up mRi output into separate files
if [ ! -d ${analysisDir}/ClusterCorr/splitEff ]; then
	mkdir ${analysisDir}/ClusterCorr/splitEff
fi
cd ${analysisDir}/ClusterCorr/splitEff
fslsplit ${statsMap} effect_ -t 

# Get number of effects for current model
numEff=$((`fslinfo $statsMap | head -5 | tail -1 | cut -d' ' -f12`-1))

# Use cluster command in FSL
for i in $(seq -w 01 ${numEff})
do

input_map=effect_00${i}.nii.gz
output_cluster_table=effect_${i}_clustTable.txt
output_cluster_map=effect_${i}_clust.nii.gz

# Threshold indiv voxels at p>0.001 (df > 200)
if [  ${i} -le 05 ]; then
critT=10.95
elif [ ${i} -gt 05 ]; then
critT=7.00
fi

if [ ${i} -eq 02 ]; then
critT=8.93
elif [ ${i} -eq 03 ]; then
critT=7.00
elif [ ${i} -gt 07 ]; then
critT=4.69
fi

# This does the thresholding and clustering in one step:
cluster -i ${input_map} -t ${critT} --connectivity=6 -o ${output_cluster_map} | awk -v x=$minClust '$2 > x' > ${output_cluster_table}

# Threshold cluster map
# Identify minimum cluster ID  
minClustID=`tail -1 ${output_cluster_table} | cut -d$'\t' -f1`

if [[ $minClustID =~ ^-?[0-9]+$ ]]; then 

	# Parameters:
	cluster_thr_map=effect_${i}_clustThr.nii.gz
	
	# Threshold to exclude clusters smaller than minumum cluster size 
	fslmaths ${output_cluster_map} -thr ${minClustID} ${cluster_thr_map}

	# Mask originial stats map to include only voxels in surviving clusters
	correctMap=${analysisDir}/ClusterCorr/effect${i}_ClusterCorrect.nii.gz
	fslmaths ${input_map} -mas ${cluster_thr_map} ${correctMap}
fi
done

