#To create mlm output masks of the ROI clusters

home=/media/vym/Data/YaleBD/HLM_analysis/Despike_analysis
cd $home
effectNo=18

for laterality in left right
do
for cluster in vlPFC VS
do

## Need to first identify the cluster ID of the cluster of interest, from the F*_cluster_thr.nii.gz file. Here I'm interested in leftOFC, rightOFC, and rightmFG
if [ ${laterality} = "left" ]; then
	if [ ${cluster} = "vlPFC" ]; then
		cluster_ID=569
	elif [ ${cluster} = "VS" ]; then
		cluster_ID=568
    fi
elif [ ${laterality} = "right" ]; then
    if [ ${cluster} = "vlPFC" ]; then
        cluster_ID=479
    elif [ ${cluster} = "VS" ]; then
        cluster_ID=478
    fi
fi


directory=${home}/Amygdala_nucl_analysis_block/Output_data/${laterality}_HO_WH_amyg*diag*neg-pos_conn_block_NoOut
# Create mask of just the cluster
cluster_thr_map=${directory}/ClusterCorr/splitEff/effect_${effectNo}_clustThr.nii.gz
fslmaths ${cluster_thr_map} -thr ${cluster_ID} -uthr ${cluster_ID} ${directory}/ClusterCorr/effect${effectNo}_cluster_${cluster}.nii.gz

# Get mlm map of just the cluster 
input_map=${directory}/F${effectNo}_mlmTest.nii.gz
single_cluster_map=${directory}/ClusterCorr/effect${effectNo}_cluster_${cluster}.nii.gz 
corrected_fmap=${directory}/ClusterCorr/effect${effectNo}_mlmCC_${cluster}.nii.gz
fslmaths ${input_map} -mas ${single_cluster_map} ${corrected_fmap}

done
done
done

