# Using fslmeants to extract mean cluster TS of original data, using masks results from MLM analysis
## Need to first apply a prior cluster-general ROI mask to stats map

effectNo=18

home=/media/vym/Data/YaleBD/HLM_analysis
cd ${home}

for laterality in left #left
do
for cluster in VS vlPFC
do

directory=${home}/Despike_analysis/Amygdala_HAMD_Bipolar/Output_data/${laterality}_HO_WH_amyg*diag*neg-pos_conn_block_NoOut

if [ ! -d ${directory}/ClusterCorr/Extracted_ts ];
then
mkdir ${directory}/ClusterCorr/Extracted_ts
fi

echo 'Extracting fslmeants for' ${laterality} ${scale} ${cluster}
fslmeants -i ${home}/allsubj_despike_block -o ${directory}/ClusterCorr/Extracted_ts/meanTS_${cluster}.txt -m ${directory}/ClusterCorr/effect${effectNo}_mlmCC_${cluster}.nii.gz

done
done
done
