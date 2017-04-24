######
home=<path to home>
cd $home

##################################################################################
## full path to subject_list
subject_list=${home}/subjects.txt
## location of onset files
onset_data=${home}/st_onsets.txt

##################################################################################

## Single-trial Deconvolve 
for subject in $(cat ${subject_list})
do

if [ ! -d ${home}/${subject}/ ]
then
mkdir ${home}/${subject}/
fi

if [ ! -d ${home}/${subject}/st_output ]
then
mkdir ${home}/${subject}/st_output
fi
done

# Single-trial deconvolve
## NB: The input file has not been cleaned of WM and CSF confounds!
for subject in $(cat ${subject_list})
do
echo 'Single-trial Deconvolve for' ${subject} 
3dDeconvolve -input ../${subject}/preprocessed/${subject}_prep.nii \
-num_stimts 7 \
-stim_times_IM 1 ${onset_data} 'BLOCK(20,1)' -stim_label 1 onsets \
-stim_file 2 ../${subject}/preprocessed/rot_x.txt \
-stim_label 2 rot_x \
-stim_base 2 \
-stim_maxlag 2 1 \
-stim_file 3 ../${subject}/preprocessed/rot_y.txt \
-stim_label 3 rot_y \
-stim_base 3 \
-stim_maxlag 3 1 \
-stim_file 4 ../${subject}/preprocessed/rot_z.txt \
-stim_label 4 rot_z \
-stim_base 4 \
-stim_maxlag 4 1 \
-stim_file 5 ../${subject}/preprocessed/trans_x.txt \
-stim_label 5 trans_x \
-stim_base 5 \
-stim_maxlag 5 1 \
-stim_file 6 ../${subject}/preprocessed/trans_y.txt \
-stim_label 6 trans_y \
-stim_base 6 \
-stim_maxlag 6 1 \
-stim_file 7 ../${subject}/preprocessed/trans_z.txt \
-stim_label 7 trans_z \
-stim_base 7 \
-stim_maxlag 7 1 \
-bucket ${home}/${subject}/st_output/st_stats_block -nofullf_atall

# Convert AFNI BRIK files to .nii files
echo 'Converting BRIK to NIFTI for' ${subject}
3dAFNItoNIFTI -prefix ${home}/${subject}/st_output/st_statsout_block ${home}/${subject}/st_output/st_stats_block+tlrc.BRIK

# Concatenate AFNI sub-bricks in the .nii file
echo 'Concatenate sub-briks in .nii file for' ${subject}
cd ${home}/${subject}/st_output
3dTcat -prefix ${subject}_block_fsl.nii 'st_statsout_block.nii[0..$]' -tr 2

# Depsike
echo 'Despiking for' ${subject}
3dDespike -ignore 0 -nomask -prefix despiked_block ${home}/${subject}/st_output/${subject}_block_fsl.nii

# Convert despiked BRIK file to .nii file, again
echo 'Converting despike BRIK to .nii for' ${subject}
3dAFNItoNIFTI -prefix ${home}/${subject}/st_output/despiked_${subject}_block_fsl.nii  ${home}/${subject}/st_output/despiked_block+tlrc.BRIK

cd $home
done

printf 'Finished!'
