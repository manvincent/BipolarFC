# Script to create allsubj_despiked concatenated .nii file! 

home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI/single_trials_analysis/HLM_analysis
cd ${home}

if [ ! -d ${home}/temp/ ]
then
mkdir ${home}/temp/
fi

sub_home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI/single_trials_analysis/
cd ${sub_home}

subject_list=${sub_home}/subjects.txt
# Copying each subject's despiked file to temp
for subject in $(cat ${subject_list})
do
echo 'Copying' ${subject} 'despiked nii to temp'
cp ${sub_home}/${subject}/st_output/despiked_${subject}_fsl.nii ${home}/temp
done

# Concatenating to allsubj_despike.nii
cd ${home}/temp
echo 'Concatenating...'
fslmerge -a ${home}/allsubj_despike ${home}/temp/*.nii

# Remove temp file
echo 'Removing temp files'
rm -R ${home}/temp

# Unzip concatenated despike file
echo 'Unzipping...'
gzip -d ${home}/allsubj_despike.nii 

printf 'Finished!\n'


