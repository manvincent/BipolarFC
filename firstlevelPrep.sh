######
home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI
subject_list=${home}/subjects.txt
#######

cd $home

## Create design files for each subject, from subject BPD_S01
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}
echo 'Creating Design for ' $subject
sed 's/BPD_S01/'$subject'/g' ../BPD_S01/first_level.feat/design.fsf > ./${subject}_design.fsf
cd $home
done

## Running the pre-processing for each subject
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}
echo 'Running preprocessing for' $subject
feat ${subject}_design.fsf
cd $home
done

## Creating the preprocessed image file, post-preprocessing
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/first_level.feat
echo 'constructing preprocessed .nii for' $subject
applywarp --ref=./reg/standard.nii.gz --in=filtered_func_data.nii.gz --out=../preprocessed/${subject}_prep.nii.gz --premat=./reg/example_func2highres.mat --warp=./reg/highres2standard_warp.nii.gz
cd $home
done

## Alternatively, getting the preprocessed image file to standard space:
## This does the same thing as the applywarp command above 
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/
echo 'constructing preprocessed .nii for' $subject
featregapply ./first_level.feat -l filtered_func_data
cd $home
done

