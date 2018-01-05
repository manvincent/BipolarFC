######
home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI
subject_list=${home}/subjects.txt
#######

cd $home

## Rename anat and epi files
for subject in $(cat ${subject_list})
do
echo 'Renaming files for' $subject
cd ./${subject}/anat
mv S* mprage.nii
cd ../epi
mv S* func_epi.nii
cd $home
done

## Compress files
for subject in $(cat ${subject_list})
do
echo 'Compressing files for' $subject
cd ./${subject}/anat
gzip mprage.nii
cd ../epi
gzip func_epi.nii
cd $home
done

## Brain extraction on anatomical
for subject in $(cat ${subject_list})
do
echo 'Brain extract for' $subject
cd $home
cd ./${subject}/anat
bet mprage.nii.gz mprage_brain.nii.gz -R
done

## Creating motion confound file to use as regressor
for subject in $(cat ${subject_list})
do
echo 'Motion confound for' $subject
cd ${home}/${subject}
mkdir ./preprocessed
fsl_motion_outliers -i ./epi/func_epi.nii.gz -o ./preprocessed/${subject}_motion_confound -p ./preprocessed/${subject}_metric_plot --dummy=1 -v

done


printf 'Finished!'

