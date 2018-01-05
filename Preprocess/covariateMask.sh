######
home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI
subject_list=${home}/subjects.txt
#######

cd $home

## Segmentation using FAST on each subject's T1
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/
mkdir WM_CSFconfounds
echo 'Segmentation for' $subject
/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -g --nopve -o /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/${subject}_mprage_brain /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/anat/mprage_brain
# Output: 0=CSF; 1=GM; 2=WM 
cd $home
done

## FLIRT transform segmentation to standard space: WM and CSF
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/
echo 'Transform WM map for' $subject # In *.feat/reg/
/usr/local/fsl/bin/flirt -in /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/${subject}_mprage_brain_seg_2.nii.gz -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain -out /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/${subject}_standard_WM_mask -omat /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/standard_WM_mask.mat -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear

cd ${home}/${subject}/
echo 'Transform CSF map for' $subject
/usr/local/fsl/bin/flirt -in /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/${subject}_mprage_brain_seg_0.nii.gz -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain -out /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/${subject}_standard_CSF_mask -omat /Volumes/MacintoshHD3/YaleClinicalData/fMRI/${subject}/WM_CSFconfounds/standard_CSF_mask.mat -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear

cd $home
done

## Create binary masks out of standard space WM and CSF maps
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/
echo 'Binarize WM map to create mask for' $subject
fslmaths ./WM_CSFconfounds/${subject}_standard_WM_mask -thr 0.5 -bin ./WM_CSFconfounds/${subject}_standard_WM_mask50_bin

cd ${home}/${subject}/
echo 'Binarize CSF map to create mask for' $subject
fslmaths ./WM_CSFconfounds/${subject}_standard_CSF_mask -thr 0.5 -bin ./WM_CSFconfounds/${subject}_standard_CSF_mask50_bin

cd $home
done


## Extracting mean ts for masked, standard WM and CSF
for subject in $(cat ${subject_list})
do
cd ${home}/${subject}/
echo 'Extracting mean TS in WM for' $subject
fslmeants -i ./preprocessed/${subject}_prep.nii -m ./WM_CSFconfounds/${subject}_standard_WM_mask50_bin -o ./WM_CSFconfounds/${subject}_WMconfound_meants.txt

cd ${home}/${subject}/
echo 'Extracting mean TS in CSF for' $subject
fslmeants -i ./preprocessed/${subject}_prep.nii -m ./WM_CSFconfounds/${subject}_standard_CSF_mask50_bin -o ./WM_CSFconfounds/${subject}_CSFconfound_meants.txt

cd $home
done



printf 'FINISHED!\n'


