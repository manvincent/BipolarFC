home=/Volumes/MacintoshHD3/YaleClinicalData/fMRI
subjectlist=${home}/subjects.txt
cd ${home} 

for subject in $(cat ${subjectlist})
do

echo 'Extracting for' ${subject}

# Pull out first column (rot_x) into a temp file
cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $1}' > ${home}/${subject}/preprocessed/rot_x_temp.txt
# Convert scientific notation in temp file and save as rot_x file
cat ${home}/${subject}/preprocessed/rot_x_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/rot_x.txt
# Remove temp file  
rm ${home}/${subject}/preprocessed/rot_x_temp.txt

cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $2}' > ${home}/${subject}/preprocessed/rot_y_temp.txt
cat ${home}/${subject}/preprocessed/rot_y_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/rot_y.txt
rm ${home}/${subject}/preprocessed/rot_y_temp.txt

cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $3}' > ${home}/${subject}/preprocessed/rot_z_temp.txt
cat ${home}/${subject}/preprocessed/rot_z_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/rot_z.txt
rm ${home}/${subject}/preprocessed/rot_z_temp.txt

cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $4}' > ${home}/${subject}/preprocessed/trans_x_temp.txt
cat ${home}/${subject}/preprocessed/trans_x_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/trans_x.txt
rm ${home}/${subject}/preprocessed/trans_x_temp.txt


cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $5}' > ${home}/${subject}/preprocessed/trans_y_temp.txt
cat ${home}/${subject}/preprocessed/trans_y_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/trans_y.txt
rm ${home}/${subject}/preprocessed/trans_y_temp.txt


cat ${home}/${subject}/first_level.feat/mc/prefiltered_func_data_mcf.par | sed 's/\|/ /'|awk '{print $6}' > ${home}/${subject}/preprocessed/trans_z_temp.txt
cat ${home}/${subject}/preprocessed/trans_z_temp.txt | awk '{ print sprintf("%.9f", $1); }' > ${home}/${subject}/preprocessed/trans_z.txt
rm ${home}/${subject}/preprocessed/trans_z_temp.txt

done

printf 'Finished!\n'
