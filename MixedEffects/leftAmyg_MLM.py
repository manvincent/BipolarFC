import multiprocessing
nume_cores=multiprocessing.cpu_count()
import os
home = '/media/vym/Data/YaleBD/HLM_analysis'
os.chdir('%s' %(home))

output = 'Despike_analysis/Amygdala_nucl_analysis_block/Output_data'
if not os.path.exists('%s/%s' %(home,output)):
     os.makedirs('%s/%s' %(home,output))


import numpy
import nibabel as nib
import rpy2
import scipy
import nipy

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects import FactorVector
from rpy2.robjects import Formula

from  rpy2.robjects.packages import importr

# Install necessary R libraries 
# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# packNames = ('lme4') # Specify libraries to install here
# from robjects.vectors import StrVector 
# installNames = [x for packnames]
# if len(installNames) > 0: 
#    utils.install_packages(StrVector(installNames))

r = robjects.r
r.require('lme4')
r.require("nlme")

stats = importr('stats')
base = importr('base')

maskImage = nib.load('%s/Masks/brain_mask.nii' %(home))
mask = maskImage.get_data()

# seed weighted time series (already extracted from seed ROI mask using fslmeants externally...)
laterality = 'left'
seedtype = 'HO_WH'

seedTS_dir = 'Despike_analysis/Amygdala_nucl_analysis_cont/Nucl_weighted_ts'
seed_file = numpy.loadtxt('%s/%s/%s_%s_weightedTS_block.txt' %(home, seedTS_dir, laterality, seedtype), dtype=float)
seed_vector = FloatVector(seed_file)
robjects.globalenv["weighted_ave_seed_beta"] = seed_vector

# input aggregated beta data
pp01 = nib.load('%s/allsubj_despike_block.nii' %(home))
s01 = pp01.get_data()



mlm = numpy.loadtxt('%s/allbeta_nohdr.csv' %(home), dtype=float, delimiter=',')
subject = FloatVector(mlm[:,0])
diagnosis = FloatVector(mlm[:,1])
condition = FloatVector(mlm[:,2])
block = FloatVector(mlm[:,3])
BPD = FloatVector(mlm[:,4])
BPM = FloatVector(mlm[:,5])
MDD = FloatVector(mlm[:,6])
BPD_BPM = FloatVector(mlm[:,7])
BPD_MDD = FloatVector(mlm[:,8])
BPM_BPD = FloatVector(mlm[:,9])
BPM_MDD = FloatVector(mlm[:,10])
MDD_BPD = FloatVector(mlm[:,11])
MDD_BPM = FloatVector(mlm[:,12])
MDD_Bipolar = FloatVector(mlm[:,13])
Bipolar_MDD = FloatVector(mlm[:,14])
pos_neut = FloatVector(mlm[:,15])
neg_neut = FloatVector(mlm[:,16])
posneg_neut = FloatVector(mlm[:,17])
pos_neg = FloatVector(mlm[:,18])
neg_pos = FloatVector(mlm[:,19])


robjects.globalenv["subject"] = subject
robjects.globalenv["diagnosis"] = diagnosis
robjects.globalenv["condition"] = condition
robjects.globalenv["block"] = block
robjects.globalenv["BPD"] = BPD
robjects.globalenv["BPM"] = BPM
robjects.globalenv["MDD"] = MDD
robjects.globalenv["neg_pos"] = neg_pos



T0_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T1_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T2_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T3_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T4_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T5_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T6_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T7_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T8_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T9_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T10_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T11_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T12_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T13_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T14_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T15_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T16_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T17_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T18_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
T19_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)

beta0_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta1_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta2_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta3_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta4_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta5_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta6_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta7_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta8_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta9_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta10_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta11_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta12_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta13_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta14_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta15_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta16_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta17_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta18_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
beta19_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)

F0_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F1_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F2_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F3_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F4_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F5_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F6_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F7_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F8_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
F9_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)

P0_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P1_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P2_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P3_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P4_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P5_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P6_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P7_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P8_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
P9_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)



residSD_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)
residVAR_img = numpy.zeros((s01.shape[0], s01.shape[1], s01.shape[2], 1), dtype=numpy.float32)

# Alternatively, to input covariate vectors extracted separately (e.g. through fslmeants) 
covar_file1 = numpy.loadtxt('%s/Covariates/WMsignal_meanbeta_block.txt' %(home), dtype=float)
ave_covarWM_beta = FloatVector(covar_file1)
robjects.globalenv["ave_covarWM_beta"] = ave_covarWM_beta

covar_file2 = numpy.loadtxt('%s/Covariates/CSFsignal_meanbeta_block.txt' %(home), dtype=float)
ave_covarCSF_beta = FloatVector(covar_file2)
robjects.globalenv["ave_covarCSF_beta"] = ave_covarCSF_beta

noConverge = []
for x in xrange(0, s01.shape[0]-1):
    print x
    for y in xrange(0, s01.shape[1]-1):
        for z in xrange(0, s01.shape[2]-1):
            if mask[x,y,z] > .5:
                try:
                    fMRI_data = FloatVector((s01[x,y,z,:]))
                    robjects.globalenv["fMRI_data"] = fMRI_data
                    lme_mri = r.lme(Formula("fMRI_data ~ weighted_ave_seed_beta * as.factor(diagnosis) * as.factor(neg_pos) + ave_covarCSF_beta + ave_covarWM_beta"), random = Formula('~1|subject'))
                    anova = r.anova(lme_mri)
                    F0_img[x,y,z] = anova[2][0]
                    F1_img[x,y,z] = anova[2][1]
                    F2_img[x,y,z] = anova[2][2]
                    F3_img[x,y,z] = anova[2][3]
                    F4_img[x,y,z] = anova[2][4]
                    F5_img[x,y,z] = anova[2][5]
                    F6_img[x,y,z] = anova[2][6]
                    F7_img[x,y,z] = anova[2][7]
                    F8_img[x,y,z] = anova[2][8]
                    F9_img[x,y,z] = anova[2][9]
                    P0_img[x,y,z] = anova[3][0]
                    P1_img[x,y,z] = anova[3][1]
                    P2_img[x,y,z] = anova[3][2]
                    P3_img[x,y,z] = anova[3][3]
                    P4_img[x,y,z] = anova[3][4]
                    P5_img[x,y,z] = anova[3][5]
                    P6_img[x,y,z] = anova[3][6]
                    P7_img[x,y,z] = anova[3][7]
                    P8_img[x,y,z] = anova[3][8]
                    P9_img[x,y,z] = anova[3][9]
                    betas = r.fixef(lme_mri)
                    ses = r.sqrt(r.diag(r.vcov(lme_mri)))
                    beta0_img[x,y,z] = betas[0]
                    beta1_img[x,y,z] = betas[1]
                    beta2_img[x,y,z] = betas[2]
                    beta3_img[x,y,z] = betas[3]
                    beta4_img[x,y,z] = betas[4]
                    beta5_img[x,y,z] = betas[5]
                    beta6_img[x,y,z] = betas[6]
                    beta7_img[x,y,z] = betas[7]
                    beta8_img[x,y,z] = betas[8]
                    beta9_img[x,y,z] = betas[9]
                    beta10_img[x,y,z] = betas[10]
                    beta11_img[x,y,z] = betas[11]
                    beta12_img[x,y,z] = betas[12]
                    beta13_img[x,y,z] = betas[13]
                    beta14_img[x,y,z] = betas[14]
                    beta15_img[x,y,z] = betas[15]
                    beta16_img[x,y,z] = betas[16]
                    beta17_img[x,y,z] = betas[17]
                    beta18_img[x,y,z] = betas[18]
                    beta19_img[x,y,z] = betas[19]
                    T0_img[x,y,z] = betas[0] / ses[0]
                    T1_img[x,y,z] = betas[1] / ses[1]
                    T2_img[x,y,z] = betas[2] / ses[2]
                    T3_img[x,y,z] = betas[3] / ses[3]
                    T4_img[x,y,z] = betas[4] / ses[4]
                    T5_img[x,y,z] = betas[5] / ses[5]
                    T6_img[x,y,z] = betas[6] / ses[6]
                    T7_img[x,y,z] = betas[7] / ses[7]
                    T8_img[x,y,z] = betas[8] / ses[8]
                    T9_img[x,y,z] = betas[9] / ses[9]
                    T10_img[x,y,z] = betas[10] / ses[10]
                    T11_img[x,y,z] = betas[11] / ses[11]
                    T12_img[x,y,z] = betas[12] / ses[12]
                    T13_img[x,y,z] = betas[13] / ses[13]
                    T14_img[x,y,z] = betas[14] / ses[14]
                    T15_img[x,y,z] = betas[15] / ses[15]
                    T16_img[x,y,z] = betas[16] / ses[16]
                    T17_img[x,y,z] = betas[17] / ses[17]
                    T18_img[x,y,z] = betas[18] / ses[18]
                    T19_img[x,y,z] = betas[19] / ses[19]
                    resid = r.VarCorr(lme_mri)
                    residVAR_img[x,y,z] = resid[1]
                    residSD_img[x,y,z] = resid[3]
                except:
                    print[x, y, z, 'did not converge']
                    noConverge.append([x, y, z])


analysis = '%s_%s_amyg*diag*neg-pos_conn_block_NoOut' %(laterality, seedtype)
if not os.path.exists('%s/%s/%s' %(home,output,analysis)):
     os.makedirs('%s/%s/%s' %(home,output,analysis))
                    
tempImg = nib.Nifti1Image(F0_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F0_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F1_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F1_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F2_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F2_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F3_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F3_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F4_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F4_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F5_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F5_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F6_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F6_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F7_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F7_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F8_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F8_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(F9_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/F9_mlmTest.nii.gz' %(home,output,analysis))
                         
tempImg = nib.Nifti1Image(P0_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P0_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P1_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P1_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P2_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P2_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P3_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P3_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P4_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P4_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P5_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P5_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P6_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P6_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P7_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P7_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P8_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P8_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(P9_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/P9_mlmTest.nii.gz' %(home,output,analysis))

tempImg = nib.Nifti1Image(beta0_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B0_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta1_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B1_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta2_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B2_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta3_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B3_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta4_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B4_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta5_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B5_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta6_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B6_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta7_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B7_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta8_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B8_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta9_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B9_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta10_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B10_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta11_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B11_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta12_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B12_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta13_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B13_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta14_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B14_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta15_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B15_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta16_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B16_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta17_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B17_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta18_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B18_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(beta19_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/B19_mlmTest.nii.gz' %(home,output,analysis))

tempImg = nib.Nifti1Image(T0_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T0_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T1_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T1_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T2_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T2_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T3_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T3_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T4_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T4_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T5_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T5_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T6_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T6_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T7_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T7_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T8_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T8_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T9_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T9_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T10_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T10_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T11_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T11_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T12_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T12_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T13_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T13_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T14_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T14_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T15_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T15_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T16_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T16_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T17_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T17_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T18_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T18_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(T19_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/T19_mlmTest.nii.gz' %(home,output,analysis))

     
tempImg = nib.Nifti1Image(residSD_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/resid_sd_mlmTest.nii.gz' %(home,output,analysis))
tempImg = nib.Nifti1Image(residVAR_img, pp01.get_affine())
tempImg.to_filename('%s/%s/%s/resid_var_mlmTest.nii.gz' %(home,output,analysis))

