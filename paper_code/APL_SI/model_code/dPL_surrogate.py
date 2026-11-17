import sys  
sys.path.append(r'/share/home/dq152/APL_SI/model_code')
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
import os
import torch

# GPU setting 
gpuid = 0
torch.cuda.set_device(gpuid)

cDir = r'/share/home/dq152/APL_SI/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND') 

subsetLst = 'South_China120x220'

tRangeLst = [20050101, 20051231] 

hiddenSizeLst = 512  

df = dbCsv.DataframeCsv(rootDB=rootDB, subset=subsetLst, tRange=tRangeLst)

# ===== varForcing =====  
varForcing = [
    'ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10'
]
# ===== parameters ===== 
varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
            'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
            'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
            'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
            'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
            #'csol_l1','csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
            #'k_solids_l1','k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
            #'tkdry_l1','tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
            #'tksatf_l1','tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
            #'tksatu_l1','tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8'
]

forcing = df.getDataTs(varForcing, doNorm=True, rmNan=True) 
parameters = df.getDataConst(varConst, doNorm=True, rmNan=True) 

target = df.getDataTs([#'ET',
                       #'runoff',
                       'SM_layer123'#,'SM_layer45','SM_layer67','SM_layer8910',
                       #'ST_layer123','ST_layer45','ST_layer67','ST_layer8910'
                         ],doNorm=True, rmNan=True)

print("=== 形状 ===")
print(f"forcing: {forcing.shape}")
print(f"parameters: {parameters.shape}")
print(f"target: {target.shape}")

nx = forcing.shape[-1] + parameters.shape[-1]
ny = target.shape[-1]    

path_surrogate = '/share/home/dq152/APL_SI/script_data/def_ERA5LAND/SM_layer123/surrogate_model'
os.makedirs(path_surrogate,exist_ok=True)
outFolder = os.path.join(cDir, path_surrogate) 

epoch=1000

model_PF = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=hiddenSizeLst)
lossFun_PF = crit.RmseLoss()
model_PF = train.trainModel(
    model_PF, forcing, target, parameters, lossFun_PF, 
    nEpoch=epoch, 
    miniBatch=[300, 180], 
    saveEpoch=10, 
    optLr=1,                  # 初始学习率
    optRho=0.95,              # Adadelta参数
    optWd=0.00001,            # 权重衰减
    saveFolder=outFolder, 
    lyrMode=0, 
    scaleTrans=False,
    # ===== 优化器和调度器配置 =====
    optimizer_type='adadelta',
    scheduler_type='cosine_warmup',
    warmup_epochs=100,        # 100个epoch线性预热
    T_max=900,                # 900个epoch余弦退火
    eta_min=0.1,              # 最终学习率0.1
    csv_name='run_cosine_warmup'
)

modelName = 'SM_layer123'
train.saveModel(outFolder, model_PF, epoch, modelName=modelName)
