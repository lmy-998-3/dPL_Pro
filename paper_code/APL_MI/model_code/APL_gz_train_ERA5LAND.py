##########使用余弦退火学习率
import sys
sys.path.append(r'/share/home/dq152/APL_MI/model_code') 
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
import os
import torch
import numpy as np
import time
import math
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

# ===== 自定义预热余弦退火调度器 =====
class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """带预热的余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, warmup_start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热阶段
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.T_max)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay 
                    for base_lr in self.base_lrs]

# GPU setting
traingpuid = 0
torch.cuda.set_device(traingpuid)

## database
cDir = r'/share/home/dq152/APL_MI/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND')

## data subset
subsetLst=['South_China120x220']
tRangeLst=[[20050101, 20051231]]  ## training period
## hidden size of dPL
hiddenSizeLst =[512] #256,512

targetVar=[#'e',
           #'ro',
           'SM_1',#'SM_2','SM_3','SM_4',   
           'ST_1',#'ST_2','ST_3','ST_4'
           ] 

trainYear=int(str(tRangeLst[0][1])[0:4])-int(str(tRangeLst[0][0])[0:4])

df = dbCsv.DataframeCsv(rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLst[0])

# ===== Forcing =====
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
# ===== parameters  =====
varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
            'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
            'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
            'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
            'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
            'csol_l1',#'csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
            'k_solids_l1',#'k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
            'tkdry_l1',#'tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
            'tksatf_l1',#'tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
            'tksatu_l1',#'tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8' 
            ]                            

#===== std  =====
varStd = ['k_s_l1_std',#'k_s_l2_std','k_s_l3_std','k_s_l4_std','k_s_l5_std','k_s_l6_std','k_s_l7_std','k_s_l8_std',
          'theta_s_l1_std',#'theta_s_l2_std','theta_s_l3_std','theta_s_l4_std','theta_s_l5_std','theta_s_l6_std','theta_s_l7_std','theta_s_l8_std',
          'alpha_std_l1',#'alpha_std_l2','alpha_std_l3','alpha_std_l4','alpha_std_l5','alpha_std_l6','alpha_std_l7','alpha_std_l8',
          'n_std_l1',#'n_std_l2','n_std_l3','n_std_l4','n_std_l5','n_std_l6','n_std_l7','n_std_l8',
          'theta_r_std_l1',#'theta_r_std_l2','theta_r_std_l3','theta_r_std_l4','theta_r_std_l5','theta_r_std_l6','theta_r_std_l7','theta_r_std_l8',
          'csol_l1_std',#'csol_l2_std','csol_l3_std','csol_l4_std','csol_l5_std','csol_l6_std','csol_l7_std','csol_l8_std',
          'k_solids_l1_std',#'k_solids_l2_std','k_solids_l3_std','k_solids_l4_std','k_solids_l5_std','k_solids_l6_std','k_solids_l7_std','k_solids_l8_std',
          'tkdry_l1_std',#'tkdry_l2_std','tkdry_l3_std','tkdry_l4_std','tkdry_l5_std','tkdry_l6_std','tkdry_l7_std','tkdry_l8_std',
          'tksatf_l1_std',#'tksatf_l2_std','tksatf_l3_std','tksatf_l4_std','tksatf_l5_std','tksatf_l6_std','tksatf_l7_std','tksatf_l8_std',
          'tksatu_l1_std',#'tksatu_l2_std','tksatu_l3_std','tksatu_l4_std','tksatu_l5_std','tksatu_l6_std','tksatu_l7_std','tksatu_l8_std' 
           ]  

# ===== static attributes =====
varRaw = ['SAND_depth_01',# 'SAND_depth_02', 'SAND_depth_03', 'SAND_depth_04', 'SAND_depth_05', 'SAND_depth_06', 'SAND_depth_07', 'SAND_depth_08', 
          'CLAY_depth_01',# 'CLAY_depth_02','CLAY_depth_03','CLAY_depth_04','CLAY_depth_05','CLAY_depth_06','CLAY_depth_07','CLAY_depth_08',
          'GRAV_depth_01',# 'GRAV_depth_02','GRAV_depth_03','GRAV_depth_04','GRAV_depth_05','GRAV_depth_06','GRAV_depth_07','GRAV_depth_08',
          'BD_depth_01',#'BD_depth_02','BD_depth_03','BD_depth_04','BD_depth_05','BD_depth_06','BD_depth_07','BD_depth_08',
          'OC_depth_01',#'OC_depth_02','OC_depth_03','OC_depth_04','OC_depth_05','OC_depth_06','OC_depth_07','OC_depth_08',
          ###############
          'forest_height','landtype','topography']

## read database
forcing = df.getDataTs(varForcing, doNorm=True, rmNan=True) 
parameters = df.getDataConst(varConst, doNorm=False, rmNan=True) 
# ===== 读取标准差 =====
std_values = df.getDataConst(varStd, doNorm=False, rmNan=True)
rawData = df.getDataConst(varRaw, doNorm=True, rmNan=True)          
target = df.getDataTs(targetVar, doNorm=True, rmNan=False)  
target_GF = df.getDataTs(targetVar, doNorm=True, rmNan=True)  

modelTarget = target
filename = '/share/home/dq152/APL_MI/script_data/def_ERA5LAND/SM1+ST1/surrogate_model_2/model_Ep1000.pt'

model_loaded_PF_SMAP = torch.load(filename) 
model_loaded_PF_SMAP.eval()

print(f"调试信息:")
print(f"  forcing特征数: {forcing.shape[-1]} ")
print(f"  rawData特征数: {rawData.shape[-1]}")
print(f"  目标变量数: {len(targetVar)} ")
print(f"  参数数量: {len(varConst)} ")

nx = (forcing.shape[-1] + rawData.shape[-1], rawData.shape[-1], len(varConst))
ny = target.shape[-1]

# ======= Epoch setting =======
epoch=1000
saveEpoch=10
minibatch=[300, 180]

path_Inv = 'def_ERA5LAND/SM1+ST1/gz_model_2/'

# print(f"参数设置{csv_name}")
outFolder = os.path.join(cDir, path_Inv)
os.makedirs(outFolder, exist_ok=True)

#====== dPL setting ======
# 在计算动态范围之前，检查参数和标准差的统计信息
print("Parameters stats - min:", parameters.min(axis=0), "max:", parameters.max(axis=0))
print("Std values stats - min:", std_values.min(axis=0), "max:", std_values.max(axis=0))
print("Std values summary before fix:")
for i, var_name in enumerate(varStd):
    std_col = std_values[:, i]
    zero_std_count = (std_col == 0).sum()
    neg_std_count = (std_col < 0).sum()
    print(f"{var_name}: zero_std={zero_std_count}, neg_std={neg_std_count}")

# 处理负值和零值标准差
std_values = np.maximum(std_values, 1e-6)

#######################################################################
k_num = 1.5
lb_per_site = parameters - k_num * std_values
ub_per_site = parameters + k_num * std_values
lb_per_site = np.maximum(lb_per_site, 0)

#多目标预测，80个(列)参数的各自13997站点的means和std
means = np.array([28.163618062,# 27.541479095, 24.564728172, 21.902943887, 16.866018413, 15.123596378, 16.483777283, 10.562207479,        #  k_s  l1-l8
                  0.463784877,# 0.470218189, 0.452759249, 0.438032793, 0.427949315, 0.413760248, 0.419960942, 0.416285460,                #  theta_s
                  0.014868566,# 0.015163849, 0.014608541, 0.015141477, 0.014341584, 0.013997189, 0.014955216, 0.024649249,                #  VGM_alpha
                  1.301782073,# 1.301772755, 1.298171419, 1.302800386, 1.303208470, 1.308617403, 1.318899729, 1.214409391,                #  VGM_n
                  0.102488202,# 0.101397796, 0.104363607, 0.106434799, 0.116347700, 0.117846939, 0.112617342, 0.203921497,                #  VGM_theta_r
                  1230280.747892406,# 1215206.906033436, 1254415.778702579, 1286331.801412803, 1312431.922590555, 1345684.192425163, 1332273.657917768, 1349377.497311924,   #  csol
                  1.722315957,# 1.719475811, 1.767275235, 1.831943238, 1.872237049, 1.915399869, 1.900168003, 1.866068105,                #  k_solids
                  0.171747215,# 0.167506819, 0.180671561, 0.195230205,0.207036439, 0.221715076, 0.217790997, 0.221848152,                 #  tkdry
                  2.527300874,# 2.537129753, 2.570000370, 2.631200483, 2.665805699, 2.695156535, 2.687086304, 2.629560325,                #  tksatf
                  1.328971824,# 1.321578506, 1.371686754, 1.434020842, 1.474235596, 1.520394221, 1.503537279, 1.480030838                 #  tksatu
                  ])                                                
 
stds = np.array([11.031442134,# 10.507641195, 9.045515796, 7.875010631, 5.149641253, 4.913943192, 6.004187607, 3.108306948,               #  k_s  l1-l8
                 0.032424086,# 0.034592485, 0.030648375, 0.033425484, 0.034941786, 0.034905419, 0.039828920, 0.040618952,                 #  theta_s
                 0.004491232,# 0.004278898, 0.004371184, 0.004803268, 0.004753460, 0.005019046, 0.005027846, 0.011397554,                 #  VGM_alpha
                 0.034390302,# 0.036880399, 0.037440408, 0.037822610, 0.037872810, 0.056390732, 0.067985644, 0.082400935,                 #  VGM_n
                 0.015151712, #0.013268619, 0.017176003, 0.023559970, 0.028012649, 0.025420583, 0.025408952, 0.097939914,                 #  VGM_theta_r
                 78625.635866540, #80678.701124916, 73047.237570532, 76785.671270351, 78433.600802595, 77981.564029115, 92796.861883904, 87792.779217692,     #  csol
                 0.065276886, #0.064849079, 0.063906143, 0.071936332, 0.085965652, 0.090853290, 0.103658924, 0.124663967,                 #  k_solids
                 0.067588372, #0.068529240, 0.067959246, 0.070570498, 0.073689829, 0.073925810, 0.076381287, 0.077958202,                 #  tkdry
                 0.075621095, #0.072533230, 0.063185934, 0.057810981, 0.063020336, 0.064142469, 0.074939414, 0.096965392,                 #  tksatf
                 0.069975499, #0.071930028, 0.070876845, 0.082152739, 0.096459726, 0.101557135, 0.114800565, 0.132714217                  #  tksatu
                 ])                

lb_per_site_norm = np.zeros_like(lb_per_site)
ub_per_site_norm = np.zeros_like(ub_per_site)

for i in range(len(varConst)):
    lb_per_site_norm[:, i] = (lb_per_site[:, i] - means[i]) / stds[i]
    ub_per_site_norm[:, i] = (ub_per_site[:, i] - means[i]) / stds[i]

print("Normalized range stats after conversion:")
print("lb_per_site_norm - min:", lb_per_site_norm.min(axis=0), "max:", lb_per_site_norm.max(axis=0))
print("ub_per_site_norm - min:", ub_per_site_norm.min(axis=0), "max:", ub_per_site_norm.max(axis=0))

lb_per_site_tensor = torch.tensor(lb_per_site_norm, dtype=torch.float32).cuda()
ub_per_site_tensor = torch.tensor(ub_per_site_norm, dtype=torch.float32).cuda()

lossFun_RMSE = crit.RmseLoss()
lossFun_Range = crit.RangeBoundLoss(factor=1)
lossFun_Inv = crit.sumOfLoss(lossFun_RMSE, lossFun_Range)

# ===================== 训练模型（使用Adam + 预热余弦退火）=====================
# 训练脚本中的调用
model_Inv = rnn.CudnnLstmModel_Inv(nx=nx, ny=ny, hiddenSize=hiddenSizeLst[0], filename=filename)
model_Inv = train.trainModel(
    model_Inv, (forcing, target_GF), modelTarget, rawData, lossFun_Inv, 
    nEpoch=epoch, 
    miniBatch=minibatch,
    optLr=0.5,                # 初始学习率0.5（与原始代码一致）
    optRho=0.95,              # Adadelta参数
    optWd=0.00001,            # 权重衰减
    saveEpoch=saveEpoch, 
    saveFolder=outFolder, 
    lyrMode=0, 
    scaleTrans=False,
    optimizer_type='adadelta',
    scheduler_type='cosine_warmup',
    warmup_epochs=50,        # 100个epoch线性预热
    T_max=950,                # 900个epoch余弦退火
    eta_min=0.1,              # 最终学习率0.1（与原始代码一致）
    lb_per_site=lb_per_site_tensor, 
    ub_per_site=ub_per_site_tensor
)

modelName = 'dPL_gz_model_Adadelta_CosineWarmup'
train.saveModel(outFolder, model_Inv, epoch, modelName=modelName)

############################################################  valid
subset = 'South_China120x220'

tRange_train = [20050101, 20051231]
val_years = [2006]

model_dir = outFolder
surrogate_model_path = filename

print("加载代理模型...")
surrogate_model = torch.load(surrogate_model_path, map_location='cuda:0')
surrogate_model.eval()

print("加载训练数据...")
df_train = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange_train)
forcing_train = torch.from_numpy(df_train.getDataTs(varForcing, doNorm=True, rmNan=True)).float().cuda()
rawData = torch.from_numpy(df_train.getDataConst(varRaw, doNorm=True, rmNan=True)).float().cuda()
target_GF_train = torch.from_numpy(df_train.getDataTs(targetVar, doNorm=True, rmNan=True)).float().cuda()

parameters = df_train.getDataConst(varConst, doNorm=False, rmNan=True)
std_values = np.maximum(df_train.getDataConst(varStd, doNorm=False, rmNan=True), 1e-6)
##########################################################################################
lb_per_site = np.maximum(parameters - k_num * std_values, 0)
lb_per_site_norm = np.array([(lb_per_site[:, i] - means[i]) / stds[i] for i in range(len(varConst))]).T
ub_per_site_norm = np.array([(parameters[:, i] + k_num * std_values[:, i] - means[i]) / stds[i] for i in range(len(varConst))]).T

lb_t = torch.from_numpy(lb_per_site_norm).float().cuda()
ub_t = torch.from_numpy(ub_per_site_norm).float().cuda()

model_files = sorted([(int(re.search(r'model_Ep(\d+)\.pt', f).group(1)), os.path.join(model_dir, f)) 
                      for f in os.listdir(model_dir) if f.startswith('model_Ep') and f.endswith('.pt')], key=lambda x: x[0])

class RangeBoundLoss(torch.nn.Module):
    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor
    def forward(self, x, lb, ub):
        v = torch.relu(x - ub) + torch.relu(lb - x)
        return self.factor * (v.sum() / (v > 0).sum()) if (v > 0).any() else torch.tensor(0.0, device=x.device)

class RmseLoss(torch.nn.Module):
    def forward(self, output, target):
        loss = 0
        for k in range(target.shape[2]):
            p, t = output[:, :, k], target[:, :, k]
            mask = ~torch.isnan(t)
            if mask.any():
                loss += torch.sqrt(((p[mask] - t[mask])**2).mean())
        return loss

def evaluate_model(model, forcing_val, target_val_norm, batch_size=100):
    model.eval()
    surrogate_model.eval()
    ngrid = forcing_train.shape[0]
    total_rmse, total_range, n_batches = 0, 0, 0
    per_var_rmse = {v: [] for v in targetVar}
    
    with torch.no_grad():
        for i in range(0, ngrid, batch_size):
            idx = slice(i, min(i + batch_size, ngrid))
            
            x_train = forcing_train[idx].permute(1, 0, 2)
            x_val = forcing_val[idx].permute(1, 0, 2)
            y_train = target_GF_train[idx].permute(1, 0, 2)
            c_val = rawData[idx].unsqueeze(1).repeat(1, x_val.shape[0], 1).permute(1, 0, 2)
            
            # 参数反演模型生成参数（与训练时一致，scaleTrans=True）
            _, param = model((x_train, x_val, c_val), y_train, testMode=True, lyrMode=0, scaleTrans=True)
            
            # 参数形状: (batch, nparam) 或 (time, batch, nparam)
            if param.dim() == 3:
                param_batch = param[-1, :, :]
            else:
                param_batch = param
            
            # 将参数输入代理模型（代理模型需要 (time, batch, nx) 格式，其中 nx = 7 + 80）
            param_expanded = param_batch.unsqueeze(0).repeat(x_val.shape[0], 1, 1)
            x_surrogate = torch.cat((x_val, param_expanded), dim=2)
            y_pred = surrogate_model(x_surrogate)
            y_pred = y_pred.permute(1, 0, 2)
            y_true = target_val_norm[idx]
            
            nt = min(y_pred.shape[1], y_true.shape[1])
            y_pred, y_true = y_pred[:, :nt, :], y_true[:, :nt, :]
            
            total_rmse += RmseLoss()(y_pred.permute(1,0,2), y_true.permute(1,0,2)).item()
            
            for k, vn in enumerate(targetVar):
                p, t = y_pred[:, :, k], y_true[:, :, k]
                mask = ~torch.isnan(t)
                if mask.any():
                    per_var_rmse[vn].append(torch.sqrt(((p[mask] - t[mask])**2).mean()).item())
            
            # 范围约束损失（与训练时一致）
            param_for_range = param_batch
            range_loss = RangeBoundLoss()(param_for_range, lb_t[idx], ub_t[idx]).item()
            total_range += range_loss
            n_batches += 1
    
    avg_per_var = {v: np.mean(lst) for v, lst in per_var_rmse.items() if lst}
    return total_rmse/n_batches, total_range/n_batches, avg_per_var

color_map = {
    'e': 'green', 'ro': 'skyblue',
    'SM_1': 'darkblue', 'SM_2': 'blue', 'SM_3': 'steelblue', 'SM_4': 'lightblue',
    'ST_1': 'darkred', 'ST_2': 'red', 'ST_3': 'salmon', 'ST_4': 'lightcoral'
}

for year in val_years:
    print(f"\n验证年份: {year}")
    output_dir = os.path.join(model_dir, f'valid_{year}')
    os.makedirs(output_dir, exist_ok=True)
    
    tRange_val = [year*10000 + 101, year*10000 + 1231]
    df_val = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange_val)
    forcing_val = torch.from_numpy(df_val.getDataTs(varForcing, doNorm=True, rmNan=True)).float().cuda()
    target_val_norm = torch.from_numpy(df_val.getDataTs(targetVar, doNorm=True, rmNan=True)).float().cuda()
    
    results = []
    for epoch, model_path in model_files:
        try:
            model = torch.load(model_path, map_location='cuda:0')
            rmse_loss, range_loss, per_var = evaluate_model(model, forcing_val, target_val_norm)
            results.append({'epoch': epoch, 'total_loss': rmse_loss + range_loss, 
                           'rmse_loss': rmse_loss, 'range_loss': range_loss, **per_var})
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Epoch {epoch} 失败: {e}")
    
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(output_dir, 'validation_loss.csv'), index=False)
        
        best_idx = df_res['total_loss'].idxmin()
        best = df_res.loc[best_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = df_res['epoch'].values
        
        axes[0].plot(epochs, df_res['total_loss'], 'b-o', markersize=3, linewidth=1.5, label='Total Loss')
        axes[0].plot(epochs, df_res['rmse_loss'], 'g-o', markersize=3, linewidth=1.5, label='RMSE Loss')
        axes[0].plot(epochs, df_res['range_loss'], 'orange', marker='s', markersize=3, linewidth=1.5, label='Range Loss')
        axes[0].axvline(best['epoch'], color='r', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Loss Curves (Best: {int(best["epoch"])}, {best["total_loss"]:.4f})')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for v in targetVar:
            if v in df_res.columns:
                axes[1].plot(epochs, df_res[v], '-o', markersize=2, linewidth=1, 
                            color=color_map[v], alpha=0.7)
                axes[1].text(epochs[-1], df_res[v].values[-1], v, fontsize=9, 
                            color=color_map[v], ha='left', va='center')
        
        axes[1].axvline(best['epoch'], color='r', linestyle='--', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Per-Variable RMSE')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Validation {year} - gz Model Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

print("\n所有验证完成")

#############集中为一张大图
val_years = [2005, 2006, 2007, 2009, 2010]
experiment_name = 'SM1+ST1'
base_dir = path_Inv

color_map = {
    'e': 'green', 'ro': 'skyblue',
    'SM_1': 'darkblue', 'SM_2': 'blue', 'SM_3': 'steelblue', 'SM_4': 'lightblue',
    'ST_1': 'darkred', 'ST_2': 'red', 'ST_3': 'salmon', 'ST_4': 'lightcoral'
}

fig, axes = plt.subplots(5, 2, figsize=(16, 30))

for i, year in enumerate(val_years):
    valid_dir = os.path.join(base_dir, f'valid_{year}')
    loss_file = os.path.join(valid_dir, 'validation_loss.csv')
    
    if not os.path.exists(loss_file):
        continue
    
    df_res = pd.read_csv(loss_file)
    best_idx = df_res['total_loss'].idxmin()
    best = df_res.loc[best_idx]
    epochs = df_res['epoch'].values
    
    ax1 = axes[i, 0]
    ax2 = axes[i, 1]
    
    ax1.plot(epochs, df_res['total_loss'], 'b-o', markersize=3, linewidth=1.5, label='Total Loss')
    ax1.plot(epochs, df_res['rmse_loss'], 'g-o', markersize=3, linewidth=1.5, label='RMSE Loss')
    ax1.plot(epochs, df_res['range_loss'], 'orange', marker='s', markersize=3, linewidth=1.5, label='Range Loss')
    ax1.axvline(best['epoch'], color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves (Best: {int(best["epoch"])}, {best["total_loss"]:.4f})')
    # ax1.set_ylim(3.3, 4.0)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for v in targetVar:
        if v in df_res.columns:
            line, = ax2.plot(epochs, df_res[v], '-o', markersize=2, linewidth=1, 
                             color=color_map[v], alpha=0.7)
            ax2.text(epochs[-1], df_res[v].values[-1], v, fontsize=9, 
                    color=color_map[v], ha='left', va='center')
    
    ax2.axvline(best['epoch'], color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Per-Variable RMSE')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

plt.suptitle(f'{experiment_name}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'valid_gz_2006.png'), dpi=150, bbox_inches='tight')
plt.close()

print("大图绘制完成")