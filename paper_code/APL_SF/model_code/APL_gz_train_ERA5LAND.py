#########余弦退火
import sys
sys.path.append(r'/share/home/dq152/APL_SF/model_code') 
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re

traingpuid = 0
torch.cuda.set_device(traingpuid)

cDir = r'/share/home/dq152/APL_SF/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND')

subsetLst=['South_China120x220']

tRangeLst=[[20050101, 20051231]]  ## training period

hiddenSizeLst = [512] #256,512

## dPL target 
targetVar=['SM_1']
# training year
trainYear=int(str(tRangeLst[0][1])[0:4])-int(str(tRangeLst[0][0])[0:4])

df = dbCsv.DataframeCsv(rootDB=rootDB, subset=subsetLst[0], tRange=tRangeLst[0])

# ===== varForcing, dynamic inputs =====
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
# ===== parameters  =====
varConst = ['k_s_l1',
            'theta_s_l1',
            'VGM_alpha_l1',
            'VGM_n_l1',
            'VGM_theta_r_l1'] 
# ===== static attributes =====
varRaw = ['SAND_depth_01',
          'CLAY_depth_01',
          'GRAV_depth_01',
          'BD_depth_01',
          'OC_depth_01',
          'landtype','forest_height','topography']#######新增3个静态参数

## read database
forcing = df.getDataTs(varForcing, doNorm=True, rmNan=True) 
parameters = df.getDataConst(varConst, doNorm=True, rmNan=True) 
rawData = df.getDataConst(varRaw, doNorm=True, rmNan=True)          
target = df.getDataTs([targetVar[0]], doNorm=True, rmNan=False)  
target_GF = df.getDataTs([targetVar[0]], doNorm=True, rmNan=True)  

modelTarget = target

## ====================================================== read Surrogate model ======================================
# filename ='/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/surrogate_model_2/model_Ep300.pt'
# filename ='/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/surrogate_model_3/model_Ep260.pt'

filename ='/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/surrogate_model_2/model_Ep950.pt'

model_loaded_PF_SMAP = torch.load(filename) 
model_loaded_PF_SMAP.eval()

#========================== gz
nx = (forcing.shape[-1] + rawData.shape[-1], rawData.shape[-1], len(varConst))
ny = target.shape[-1]

# ======= dPL dataframe setting =======
epoch=1000
saveEpoch=10
minibatch=[300, 180]

# path_Inv = 'def_ERA5LAND/SM_layer123/gz_model_2'
# path_Inv = 'def_ERA5LAND/SM_layer123/gz_model_3'

path_Inv = 'def_ERA5LAND/SM_layer123/gz_model_2'

outFolder = os.path.join(cDir, path_Inv)
os.makedirs(outFolder, exist_ok=True)

#============================================================ dPL setting ============================================

#================== loss function
lossFun_Inv1 = crit.RmseLoss()

## have to normalize the parameters' upper bound (ub) and lower bound (lb) here
par_max = np.array([8000,     0.8,     0.1,      10,     0.2    ])   
par_min = np.array([8,        0.2,     0.001,    1.05,   0.01   ])

means = np.array([28.163618062, 0.463784877, 0.014868566, 1.301782073, 0.102488202]) 
stds  = np.array([11.031442134, 0.032424086, 0.004491232, 0.034390302, 0.015151712]) 


normalized_max = (par_max - means) / stds
normalized_min = (par_min - means) / stds

ub = normalized_max
lb = normalized_min

lossFun_Inv2 = crit.RangeBoundLoss(lb=lb, ub=ub, factor=1)
lossFun_Inv = crit.sumOfLoss(lossFun_Inv1, lossFun_Inv2)  

print(f"全局约束范围已设置:")
print(f"  lb (归一化): min={lb.min():.4f}, max={lb.max():.4f}")
print(f"  ub (归一化): min={ub.min():.4f}, max={ub.max():.4f}")

# ===== 训练模型（Adadelta + 预热余弦退火，LR: 0.5→0.1）=====
model_Inv = rnn.CudnnLstmModel_Inv(nx=nx, ny=ny, hiddenSize=hiddenSizeLst[0], filename=filename)
model_Inv = train.trainModel(
    model_Inv, (forcing, target_GF), modelTarget, rawData, lossFun_Inv, 
    nEpoch=epoch, 
    miniBatch=minibatch,
    optLr=0.5,                # 初始学习率0.5
    optRho=0.95,              # Adadelta参数
    optWd=0.00001,            # 权重衰减
    saveEpoch=saveEpoch, 
    saveFolder=outFolder, 
    lyrMode=0, 
    scaleTrans=False,
    # ===== 优化器和调度器配置 =====
    optimizer_type='adadelta',
    scheduler_type='cosine_warmup',
    warmup_epochs=50,        # 100个epoch线性预热
    T_max=950,                # 900个epoch余弦退火
    eta_min=0.1,              # 最终学习率0.1
    csv_name='run_cosine_warmup'
)

## save trained model
modelName = 'dPL_gz_model_final'
train.saveModel(outFolder, model_Inv, epoch, modelName=modelName)
print(f"训练完成！模型已保存为: {modelName}")

##################################################################################### validation ##############################################################

# GPU设置
valgpuid = 0
torch.cuda.set_device(valgpuid)

# ==================== 路径和配置 ====================
cDir = r'/share/home/dq152/APL_SF/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND')
subset = 'South_China120x220'
tRange_train = [20050101, 20051231]
val_years = [2006]
model_dir = outFolder
surrogate_model_path = filename

class RangeBoundLoss(torch.nn.Module):
    def __init__(self, lb, ub, factor=1):
        super().__init__()
        self.register_buffer('lb', lb)
        self.register_buffer('ub', ub)
        self.factor = factor
    def forward(self, x):
        upper_violation = torch.relu(x - self.ub)
        lower_violation = torch.relu(self.lb - x)
        total_violation = upper_violation + lower_violation
        violation_mask = total_violation > 0
        if violation_mask.sum() > 0:
            return self.factor * total_violation.sum() / violation_mask.sum()
        return torch.tensor(0.0, device=x.device)

class RmseLoss(torch.nn.Module):
    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            if len(p) > 0:
                loss += torch.sqrt(((p - t)**2).mean())
        return loss

def evaluate_model(model, forcing_train, target_GF_train, forcing_val, target_val_norm, 
                   rawData, criterion_rmse, criterion_range, batch_size=300):
    model.eval()
    ngrid = forcing_train.shape[0]
    total_rmse, total_range, n_batches = 0, 0, 0
    
    with torch.no_grad():
        for i in range(0, ngrid, batch_size):
            end = min(i + batch_size, ngrid)
            sl = slice(i, end)
            
            x_train = forcing_train[sl].permute(1, 0, 2)
            y_train = target_GF_train[sl].permute(1, 0, 2)
            x_val = forcing_val[sl].permute(1, 0, 2)
            raw = rawData[sl].unsqueeze(1).repeat(1, forcing_train.shape[1], 1).permute(1, 0, 2)
            
            y_pred, param = model((x_train, x_val, raw), y_train, testMode=True, lyrMode=0, scaleTrans=True)
            
            y_pred = y_pred.permute(1, 0, 2)
            nt = min(y_pred.shape[1], target_val_norm[sl].shape[1])
            rmse_loss = criterion_rmse(y_pred[:, :nt, :].permute(1, 0, 2), 
                                        target_val_norm[sl][:, :nt, :].permute(1, 0, 2))
            
            param_flat = param[-1, :, :] if param.dim() == 3 else param
            range_loss = criterion_range(param_flat)
            
            total_rmse += rmse_loss.item()
            total_range += range_loss.item()
            n_batches += 1
    
    return {'rmse_loss': total_rmse / n_batches, 'range_loss': total_range / n_batches}

# ==================== 主程序 ====================
print("加载数据...")
df_train = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange_train)
forcing_train = torch.from_numpy(df_train.getDataTs(varForcing, doNorm=True, rmNan=True)).float().cuda()
target_GF_train = torch.from_numpy(df_train.getDataTs(targetVar, doNorm=True, rmNan=True)).float().cuda()
rawData = torch.from_numpy(df_train.getDataConst(varRaw, doNorm=True, rmNan=True)).float().cuda()

# 加载代理模型
surrogate = torch.load(surrogate_model_path, map_location='cuda:0')
surrogate.eval()

# 查找所有模型 - 确保按epoch排序
model_files = []
for f in os.listdir(model_dir):
    match = re.search(r'model_Ep(\d+)\.pt', f)
    if match:
        epoch = int(match.group(1))
        if epoch <= 1200:
            model_files.append((epoch, os.path.join(model_dir, f)))
model_files.sort(key=lambda x: x[0])  # 按epoch排序

criterion_rmse = RmseLoss().cuda()
criterion_range = RangeBoundLoss(torch.from_numpy(normalized_min).float().cuda(),
                                  torch.from_numpy(normalized_max).float().cuda()).cuda()

# 存储每年结果
all_year_results = {}

for year in val_years:
    print(f"处理 {year}...")
    tRange_val = [year*10000+101, year*10000+1231]
    df_val = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange_val)
    forcing_val = torch.from_numpy(df_val.getDataTs(varForcing, doNorm=True, rmNan=True)).float().cuda()
    target_val_norm = torch.from_numpy(df_val.getDataTs(targetVar, doNorm=True, rmNan=True)).float().cuda()
    
    results = []
    for epoch, model_path in model_files:
        model = torch.load(model_path, map_location='cuda:0')
        model.eval()
        loss_dict = evaluate_model(model, forcing_train, target_GF_train, forcing_val, 
                                    target_val_norm, rawData, criterion_rmse, criterion_range)
        results.append({'epoch': epoch, 'total_loss': loss_dict['rmse_loss'] + loss_dict['range_loss'],
                        'rmse_loss': loss_dict['rmse_loss'], 'range_loss': loss_dict['range_loss']})
        del model
        torch.cuda.empty_cache()
    
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('epoch')  # 确保数据按epoch排序
    df_res.to_csv(os.path.join(model_dir, f'valid_{year}_loss.csv'), index=False)
    all_year_results[year] = df_res

# ==================== 绘制2x3子图 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, year in enumerate(val_years):
    ax = axes[idx]
    df_res = all_year_results[year]
    epochs = df_res['epoch'].values
    total_loss = df_res['total_loss'].values
    rmse_loss = df_res['rmse_loss'].values
    range_loss = df_res['range_loss'].values
    
    # 绘制曲线 - 使用'o-'确保数据点正确连接
    ax.plot(epochs, total_loss, 'b-o', linewidth=1.5, markersize=3, label='Total Loss')
    ax.plot(epochs, rmse_loss, 'g-o', linewidth=1.5, markersize=3, label='RMSE Loss')
    ax.plot(epochs, range_loss, 'orange', marker='o', markersize=3, linewidth=1.5, label='Range Loss')
    
    best_idx = df_res['total_loss'].idxmin()
    best_epoch = df_res.loc[best_idx, 'epoch']
    best_loss = df_res.loc[best_idx, 'total_loss']
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.scatter([best_epoch], [best_loss], color='red', s=80, zorder=5, marker='o')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{year} (Best: Epoch {best_epoch}, Loss={best_loss:.4f})', fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs.min(), epochs.max())

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'valid_loss_2006.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n完成! 结果保存在: {model_dir}")

