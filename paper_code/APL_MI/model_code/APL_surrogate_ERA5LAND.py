import sys  
sys.path.append(r'/share/home/dq152/APL_MI/model_code')
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# GPU setting 
gpuid = 0
torch.cuda.set_device(gpuid)

cDir = r'/share/home/dq152/APL_MI/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND_sur2') 
# rootDB = os.path.join(cDir, 'data_input_ERA5LAND_sur3') 

subsetLst = 'South_China120x220'

tRangeLst = [20050101, 20051231] 

hiddenSizeLst = 512  

df = dbCsv.DataframeCsv(rootDB=rootDB, subset=subsetLst, tRange=tRangeLst)

# ===== varForcing =====  
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
# ===== parameters ===== 
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

forcing = df.getDataTs(varForcing, doNorm=True, rmNan=True) 
parameters = df.getDataConst(varConst, doNorm=True, rmNan=True) 

target = df.getDataTs([#'ET',
                       #'runoff',
                       'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
                       'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910'
                         ],doNorm=True, rmNan=True)

print("=== 形状 ===")
print(f"forcing: {forcing.shape}")
print(f"parameters: {parameters.shape}")
print(f"target: {target.shape}")

nx = forcing.shape[-1] + parameters.shape[-1]
ny = target.shape[-1]    

path_surrogate = '/share/home/dq152/APL_MI/script_data/def_ERA5LAND/SM1+ST1/surrogate_model_2'
# path_surrogate = '/share/home/dq152/APL_MI/script_data/def_ERA5LAND/SM1+ST1/surrogate_model_3'
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
    optLr=1,                # 初始学习率0.5
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

modelName = 'SM1+ST1'
train.saveModel(outFolder, model_PF, epoch, modelName=modelName)

################################################################################### validation ################################################################

model_dir = path_surrogate

# 数据配置
subsetLst = 'South_China120x220'
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
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
##############################################################
years = list(range(2005, 2006))
colors = ['blue', 'green', 'brown', 'orange', 'yellow', 'purple']
epoch_list = list(range(10, 1010, 10))

save_path = path_surrogate
os.makedirs(save_path, exist_ok=True)

# 准备数据函数
def prepare_data(year):
    t_start = year * 10000 + 101
    t_end = year * 10000 + 1231
    df = dbCsv.DataframeCsv(rootDB=rootDB, subset=subsetLst, tRange=[t_start, t_end])
    forcing = df.getDataTs(varForcing, doNorm=True, rmNan=True)
    parameters = df.getDataConst(varConst, doNorm=True, rmNan=True)
    target = df.getDataTs([#'ET', 'runoff',
                           'SM_layer123',#'SM_layer45','SM_layer67','SM_layer8910',
                           'ST_layer123',#'ST_layer45','ST_layer67','ST_layer8910'
                                 ], doNorm=True, rmNan=True)
    
    c_expanded = np.repeat(parameters[:, np.newaxis, :], forcing.shape[1], axis=1)
    x_combined = np.concatenate([forcing, c_expanded], axis=2)
    x_tensor = torch.from_numpy(np.swapaxes(x_combined, 1, 0)).float()
    y_tensor = torch.from_numpy(np.swapaxes(target, 1, 0)).float()
    
    if torch.cuda.is_available():
        x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
    
    return x_tensor, y_tensor, x_combined.shape[0]

# 评估函数
def evaluate_year(year):
    x_tensor, y_tensor, ngrid = prepare_data(year)
    losses = []
    lossFun = crit.RmseLoss()
    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
    
    for epoch in tqdm(epoch_list, desc=str(year)):
        model_file = os.path.join(model_dir, f'model_Ep{epoch}.pt')
        if not os.path.exists(model_file):
            losses.append(np.nan)
            continue
        
        model = torch.load(model_file)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        batch_size = 300
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, ngrid, batch_size):
                x_batch = x_tensor[:, i:i+batch_size, :]
                y_batch = y_tensor[:, i:i+batch_size, :]
                y_pred = model(x_batch)
                total_loss += lossFun(y_pred, y_batch).item() * (min(batch_size, ngrid - i))
        
        losses.append(total_loss / ngrid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    losses = np.array(losses)
    valid_mask = ~np.isnan(losses)
    best_idx = np.argmin(losses[valid_mask])
    best_epoch = epoch_list[best_idx]
    best_loss = losses[valid_mask][best_idx]
    
    # 保存年份损失CSV
    np.savetxt(os.path.join(save_path, f'loss_{year}.csv'), 
               np.column_stack((epoch_list, losses)), 
               delimiter=',', header='Epoch,Loss', comments='')
    
    return np.array(epoch_list), losses, best_epoch, best_loss

# 绘图
plt.figure(figsize=(14, 8))
best_info = []

for year, color in zip(years, colors):
    epochs_arr, losses, best_epoch, best_loss = evaluate_year(year)
    valid_mask = ~np.isnan(losses)
    plt.plot(epochs_arr[valid_mask], losses[valid_mask], color=color, linewidth=1.5, label=f'{year}')
    plt.scatter(best_epoch, best_loss, color=color, s=80, zorder=5, marker='o')
    best_info.append(f"{year}: Epoch {best_epoch}, Loss={best_loss:.4f}")
    print(f"{year}: Best Epoch={best_epoch}, Loss={best_loss:.6f}")

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Valid Loss (RMSE)', fontsize=12)
plt.title('Valid Loss SM1+ST1 (2006)', fontsize=14)
plt.ylim(0, 3)
plt.grid(True, alpha=0.3)

# 图例放在左下方
plt.legend(loc='lower right', fontsize=9)

# 在左下方添加最优模型信息文本框
text_str = '\n'.join(best_info)
plt.text(0.02, 0.02, text_str, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(os.path.join(save_path, 'valid_2006.png'), dpi=150, bbox_inches='tight')

print(f"\n结果保存至: {save_path}")
print(f"已保存文件: loss_2006.csv, valid_2006.png")

