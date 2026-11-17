import sys
sys.path.append(r"/share/home/dq152/APL_SI/model_code_ERA5LAND")

from hydroDL import master, utils
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import linregress
import traceback

# GPU设置
torch.cuda.set_device(0)

# 路径配置
cDir = r'/share/home/dq152/APL_SI/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND')
saveResultsPath = '/share/home/dq152/APL_SI/script_data/def_ERA5LAND/SM_layer123/dpl_opt_results/'
os.makedirs(saveResultsPath, exist_ok=True)

# 数据配置
subset = ['South_China120x220']
tRange = [20050101, 20051231]
tRangeTest = [[20060101, 20061231]]

# 加载模型
model_PF_SMAP = torch.load('/share/home/dq152/APL_SI/script_data/def_ERA5LAND/SM_layer123/surrogate_model/model_Ep980.pt', map_location='cuda:0').eval()
model_gz = torch.load('/share/home/dq152/APL_SI/script_data/def_ERA5LAND/SM_layer123/gz_model/model_Ep460.pt', map_location='cuda:0').eval()

# 变量定义
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
varConst = ['k_s_l1',#'k_s_l2','k_s_l3','k_s_l4','k_s_l5','k_s_l6','k_s_l7','k_s_l8',
            'theta_s_l1',#'theta_s_l2','theta_s_l3','theta_s_l4','theta_s_l5','theta_s_l6','theta_s_l7','theta_s_l8',
            'VGM_alpha_l1',#'VGM_alpha_l2','VGM_alpha_l3','VGM_alpha_l4','VGM_alpha_l5','VGM_alpha_l6','VGM_alpha_l7','VGM_alpha_l8',           
            'VGM_n_l1',#'VGM_n_l2','VGM_n_l3','VGM_n_l4','VGM_n_l5','VGM_n_l6','VGM_n_l7','VGM_n_l8',
            'VGM_theta_r_l1',#'VGM_theta_r_l2','VGM_theta_r_l3','VGM_theta_r_l4','VGM_theta_r_l5','VGM_theta_r_l6','VGM_theta_r_l7','VGM_theta_r_l8',
    # 'csol_l1','csol_l2','csol_l3','csol_l4','csol_l5','csol_l6','csol_l7','csol_l8',
    # 'k_solids_l1','k_solids_l2','k_solids_l3','k_solids_l4','k_solids_l5','k_solids_l6','k_solids_l7','k_solids_l8',
    # 'tkdry_l1','tkdry_l2','tkdry_l3','tkdry_l4','tkdry_l5','tkdry_l6','tkdry_l7','tkdry_l8',
    # 'tksatf_l1','tksatf_l2','tksatf_l3','tksatf_l4','tksatf_l5','tksatf_l6','tksatf_l7','tksatf_l8',
    # 'tksatu_l1','tksatu_l2','tksatu_l3','tksatu_l4','tksatu_l5','tksatu_l6','tksatu_l7','tksatu_l8']
]
targetVar = [#'e', 'ro',
             'SM_1', #'SM_2', 'SM_3', 'SM_4', 
             #'ST_1', 'ST_2', 'ST_3', 'ST_4']
]
target_vars_display = [#'ET', 'runoff', 
                       'SM_layer123',# 'SM_layer45', 'SM_layer67', 'SM_layer8910', 
                       #'ST_layer123', 'ST_layer45', 'ST_layer67', 'ST_layer8910']
]
varRaw = ['SAND_depth_01',#'SAND_depth_02', 'SAND_depth_03', 'SAND_depth_04', 'SAND_depth_05', 'SAND_depth_06', 'SAND_depth_07', 'SAND_depth_08', 
          'CLAY_depth_01',#'CLAY_depth_02','CLAY_depth_03','CLAY_depth_04','CLAY_depth_05','CLAY_depth_06','CLAY_depth_07','CLAY_depth_08',
          'GRAV_depth_01',#'GRAV_depth_02','GRAV_depth_03','GRAV_depth_04','GRAV_depth_05','GRAV_depth_06','GRAV_depth_07','GRAV_depth_08',
          'BD_depth_01',#'BD_depth_02','BD_depth_03','BD_depth_04','BD_depth_05','BD_depth_06','BD_depth_07','BD_depth_08',
          'OC_depth_01',#'OC_depth_02','OC_depth_03','OC_depth_04','OC_depth_05','OC_depth_06','OC_depth_07','OC_depth_08',
     ###############
        'forest_height','landtype','topography']

def testModel(model, x, c, batchSize=100, filePathLst=None, lyrMode=0, scaleTrans=False, model_type='gz'):
    xTrain_temp, xTest_temp = x[0]
    z = x[1]
    ngrid, ntTrain, nx_input = xTrain_temp.shape
    ny = model.ny
    
    nc_est = c[0].shape[-1] if isinstance(c, tuple) else c.shape[-1] if c is not None else 0
    nc_fixed = c[1].shape[-1] if isinstance(c, tuple) else 0
    fixed_params = isinstance(c, tuple)
    
    fLst = [open(fp, 'w') for fp in filePathLst]
    all_parameters = []
    
    for i in range(0, ngrid, batchSize):
        i_end = min(i+batchSize, ngrid)
        x_train_batch = torch.from_numpy(np.swapaxes(xTrain_temp[i:i_end], 1, 0)).float().cuda()
        x_test_batch = torch.from_numpy(np.swapaxes(xTest_temp[i:i_end], 1, 0)).float().cuda()
        
        if fixed_params:
            c_est = torch.from_numpy(np.swapaxes(np.repeat(np.reshape(c[0][i:i_end], [i_end-i, 1, nc_est]), ntTrain, axis=1), 1, 0)).float().cuda()
            c_fixed = torch.from_numpy(np.swapaxes(np.repeat(np.reshape(c[1][i:i_end], [i_end-i, 1, nc_fixed]), ntTrain, axis=1), 1, 0)).float().cuda()
            x_tuple = (x_train_batch, x_test_batch, c_est, c_fixed)
        elif c is not None:
            c_batch = torch.from_numpy(np.swapaxes(np.repeat(np.reshape(c[i:i_end], [i_end-i, 1, c.shape[-1]]), ntTrain, axis=1), 1, 0)).float().cuda()
            x_tuple = (x_train_batch, x_test_batch, c_batch)
        else:
            x_tuple = (x_train_batch, x_test_batch)
        
        z_batch = torch.from_numpy(np.swapaxes(z[i:i_end], 1, 0)).float().cuda()
        yP, Param_R2P = model(x_tuple, z_batch, testMode=True, outModel=None, lyrMode=lyrMode, 
                              scaleTrans=scaleTrans, routingModel=False, lenF=0, fixed_params=fixed_params)
        
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        for k in range(ny):
            pd.DataFrame(yOut[:, :, k]).to_csv(fLst[k], header=False, index=False)
        
        params = Param_R2P.detach().cpu().numpy()
        if len(params.shape) == 3:
            all_parameters.append(params[-1 if params.shape[0] > 1 else 0])
        else:
            all_parameters.append(params if params.shape[0] == i_end-i else params[-1:])
        
        torch.cuda.empty_cache()
    
    if all_parameters:
        try:
            param_df = pd.DataFrame(np.concatenate(all_parameters, axis=0))
            if len(param_df.shape) == 3:
                param_df = pd.DataFrame(param_df.values[:, 0, :])
            param_df.to_csv(fLst[-1], header=False, index=False)
        except Exception as e:
            print(f"参数保存失败: {e}")
    
    [f.close() for f in fLst]

def calculate_statistics(pred, obs):
    n_sites, n_time = pred.shape
    results = {k: np.full(n_sites, np.nan) for k in ['bias','rmse','corr','r2','kge','nse','valid_points']}
    
    for i in range(n_sites):
        mask = ~np.isnan(pred[i]) & ~np.isnan(obs[i])
        p, o = pred[i][mask], obs[i][mask]
        if len(p) < 2:
            continue
            
        results['valid_points'][i] = len(p)
        results['bias'][i] = np.mean(p - o)
        results['rmse'][i] = np.sqrt(np.mean((p - o) ** 2))
        
        if np.std(p) > 0 and np.std(o) > 0:
            results['corr'][i] = np.corrcoef(p, o)[0, 1]
            try:
                slope, intercept, r_value, p_value, std_err = linregress(o, p)
                results['r2'][i] = r_value ** 2
            except:
                pass
            
            alpha = np.std(p) / np.std(o)
            beta = np.mean(p) / np.mean(o) if np.mean(o) != 0 else np.nan
            results['kge'][i] = 1 - np.sqrt((results['corr'][i]-1)**2 + (alpha-1)**2 + (beta-1)**2) if not np.isnan(alpha) and not np.isnan(beta) else np.nan
        
        denom = np.sum((o - np.mean(o))**2)
        results['nse'][i] = 1 - np.sum((p-o)**2)/denom if denom > 0 else np.nan
    
    results['total_points'] = np.full(n_sites, n_time)
    return results

# 主处理循环
for test_range in tRangeTest:
    df_tr = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset[0], tRange=tRange)
    df_test = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset[0], tRange=test_range)
    
    forcing = df_tr.getDataTs(varForcing, doNorm=True, rmNan=True)
    rawData = df_tr.getDataConst(varRaw, doNorm=True, rmNan=True)
    target_GF = df_tr.getDataTs(targetVar, doNorm=True, rmNan=True)
    forcing_test = df_test.getDataTs(varForcing, doNorm=True, rmNan=True)
    target_test = df_test.getDataTs(targetVar, doNorm=False, rmNan=False)
    
    ngrid = df_test.getGeo()[0].shape[0]
    crd = df_test.getGeo()
    crd_lat, crd_lon = np.expand_dims(crd[0], axis=1), np.expand_dims(crd[1], axis=1)
    
    outPath_gz = os.path.join(saveResultsPath, 'gz/')
    outPath_params = os.path.join(saveResultsPath, 'parameters/')
    os.makedirs(outPath_gz, exist_ok=True)
    os.makedirs(outPath_params, exist_ok=True)
    
    # 运行gz模型
    ny = len(targetVar)
    gz_files = [os.path.join(outPath_gz, f'yOut_dPL_gz_var{i}') for i in range(ny)] + [os.path.join(outPath_gz, 'param_dPL_gz')]
    
    try:
        testModel(model_gz, ((forcing, forcing_test), target_GF), rawData, 100, gz_files, 0, False, 'gz')
        
        # 读取预测结果
        predictions = []
        for i in range(ny):
            try:
                pred_df = pd.read_csv(gz_files[i], header=None)
                predictions.append(pred_df.values)
            except:
                predictions.append(np.full((ngrid, forcing_test.shape[1]), np.nan))
        yOut_gz = np.stack(predictions, axis=2) if predictions else None
        
        # 保存CoLM参数
        try:
            colm_params = df_tr.getDataConst(varConst, doNorm=False, rmNan=True)
            if colm_params.shape[0] != ngrid:
                colm_params = colm_params[:ngrid] if colm_params.shape[0] > ngrid else np.pad(colm_params, ((0, ngrid-colm_params.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
            pd.DataFrame(np.concatenate([crd_lat, crd_lon, colm_params], axis=1)).to_csv(
                os.path.join(outPath_params, f'{subset[0]}_CoLM_parameters.csv'),
                header=['lat','lon']+varConst, index=False)
        except:
            pass
        
        # 处理gz模型结果
        if yOut_gz is not None:
            # 读取参数
            try:
                gz_params = pd.read_csv(gz_files[-1], header=None).values
                if gz_params.shape[1] != len(varConst):
                    gz_params = gz_params[:, :len(varConst)] if gz_params.shape[1] > len(varConst) else np.pad(gz_params, ((0,0),(0,len(varConst)-gz_params.shape[1])), 'constant')
                
                # 反归一化参数
                params_denorm = np.array([dbCsv.transNorm(gz_params[:,k], rootDB=rootDB, fieldName=varConst[k], fromRaw=False, isConst=True) for k in range(len(varConst))]).T
                pd.DataFrame(np.concatenate([crd_lat, crd_lon, params_denorm], axis=1)).to_csv(
                    os.path.join(outPath_params, f'{subset[0]}_dPL_gz_parameters_calibrated.csv'),
                    header=['lat','lon']+varConst, index=False)
            except:
                pass
            
            # 反归一化预测并计算统计指标
            yOut_denorm = np.zeros_like(yOut_gz)
            for i in range(ny):
                try:
                    yOut_denorm[:,:,i] = dbCsv.transNorm(yOut_gz[:,:,i], rootDB=rootDB, fieldName=targetVar[i], fromRaw=False)
                except:
                    yOut_denorm[:,:,i] = yOut_gz[:,:,i]
                
                stats = calculate_statistics(yOut_denorm[:,:,i], target_test[:,:,i])
                pd.DataFrame({
                    'site_id': range(ngrid), 'lat': crd_lat.flatten(), 'lon': crd_lon.flatten(),
                    'bias': stats['bias'], 'rmse': stats['rmse'], 'corr': stats['corr'],
                    'r2': stats['r2'], 'kge': stats['kge'], 'nse': stats['nse'],
                    'valid_points': stats['valid_points'].astype(int),
                    'total_points': stats['total_points'].astype(int)
                }).to_csv(os.path.join(outPath_gz, f'statErr_gz_{target_vars_display[i]}_sites.csv'), index=False)
                
                result_df = pd.DataFrame(yOut_denorm[:,:,i]).fillna(method='ffill').fillna(method='bfill')
                pd.DataFrame(yOut_denorm[:,:,i]).to_csv(
                    os.path.join(outPath_gz, f'dPL_gz_{target_vars_display[i]}_prediction.csv'),
                    header=False, index=False)
    except Exception as e:
        print(f"gz模型处理失败: {e}")
        traceback.print_exc()

print("多模型参数校准和评估完成！")