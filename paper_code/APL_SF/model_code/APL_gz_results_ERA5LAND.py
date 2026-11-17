import sys
sys.path.append(r"/share/home/dq152/APL_SF/model_code")

from hydroDL.data import dbCsv
from hydroDL.model import rnn
import os
import numpy as np
import torch
import pandas as pd
from scipy.stats import linregress

# GPU设置
torch.cuda.set_device(0)

# 路径配置
cDir = r'/share/home/dq152/APL_SF/script_data/'
rootDB = os.path.join(cDir, 'data_input_ERA5LAND')

# saveResultsPath = '/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/dpl_opt_results/'
saveResultsPath = '/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/dpl_opt_results_2/'
# saveResultsPath = '/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/dpl_opt_results_3/'

os.makedirs(saveResultsPath, exist_ok=True)

# 数据配置
subset = ['South_China120x220']
tRange = [20050101, 20051231]
tRangeTest = [[year * 10000 + 101, year * 10000 + 1231] for year in range(2005, 2011)]

# 变量定义
varForcing = ['ssrd', 'strd', 'Q', 't2m', 'tp', 'sp', 'u10', 'v10']
varConst = ['k_s_l1', 'theta_s_l1', 'VGM_alpha_l1', 'VGM_n_l1', 'VGM_theta_r_l1']
targetVar = ['SM_1']
target_vars_display = ['SM_layer123']
varRaw = ['SAND_depth_01', 'CLAY_depth_01', 'GRAV_depth_01', 'BD_depth_01', 'OC_depth_01',
          'landtype', 'forest_height', 'topography']  ####### 新增3个静态参数

gz_model = torch.load('/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/gz_model_2/model_Ep100.pt', map_location='cuda:0')
# gz_model = torch.load('/share/home/dq152/APL_SF/script_data/def_ERA5LAND/SM_layer123/gz_model_3/model_Ep100.pt', map_location='cuda:0')

gz_model.eval()

def custom_testModel(model, x, c, outPath_gz, batchSize=100, lyrMode=0, scaleTrans=False, model_type='gz'):
    """测试参数反演模型"""
    (forcing_train, forcing_test), target_GF = x
    ngrid, ntTrain, _ = forcing_train.shape
    ny = model.ny
    
    os.makedirs(outPath_gz, exist_ok=True)
    
    filePathLst = [os.path.join(outPath_gz, f'yOut_dPL_{model_type}_var{i}') for i in range(ny)]
    filePathLst.append(os.path.join(outPath_gz, f'param_dPL_{model_type}'))
    
    fLst = [open(f, 'w') for f in filePathLst]
    all_params = []
    
    for i in range(0, ngrid, batchSize):
        end = min(i + batchSize, ngrid)
        
        xTrain = torch.from_numpy(np.swapaxes(forcing_train[i:end], 1, 0)).float().cuda()
        xTest = torch.from_numpy(np.swapaxes(forcing_test[i:end], 1, 0)).float().cuda()
        
        if c is not None:
            cTemp = np.repeat(c[i:end][:, None, :], ntTrain, axis=1)
            cTemp = torch.from_numpy(np.swapaxes(cTemp, 1, 0)).float().cuda()
            xTuple = (xTrain, xTest, cTemp)
        else:
            xTuple = (xTrain, xTest)
        
        zTest = torch.from_numpy(np.swapaxes(target_GF[i:end], 1, 0)).float().cuda()
        
        yP, param = model(xTuple, zTest, testMode=True, lyrMode=lyrMode, 
                         scaleTrans=scaleTrans, fixed_params=False)
        
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        paramOut = param.detach().cpu().numpy()
        
        if paramOut.ndim == 3:
            paramOut = paramOut[-1] if paramOut.shape[0] > 1 else paramOut[0]
        
        for k in range(ny):
            pd.DataFrame(yOut[:, :, k]).to_csv(fLst[k], header=False, index=False)
        
        all_params.append(paramOut)
        torch.cuda.empty_cache()
    
    if all_params:
        param_df = pd.DataFrame(np.concatenate(all_params, axis=0))
        param_df.to_csv(fLst[-1], header=False, index=False)
    
    for f in fLst:
        f.close()

def calculate_site_statistics(pred, obs):
    """计算站点统计指标"""
    n_sites = pred.shape[0]
    results = {k: np.full(n_sites, np.nan) for k in ['bias', 'rmse', 'corr', 'r2', 'kge', 'nse', 'valid_points']}
    
    for i in range(n_sites):
        mask = ~np.isnan(pred[i]) & ~np.isnan(obs[i])
        p, o = pred[i][mask], obs[i][mask]
        results['valid_points'][i] = len(p)
        
        if len(p) >= 2:
            results['bias'][i] = np.mean(p - o)
            results['rmse'][i] = np.sqrt(np.mean((p - o)**2))
            
            if np.std(p) > 0 and np.std(o) > 0:
                results['corr'][i] = np.corrcoef(p, o)[0, 1]
                results['r2'][i] = results['corr'][i] ** 2
            
            obs_mean = np.mean(o)
            ss_res = np.sum((p - o)**2)
            ss_tot = np.sum((o - obs_mean)**2)
            results['nse'][i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
            if not np.isnan(results['corr'][i]):
                r, alpha, beta = results['corr'][i], np.std(p) / np.std(o), np.mean(p) / np.mean(o)
                results['kge'][i] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return results

def average_arrays_with_nan_padding(arrays):
    """对不同长度数组进行NaN补齐后求平均"""
    max_shape = tuple(max(arr.shape[dim] for arr in arrays) for dim in range(arrays[0].ndim))
    padded_arrays = []
    
    for arr in arrays:
        padded = np.full(max_shape, np.nan)
        slices = tuple(slice(0, size) for size in arr.shape)
        padded[slices] = arr
        padded_arrays.append(padded)
    
    return np.nanmean(np.stack(padded_arrays, axis=0), axis=0)

def save_period_average(period_name, years, annual_results):
    """保存多年集合的平均预测结果和平均统计指标"""
    outPath_gz = os.path.join(saveResultsPath, 'gz', period_name)
    os.makedirs(outPath_gz, exist_ok=True)
    
    for var_display in target_vars_display:
        prediction_arrays = [annual_results[year]['predictions'][var_display] for year in years]
        prediction_mean = average_arrays_with_nan_padding(prediction_arrays)
        pd.DataFrame(prediction_mean).to_csv(
            os.path.join(outPath_gz, f'dPL_gz_{var_display}_prediction.csv'),
            header=False, index=False)
        
        stats_frames = [annual_results[year]['stats'][var_display] for year in years]
        stats_mean = stats_frames[0][['site_id', 'lat', 'lon']].copy()
        metric_columns = ['bias', 'rmse', 'corr', 'r2', 'kge', 'nse', 'valid_points', 'total_points']
        stats_stack = np.stack([df[metric_columns].values for df in stats_frames], axis=0)
        stats_mean[metric_columns] = np.nanmean(stats_stack, axis=0)
        stats_mean.to_csv(os.path.join(outPath_gz, f'statErr_gz_{var_display}_sites.csv'), index=False)

# 主处理流程
annual_results = {}

for test_i, test_range in enumerate(tRangeTest):
    year = test_range[0] // 10000
    print(f"\n处理测试期: {test_range}")
    
    df_tr = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset[0], tRange=tRange)
    df_test = dbCsv.DataframeCsv(rootDB=rootDB, subset=subset[0], tRange=test_range)
    
    ngrid = df_test.getGeo()[0].shape[0]
    crd = df_test.getGeo()
    crd_lat, crd_lon = crd[0][:, None], crd[1][:, None]
    
    # 读取数据
    forcing = df_tr.getDataTs(varForcing, doNorm=True, rmNan=True)
    rawData = df_tr.getDataConst(varRaw, doNorm=True, rmNan=True)
    target_GF = df_tr.getDataTs(targetVar, doNorm=True, rmNan=True)
    forcing_test = df_test.getDataTs(varForcing, doNorm=True, rmNan=True)
    target_test = df_test.getDataTs(targetVar, doNorm=False, rmNan=False)
    
    # 创建输出目录
    outPath_base = saveResultsPath
    outPath_gz = os.path.join(outPath_base, 'gz', str(year))
    outPath_parameters = os.path.join(outPath_base, 'parameters', str(year))
    for path in [outPath_base, outPath_gz, outPath_parameters]:
        os.makedirs(path, exist_ok=True)
    
    # ==================== 保存CoLM默认参数 ====================
    print("处理CoLM默认参数...")
    try:
        parameters_CoLM = df_tr.getDataConst(varConst, doNorm=False, rmNan=True)
        
        if parameters_CoLM.shape[0] != ngrid:
            if parameters_CoLM.shape[0] > ngrid:
                parameters_CoLM = parameters_CoLM[:ngrid, :]
            else:
                padded = np.full((ngrid, len(varConst)), np.nan)
                padded[:parameters_CoLM.shape[0], :] = parameters_CoLM
                parameters_CoLM = padded
        
        param_with_crd = np.column_stack([crd_lat, crd_lon, parameters_CoLM])
        param_columns = ['lat', 'lon'] + varConst
        pd.DataFrame(param_with_crd).to_csv(
            os.path.join(outPath_parameters, f'{subset[0]}_CoLM_parameters.csv'),
            header=param_columns, index=False)
        print(f"CoLM参数保存至: {outPath_parameters}{subset[0]}_CoLM_parameters.csv")
    except Exception as e:
        print(f"处理CoLM参数时出错: {e}")
    
    # 运行模型
    custom_testModel(gz_model, ((forcing, forcing_test), target_GF), rawData, outPath_gz,
                     batchSize=100, lyrMode=0, scaleTrans=False, model_type='gz')
    
    # 读取预测结果
    yOut_calibrated = []
    for i in range(len(targetVar)):
        pred_file = os.path.join(outPath_gz, f'yOut_dPL_gz_var{i}')
        yOut_calibrated.append(pd.read_csv(pred_file, header=None).values)
    yOut_calibrated = np.stack(yOut_calibrated, axis=2)
    
    # 读取参数
    param_file = os.path.join(outPath_gz, 'param_dPL_gz')
    parameters_calibrated = pd.read_csv(param_file, header=None).values
    
    # 处理参数数量匹配
    if parameters_calibrated.shape[1] != len(varConst):
        if parameters_calibrated.shape[1] > len(varConst):
            parameters_calibrated = parameters_calibrated[:, :len(varConst)]
        else:
            padded = np.zeros((parameters_calibrated.shape[0], len(varConst)))
            padded[:, :parameters_calibrated.shape[1]] = parameters_calibrated
            parameters_calibrated = padded
    
    # 反归一化参数
    parameters_denorm = np.zeros_like(parameters_calibrated)
    for k in range(len(varConst)):
        try:
            parameters_denorm[:, k] = dbCsv.transNorm(parameters_calibrated[:, k], rootDB=rootDB,
                                                       fieldName=varConst[k], fromRaw=False, isConst=True)
        except:
            parameters_denorm[:, k] = parameters_calibrated[:, k]
    
    # 保存校准后的参数
    param_with_crd = np.column_stack([crd_lat, crd_lon, parameters_denorm])
    pd.DataFrame(param_with_crd).to_csv(
        os.path.join(outPath_parameters, f'{subset[0]}_dPL_gz_parameters_calibrated.csv'),
        header=param_columns, index=False)
    
    # 反归一化预测结果
    yOut_denorm = np.zeros_like(yOut_calibrated)
    for i, var in enumerate(targetVar):
        yOut_denorm[:, :, i] = dbCsv.transNorm(yOut_calibrated[:, :, i], rootDB=rootDB,
                                                fieldName=var, fromRaw=False)
    
    annual_results[year] = {'predictions': {}, 'stats': {}}
    
    # 计算统计指标
    for i, var_display in enumerate(target_vars_display):
        stats = calculate_site_statistics(yOut_denorm[:, :, i], target_test[:, :, i])
        stats_df = pd.DataFrame({
            'site_id': np.arange(ngrid), 'lat': crd_lat.flatten(), 'lon': crd_lon.flatten(),
            'bias': stats['bias'], 'rmse': stats['rmse'], 'corr': stats['corr'],
            'r2': stats['r2'], 'kge': stats['kge'], 'nse': stats['nse'],
            'valid_points': stats['valid_points'].astype(int), 'total_points': yOut_denorm.shape[1]
        })
        stats_df.to_csv(os.path.join(outPath_gz, f'statErr_gz_{var_display}_sites.csv'), index=False)
        annual_results[year]['stats'][var_display] = stats_df
    
    # 保存预测结果
    for i, var_display in enumerate(target_vars_display):
        result_file = os.path.join(outPath_gz, f'dPL_gz_{var_display}_prediction.csv')
        pd.DataFrame(yOut_denorm[:, :, i]).to_csv(result_file, header=False, index=False)
        annual_results[year]['predictions'][var_display] = yOut_denorm[:, :, i]
    

save_period_average('2006-2010', [2006, 2007, 2008, 2009, 2010], annual_results)
save_period_average('2006-2008', [2006, 2007, 2008], annual_results)
save_period_average('2008-2010', [2008, 2009, 2010], annual_results)

print("\n参数校准和评估完成！")