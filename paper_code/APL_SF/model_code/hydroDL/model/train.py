import numpy as np
import torch
import time
import os
import hydroDL
from hydroDL.model import rnn
import pandas as pd
import time
from torch.optim import lr_scheduler
import math
import numpy as np
import time
from hydroDL.model import rnn, crit

def trainModel(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               optLr=1,
               optRho=0.9,
               optWd=0,
               lyrMode=0,
               scaleTrans=True,
               routingModel=False,
               lenF=10,
               csv_name=None,
               # ===== 新增：优化器和调度器参数 =====
               optimizer_type='adam',          # 'adam' 或 'adadelta'
               scheduler_type='cosine_warmup', # 'cosine', 'cosine_warmup', 'cosine_warmrestart', 'none'
               T_max=None,                     # 余弦退火周期
               eta_min=0,                      # 最小学习率
               warmup_epochs=0,                # 预热epoch数
               T_0=None,                       # 重启周期
               T_mult=2):                      # 周期倍增因子
    
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x  
    ngrid, nt, nx = x.shape   
    if c is not None:  
        if type(c) is tuple or type(c) is list:
            est_c, fixed_c = c
            nx = nx + est_c.shape[-1] + fixed_c.shape[-1]
            fixed_params = True
        else:
            nx = nx + c.shape[-1]
            fixed_params = False

    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))

    if hasattr(model, 'ctRm'):   
        if model.ctRm is True:
            nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * (rho - model.ct) / ngrid / nt)))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()
    
    # ===== 根据类型选择优化器 =====
    if optimizer_type == 'adam':
        optim = torch.optim.Adam(
            model.parameters(), 
            lr=optLr, 
            weight_decay=optWd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print(f"✓ 使用Adam优化器，初始学习率: {optLr}, 权重衰减: {optWd}")
        
    elif optimizer_type == 'adadelta':
        optim = torch.optim.Adadelta(
            model.parameters(), 
            lr=optLr, 
            rho=optRho, 
            weight_decay=optWd
        )
        print(f"✓ 使用Adadelta优化器，初始学习率: {optLr}")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}，请使用 'adam' 或 'adadelta'")
    
    # ===== 根据类型选择学习率调度器 =====
    if scheduler_type == 'cosine':
        if T_max is None:
            T_max = nEpoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=T_max, eta_min=eta_min
        )
        print(f"✓ 使用余弦退火调度器：T_max={T_max}, eta_min={eta_min}")
        
    elif scheduler_type == 'cosine_warmup':
        if T_max is None:
            T_max = nEpoch - warmup_epochs
        
        # 使用LambdaLR实现（更稳定）
        def lr_lambda(epoch):
            base_lr = optLr
            if epoch < warmup_epochs:
                # 线性预热阶段
                if warmup_epochs > 0:
                    alpha = epoch / warmup_epochs
                else:
                    alpha = 1.0
                return 0.0 + (1.0 - 0.0) * alpha  # 从0开始预热
            else:
                # 余弦退火阶段
                if T_max > 0:
                    progress = min(1.0, (epoch - warmup_epochs) / T_max)
                else:
                    progress = 1.0
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                eta_min_ratio = eta_min / base_lr if base_lr > 0 else 0
                return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        print(f"✓ 使用带预热的余弦退火调度器：")
        print(f"  预热epoch: {warmup_epochs}, 余弦周期: {T_max}, 最小学习率: {eta_min}")
        
    elif scheduler_type == 'cosine_warmrestart':
        if T_0 is None:
            T_0 = nEpoch // 3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        print(f"✓ 使用余弦退火重启调度器：T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
        
    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = None
        print("✓ 不使用学习率调度器")
        
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    model.zero_grad()
    
    # ===== 根据目标变量数量动态生成var_indices =====
    ny = y.shape[-1]  # 目标变量数量
    
    if ny == 1:
        var_indices = {'SM1': 0}
        print(f"✓ 检测到单目标变量，var_indices: {var_indices}")
    elif ny == 10:
        var_indices = {
            'ET': 0, 'Runoff': 1,
            'SM1': 2, 'SM2': 3, 'SM3': 4, 'SM4': 5,
            'ST1': 6, 'ST2': 7, 'ST3': 8, 'ST4': 9
        }
        print(f"✓ 检测到10个目标变量，var_indices已设置")
    else:
        var_indices = {}
        for i in range(ny):
            var_indices[f'var_{i}'] = i
        print(f"✓ 检测到{ny}个目标变量，var_indices: {var_indices}")
    
    if saveFolder is not None:
        if csv_name is None:
            csv_name = 'run'
        runFile = os.path.join(saveFolder, csv_name + '.csv')
        rf = open(runFile, 'w+')
        
        # ===== 动态生成CSV表头 =====
        if type(lossFun) in [hydroDL.model.crit.sumOfLoss]:
            header = 'Epoch,Total_Loss,RmseLoss,RangeBoundLoss,'
        else:
            header = 'Epoch,Total_Loss,'
        
        for var_name in var_indices.keys():
            header += f'{var_name}_RMSE,'
        header += 'Time,LR\n'
        rf.write(header)
    
    for iEpoch in range(1, nEpoch + 1): 
        # 初始化累积损失
        lossEp = 0
        lossEp1 = 0  # 总RMSE损失
        lossEp2 = 0  # 总Range损失
        
        # 动态初始化每个变量的RMSE损失
        var_rmse_losses = {var_name: 0 for var_name in var_indices.keys()}
        
        t0 = time.time()
        current_lr = optim.param_groups[0]['lr']
        
        for iIter in range(0, nIterEp):
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])  
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)         
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP = model(xTrain)
                loss = lossFun(yP, yTrain)
                
                with torch.no_grad():
                    for var_name, var_idx in var_indices.items():
                        if var_idx >= yP.shape[-1]:
                            continue
                        var_pred = yP[:, :, var_idx:var_idx+1]
                        var_target = yTrain[:, :, var_idx:var_idx+1]
                        if hasattr(lossFun, 'lossFuncs'):
                            var_loss = lossFun.lossFuncs[0](var_pred, var_target)
                        else:
                            var_loss = crit.RmseLoss()(var_pred, var_target)
                        var_rmse_losses[var_name] += var_loss.item() if isinstance(var_loss, torch.Tensor) else float(var_loss)
                
            elif type(model) in [rnn.CudnnLstmModel_R2P]:  
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP, Param_R2P = model(xTrain)
                loss = lossFun(yP, yTrain)
                
                with torch.no_grad():
                    for var_name, var_idx in var_indices.items():
                        if var_idx >= yP.shape[-1]:
                            continue
                        var_pred = yP[:, :, var_idx:var_idx+1]
                        var_target = yTrain[:, :, var_idx:var_idx+1]
                        if hasattr(lossFun, 'lossFuncs'):
                            var_loss = lossFun.lossFuncs[0](var_pred, var_target)
                        else:
                            var_loss = crit.RmseLoss()(var_pred, var_target)
                        var_rmse_losses[var_name] += var_loss.item() if isinstance(var_loss, torch.Tensor) else float(var_loss)
                
            elif type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:  
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])   
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None)
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)
                yP = model(xTrain, zTrain)
                loss = lossFun(yP, yTrain)
                
                with torch.no_grad():
                    for var_name, var_idx in var_indices.items():
                        if var_idx >= yP.shape[-1]:
                            continue
                        var_pred = yP[:, :, var_idx:var_idx+1]
                        var_target = yTrain[:, :, var_idx:var_idx+1]
                        if hasattr(lossFun, 'lossFuncs'):
                            var_loss = lossFun.lossFuncs[0](var_pred, var_target)
                        else:
                            var_loss = crit.RmseLoss()(var_pred, var_target)
                        var_rmse_losses[var_name] += var_loss.item() if isinstance(var_loss, torch.Tensor) else float(var_loss)
                
            elif type(model) in [rnn.CudnnLstmModel_Inv]:              
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho]) 
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                zTrain = selectSubset(z, iGrid, iT, rho)
                
                yP1, Param_R2P = model(xTrain, zTrain, lyrMode=lyrMode, scaleTrans=True, 
                                       routingModel=routingModel, lenF=lenF, fixed_params=fixed_params)
                
                if type(lossFun) in [hydroDL.model.crit.sumOfLoss]: 
                    yP = [yP1, Param_R2P]
                else:
                    yP = yP1

                # 损失计算
                if type(lossFun) in [hydroDL.model.crit.sumOfLoss]:
                    loss_total = lossFun(yP, yTrain)
                    loss1 = lossFun.lossFuncs[0](yP[0], yTrain)
                    loss2 = lossFun.lossFuncs[1](yP[1], yTrain)
                    
                    loss = loss_total
                    lossEp1 += loss1.item() if isinstance(loss1, torch.Tensor) else float(loss1)
                    lossEp2 += loss2.item() if isinstance(loss2, torch.Tensor) else float(loss2)
                    
                    with torch.no_grad():
                        for var_name, var_idx in var_indices.items():
                            if var_idx >= yP[0].shape[-1]:
                                continue
                            var_pred = yP[0][:, :, var_idx:var_idx+1]
                            var_target = yTrain[:, :, var_idx:var_idx+1]
                            var_rmse = lossFun.lossFuncs[0](var_pred, var_target)
                            var_rmse_losses[var_name] += var_rmse.item() if isinstance(var_rmse, torch.Tensor) else float(var_rmse)
                else:
                    loss = lossFun(yP, yTrain)
                    
                    with torch.no_grad():
                        for var_name, var_idx in var_indices.items():
                            if var_idx >= yP.shape[-1]:
                                continue
                            var_pred = yP[:, :, var_idx:var_idx+1]
                            var_target = yTrain[:, :, var_idx:var_idx+1]
                            if hasattr(lossFun, 'lossFuncs'):
                                var_loss = lossFun.lossFuncs[0](var_pred, var_target)
                            else:
                                var_loss = lossFun(var_pred, var_target)
                            var_rmse_losses[var_name] += var_loss.item() if isinstance(var_loss, torch.Tensor) else float(var_loss)
            else:
                raise Exception('unknown model')
            
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp += loss.item()

        # ===== 更新学习率 =====
        if scheduler is not None:
            scheduler.step()
            new_lr = optim.param_groups[0]['lr']
            if new_lr != current_lr and abs(new_lr - current_lr) > 1e-10:
                print(f"Epoch {iEpoch}: 学习率从 {current_lr:.6f} 调整到 {new_lr:.6f}")
        
        # ===== 平均化所有损失 =====
        lossEp = lossEp / nIterEp
        lossEp1 = lossEp1 / nIterEp if nIterEp > 0 else 0
        lossEp2 = lossEp2 / nIterEp if nIterEp > 0 else 0
        
        for var_name in var_rmse_losses:
            var_rmse_losses[var_name] /= nIterEp
        
        elapsed_time = time.time() - t0
        current_lr = optim.param_groups[0]['lr']
        
        # ===== 打印详细损失信息 =====
        logStr = f'Epoch {iEpoch} '
        logStr += f'Total {lossEp:.4f} '
        logStr += f'RMSE {lossEp1:.4f} '
        if type(lossFun) in [hydroDL.model.crit.sumOfLoss]:
            logStr += f'Range {lossEp2:.4f} '
        logStr += '| '
        for var_name in var_indices.keys():
            logStr += f'{var_name} {var_rmse_losses[var_name]:.4f} '
        logStr += f'| time {elapsed_time:.1f}s lr {current_lr:.6e}'
        print(logStr)
        
        # ===== 写入CSV文件 =====
        if saveFolder is not None:
            if type(lossFun) in [hydroDL.model.crit.sumOfLoss]:
                rf.write(f'{iEpoch},{lossEp:.6f},{lossEp1:.6f},{lossEp2:.6f},')
            else:
                rf.write(f'{iEpoch},{lossEp:.6f},')
            
            for var_name in var_indices.keys():
                rf.write(f'{var_rmse_losses[var_name]:.6f},')
            
            rf.write(f'{elapsed_time:.2f},{current_lr:.6e}\n')
            
            if iEpoch % 10 == 0:
                rf.flush()
        
        if saveFolder is not None:
            if iEpoch % saveEpoch == 0:
                modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
                print(f'  ✓ 模型保存至: {modelFile}')
    
    if saveFolder is not None:
        rf.close()
        print(f'训练完成，损失记录已保存至: {runFile}')
    return model

def compute_validation_loss_simple(model, x_valid, y_valid, c_valid, lossFun, batchSize):
    """简单模型的验证损失计算"""
    import numpy as np
    import torch
    
    ngrid, nt, nx = x_valid.shape
    ny = y_valid.shape[-1]
    
    # 如果c不为空，需要拼接
    if c_valid is not None:
        # 扩展c到与x相同的时序长度
        c_expanded = np.repeat(c_valid[:, np.newaxis, :], nt, axis=1)
        x_combined = np.concatenate([x_valid, c_expanded], axis=2)
    else:
        x_combined = x_valid
    
    # 转换为tensor
    x_tensor = torch.from_numpy(np.swapaxes(x_combined, 1, 0)).float()
    y_tensor = torch.from_numpy(np.swapaxes(y_valid, 1, 0)).float()
    
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
        y_tensor = y_tensor.cuda()
    
    # 分批计算损失
    n_batches = (ngrid + batchSize - 1) // batchSize
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(0, ngrid, batchSize):
            x_batch = x_tensor[:, i:i+batchSize, :]
            y_batch = y_tensor[:, i:i+batchSize, :]
            
            y_pred = model(x_batch)
            loss = lossFun(y_pred, y_batch)
            total_loss += loss.item() * (min(batchSize, ngrid - i))
    
    return total_loss / ngrid


def compute_validation_loss(model, x_valid, y_valid, c_valid, lossFun, 
                           batchSize, rho, lyrMode, routingModel, lenF, fixed_params):
    """复杂模型（CudnnLstmModel_Inv）的验证损失计算"""
    import numpy as np
    import torch
    
    ngrid, nt, nx = x_valid.shape
    
    # 简化处理：使用整个验证集计算损失
    total_loss = 0.0
    n_batches = (ngrid + batchSize - 1) // batchSize
    
    for i in range(0, ngrid, batchSize):
        i_end = min(i + batchSize, ngrid)
        
        x_batch = x_valid[i:i_end, :, :]
        y_batch = y_valid[i:i_end, :, :]
        
        if c_valid is not None:
            c_batch = c_valid[i:i_end, :]
            # 扩展c到时序维度
            c_expanded = np.repeat(c_batch[:, np.newaxis, :], nt, axis=1)
            x_combined = np.concatenate([x_batch, c_expanded], axis=2)
        else:
            x_combined = x_batch
        
        # 转换为tensor
        x_tensor = torch.from_numpy(np.swapaxes(x_combined, 1, 0)).float()
        y_tensor = torch.from_numpy(np.swapaxes(y_batch, 1, 0)).float()
        
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
            y_tensor = y_tensor.cuda()
        
        with torch.no_grad():
            y_pred = model(x_tensor)
            loss = lossFun(y_pred, y_tensor)
            total_loss += loss.item() * (i_end - i)
    
    return total_loss / ngrid

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model

def testModel(model, x, c, *, batchSize=None, filePathLst=None, doMC=False,
              outModel=None, lyrMode=0, savePath=None, scaleTrans=True, routingModel=False, lenF=10):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel_Inv:
            if type(x) is tuple or type(x) is list:
                xTrain_temp, xTest_temp = x  

        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None

    else:
        z = None

    if type(model) is rnn.CudnnLstmModel_Inv:
        ngrid, ntTrain, nx = xTrain_temp.shape
        ntTest = xTest_temp.shape[1]
    else:
        ngrid, nt, nx = x.shape

    if c is not None:
        if type(c) is tuple or type(c) is list:  ## only for VIC LAI factor
            est_c, fixed_c = c
            nx = nx + est_c.shape[-1] + fixed_c.shape[-1]
            nc_est = est_c.shape[-1]
            nc_fixed = fixed_c.shape[-1]
            fixed_params = True
        else:
            nx = nx + c.shape[-1]
            nc = c.shape[-1]
            fixed_params = False

    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        if type(model) in [rnn.CudnnLstmModel_R2P, rnn.CudnnLstmModel_Inv]:
            filePathLst = ['out' + str(x) for x in range(ny+1)]
        else:
            filePathLst = ['out' + str(x) for x in range(ny)]
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # 初始化参数数组（只在需要时）
    if type(model) in [rnn.CudnnLstmModel_R2P, rnn.CudnnLstmModel_Inv]:
        all_parameters = []
    
    # forward for each batch
    for i in range(0, len(iS)):
        # print('batch {}'.format(i)) # temp comment for SCEUA
        if type(model) is rnn.CudnnLstmModel_Inv:
            xTrainTemp = xTrain_temp[iS[i]:iE[i], :, :]
            xTestTemp = xTest_temp[iS[i]:iE[i], :, :]
        else:
            xTemp = x[iS[i]:iE[i], :, :]

        if c is not None:
            if type(model) is rnn.CudnnLstmModel_Inv:
                if type(c) is tuple or type(c) is list:  ## only for VIC LAI factor
                    cTemp_est = np.repeat(
                        np.reshape(est_c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc_est]), ntTrain, axis=1)
                    cTemp_fixed = np.repeat(
                        np.reshape(fixed_c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc_fixed]), ntTrain, axis=1)
                    xTrain_Inv = torch.from_numpy(np.swapaxes(xTrainTemp, 1, 0)).float()
                    xTest_Inv = torch.from_numpy(np.swapaxes(xTestTemp, 1, 0)).float()
                else:
                    cTemp = np.repeat(
                        np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), ntTrain, axis=1)
                    xTrain_Inv = torch.from_numpy(np.swapaxes(xTrainTemp, 1, 0)).float()
                    xTest_Inv = torch.from_numpy(np.swapaxes(xTestTemp, 1, 0)).float()
            else:
                cTemp = np.repeat(
                    np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
                xTest = torch.from_numpy(
                    np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        if torch.cuda.is_available():
            if type(model) is rnn.CudnnLstmModel_Inv:
                xTrain_Inv = xTrain_Inv.cuda()
                xTest_Inv = xTest_Inv.cuda()
            else:
                xTest = xTest.cuda()
        if z is not None:
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.cuda()
        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
            # if z is not None:
            #     xTest = torch.cat((xTest, zTest), dim=2)
            yP = model(xTest)
            if doMC is not False:
                ySS = np.zeros(yP.shape)
                yPnp=yP.detach().cpu().numpy()
                for k in range(doMC):
                    # print(k)
                    yMC = model(xTest, doDropMC=True).detach().cpu().numpy()
                    ySS = ySS+np.square(yMC-yPnp)
                ySS = np.sqrt(ySS)/doMC

        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
            yP = model(xTest, zTest)

        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            yP = model(xTest, zTest)

        if type(model) in [rnn.CudnnLstmModel_R2P]:
            xTemp = torch.from_numpy(np.swapaxes(xTemp,1,0)).float()
            cTemp = torch.from_numpy(np.swapaxes(cTemp,1,0)).float()
            xTemp = xTemp.cuda()
            cTemp = cTemp.cuda()
            xTest_tuple = (xTemp, cTemp)
            if outModel is None:
                yP, Param_R2P = model(xTest_tuple, outModel = outModel, lyrMode=lyrMode)
                Parameters_R2P = Param_R2P.detach().cpu().numpy().swapaxes(0, 1)
            else:
                Param_R2P = model(xTest_tuple, outModel = outModel)
                Parameters_R2P = Param_R2P.detach().cpu().numpy()
                hymod_forcing = xTemp.detach().cpu().numpy().swapaxes(0, 1)

                runFile = os.path.join(savePath, 'hymod_run.csv')
                rf = open(runFile, 'a+')

                q = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
                evap = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
                for pix in range(hymod_forcing.shape[0]):
                    model_hymod = rnn.hymod(a=Parameters_R2P[pix,0], b=Parameters_R2P[pix,1],\
                        cmax=Parameters_R2P[pix,2], rq=Parameters_R2P[pix,3],\
                            rs=Parameters_R2P[pix,4], s=Parameters_R2P[pix,5],\
                                slow=Parameters_R2P[pix,6],\
                                    fast=[Parameters_R2P[pix,7], Parameters_R2P[pix,8], Parameters_R2P[pix,9]])
                    for hymod_t in range(hymod_forcing.shape[1]):
                        q[pix, hymod_t], evap[pix, hymod_t] = model_hymod.advance(hymod_forcing[pix,hymod_t,0],hymod_forcing[pix,hymod_t,1])
                        nstepsLst = '{:.5f} {:.5f} {:.5f} {:.5f}'.format(hymod_forcing[pix,hymod_t,0], hymod_forcing[pix,hymod_t,1], q[pix,hymod_t], evap[pix,hymod_t])
                        print(nstepsLst)

                        rf.write(nstepsLst + '\n')      
        if type(model) in [rnn.CudnnLstmModel_Inv]:
            test = True

            if fixed_params is True:
                cTemp_fixed = torch.from_numpy(np.swapaxes(cTemp_fixed, 1, 0)).float()
                cTemp_fixed = cTemp_fixed.cuda()
                cTemp_est = torch.from_numpy(np.swapaxes(cTemp_est, 1, 0)).float()
                cTemp_est = cTemp_est.cuda()
                xTest_tuple = (xTrain_Inv, xTest_Inv, cTemp_est, cTemp_fixed)
            else:
                cTemp = torch.from_numpy(np.swapaxes(cTemp, 1, 0)).float()
                cTemp = cTemp.cuda()
                xTest_tuple = (xTrain_Inv, xTest_Inv, cTemp)

            yP, Param_R2P = model(xTest_tuple, zTest, testMode=test, outModel=outModel, lyrMode=lyrMode,
                                  scaleTrans=True, routingModel=routingModel,
                                  lenF=lenF, fixed_params=fixed_params)  # will also send in the y for inversion generator

        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output - 修复后的逻辑
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        
        # 对于需要保存参数的模型，收集参数数据
        if type(model) in [rnn.CudnnLstmModel_R2P, rnn.CudnnLstmModel_Inv]:
            Parameters_R2P = Param_R2P.detach().cpu().numpy().swapaxes(0, 1)
            all_parameters.append(Parameters_R2P)

        if doMC is not False:
            for k in range(ny):
                f = fLst[ny+k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(
                    f, header=False, index=False)

        model.zero_grad()
        torch.cuda.empty_cache()

    # 保存参数数据（在循环外部）
    if type(model) in [rnn.CudnnLstmModel_R2P, rnn.CudnnLstmModel_Inv]:
        if all_parameters:
            # 合并所有批次的参数
            all_parameters_concat = np.concatenate(all_parameters, axis=0)
            # 写入最后一个文件（参数文件）
            f = fLst[ny]  # 第ny个文件是参数文件
            pd.DataFrame(all_parameters_concat).to_csv(f, header=False, index=False)
            print(f"参数数据形状: {all_parameters_concat.shape}")
            print(f"参数保存到: {filePathLst[ny]}")

    for f in fLst:
        f.close()

    if type(model) in [rnn.CudnnLstmModel_R2P, rnn.CudnnLstmModel_Inv]:
        Parameters_R2P = torch.from_numpy(all_parameters_concat) if 'all_parameters_concat' in locals() else None
        if outModel is None:
            return yOut, Parameters_R2P
        else:
            return q, evap, Parameters_R2P
    else:
        return yOut


def testModelCnnCond(model, x, y, *, batchSize=None):
    ngrid, nt, nx = x.shape
    ct = model.ct
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    # cTest = torch.from_numpy(np.swapaxes(y[:, 0:ct, :], 1, 0)).float()
    cTest = torch.zeros([ct, ngrid, y.shape[-1]], requires_grad=False)
    for k in range(ngrid):
        ctemp = y[k, 0:ct, 0]
        i0 = np.where(np.isnan(ctemp))[0]
        i1 = np.where(~np.isnan(ctemp))[0]
        if len(i1) > 0:
            ctemp[i0] = np.interp(i0, i1, ctemp[i1])
            cTest[:, k, 0] = torch.from_numpy(ctemp)

    if torch.cuda.is_available():
        xTest = xTest.cuda()
        cTest = cTest.cuda()
        model = model.cuda()

    model.train(mode=False)

    yP = torch.zeros([nt - ct, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    for i in range(0, len(iS)):
        xTemp = xTest[:, iS[i]:iE[i], :]
        cTemp = cTest[:, iS[i]:iE[i], :]
        yP[:, iS[i]:iE[i], :] = model(xTemp, cTemp)
    yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
    return yOut


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomIndex(ngrid, nt, dimSubset):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   # hack
        iGrid = np.arange(0,len(iGrid))  # hack
        if nt <= rho:
            iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        if type(c) is tuple or type(c) is list:
            est_c, fixed_c = c
            nc_est = est_c.shape[-1]
            nc_fixed = fixed_c.shape[-1]
            temp1 = np.repeat(
                np.reshape(est_c[iGrid, :], [batchSize, 1, nc_est]), rho, axis=1)
            temp2 = np.repeat(
                np.reshape(fixed_c[iGrid, :], [batchSize, 1, nc_fixed]), rho, axis=1)
            cTensor_est = torch.from_numpy(np.swapaxes(temp1, 1, 0)).float()
            cTensor_fixed = torch.from_numpy(np.swapaxes(temp2, 1, 0)).float()
            # cTensor = (cTensor_est, cTensor_fixed)
            if (tupleOut):
                if torch.cuda.is_available():
                    xTensor = xTensor.cuda()
                    cTensor_est = cTensor_est.cuda()
                    cTensor_fixed = cTensor_fixed.cuda()
                out = (xTensor, cTensor_est, cTensor_fixed)
            else:
                out = torch.cat((xTensor, cTensor_est, cTensor_fixed), 2)

        else:
            nc = c.shape[-1]
            temp = np.repeat(
                np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho, axis=1)
            cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

            if (tupleOut):
                if torch.cuda.is_available():
                    xTensor = xTensor.cuda()
                    cTensor = cTensor.cuda()
                out = (xTensor, cTensor)
            else:
                out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out
