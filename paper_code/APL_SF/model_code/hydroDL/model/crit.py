import torch
import numpy as np

class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class RangeBoundLoss(torch.nn.Module):
    # limit parameters from going out of range
    def __init__(self,lb,ub,factor=1):
        super(RangeBoundLoss, self).__init__()
        self.lb=torch.Tensor(lb).cuda()
        self.ub=torch.Tensor(ub).cuda()
        self.factor=torch.tensor(factor).cuda()
        # self.factor.requires_grad = False

    def forward(self, x, *args):
        # loss = self.factor* torch.relu(x-self.ub).sum()  # default
        loss = 0
        if (torch.relu(x-self.ub)>0).sum() > 0:
            loss = self.factor* torch.relu(x-self.ub).sum()/(torch.relu(x-self.ub)>0).sum()
        else:
            loss = loss + torch.relu(x-self.ub).sum()
        if (torch.relu(self.lb-x)>0).sum() > 0:
            loss = loss+ self.factor* torch.relu(self.lb-x).sum()/(torch.relu(self.lb-x)>0).sum()
        else:
            loss = loss+ self.factor* torch.relu(self.lb-x).sum()
        return loss

class ParameterInequalityLoss(torch.nn.Module):
    # implements x(lIdx)<x(hIdx)
    # Relu ensures if x(lIdx)>x(hIdx), it gets a penalty
    def __init__(self,lIdx,hIdx,factor=1):
        super(ParameterInequalityLoss, self).__init__()
        self.lb=lIdx.cuda()
        self.ub=hIdx.cuda()
        self.factor=factor

    def forward(self, x, *args):
        loss = 0
        for i in range(0,len(self.lIdx)):
            loss = loss+self.factor*torch.relu(x[self.lIdx[i]]-x[self.hIdx[i]])
            # can this be written in vector format?
        return loss

class sumOfLoss(torch.nn.Module):
    # add multiple loss functions together.
    def __init__(self, *args, factor=1):
        super(sumOfLoss, self).__init__()
        self.lossFuncs = []
        for arg in args:
            self.lossFuncs.append(arg.cuda())

    def forward(self, output, target, *extra_args):
        # multiple inputs must be bundled into lists
        loss = 0
        if type(target) is tuple:
            target = list(target)

        for i in range(0, len(self.lossFuncs)):
            func = self.lossFuncs[i]
            if isinstance(func, RangeBoundLoss):
                # 对于RangeBoundLoss，传递额外的lb_batch和ub_batch参数
                loss = loss + func(output[i], *extra_args)
            else:
                # 其他损失函数保持原有调用方式
                if torch.is_tensor(target):
                    loss = loss + func(output[i], target)
                elif len(target) == 1:
                    loss = loss + func(output[i], target[0])
                else:
                    loss = loss + func(output[i], target[i])
        return loss



class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

## kge损失
class WeightedKgeLoss(torch.nn.Module):
    def __init__(self, weights=None):
        """
        weights: list or tensor, 每个目标变量的权重，长度等于输出变量数
                 如果不提供，则使用均匀权重
        """
        super(WeightedKgeLoss, self).__init__()
        if weights is not None:
            # 归一化权重使其和为1
            weights = np.array(weights, dtype=np.float32)
            self.weights = torch.tensor(weights / weights.sum(), dtype=torch.float32)
        else:
            self.weights = None
    
    def forward(self, output, target):
        """
        output: (batch_size, seq_len, n_outputs) - 模型预测值
        target: (batch_size, seq_len, n_outputs) - 观测值
        """
        batch_size, seq_len, n_outputs = output.shape
        
        # 处理NaN值
        kge_per_var = []
        
        for k in range(n_outputs):
            o = output[:, :, k]  # (batch_size, seq_len)
            t = target[:, :, k]
            
            # 移除NaN
            mask = ~torch.isnan(t)
            if mask.sum() < 10:  # 样本太少，跳过该变量
                kge_per_var.append(torch.tensor(0.0, device=output.device))
                continue
                
            o_valid = o[mask]
            t_valid = t[mask]
            
            # 计算相关系数
            o_centered = o_valid - o_valid.mean()
            t_centered = t_valid - t_valid.mean()
            r_num = (o_centered * t_centered).sum()
            r_den = torch.sqrt((o_centered**2).sum() * (t_centered**2).sum())
            r = r_num / (r_den + 1e-8)
            
            # 计算变异性比
            alpha = o_valid.std() / (t_valid.std() + 1e-8)
            
            # 计算偏差比
            beta = o_valid.mean() / (t_valid.mean() + 1e-8)
            
            # 计算KGE
            kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            kge_per_var.append(kge)
        
        # 计算加权平均KGE
        kge_tensor = torch.stack(kge_per_var)
        
        if self.weights is not None:
            # 将权重移到相同设备
            weights = self.weights.to(output.device)
            weighted_kge = (kge_tensor * weights).sum()
        else:
            weighted_kge = kge_tensor.mean()
        
        # 返回 1 - KGE 作为损失（最小化）
        return 1 - weighted_kge

class RmseLossANN(torch.nn.Module):
    def __init__(self, get_length=False):
        super(RmseLossANN, self).__init__()
        self.ind = get_length

    def forward(self, output, target):
        if len(output.shape) == 2:
            p0 = output[:, 0]
            t0 = target[:, 0]
        else:
            p0 = output[:, :, 0]
            t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        if self.ind is False:
            return loss
        else:
            Nday = p.shape[0]
            return loss, Nday

class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p-pmean
            t_ub = t-tmean
            temp = torch.sqrt(((p_ub - t_ub)**2).mean())
            loss = loss + temp
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t)**2).mean()
            loss = loss + temp
        return loss

class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample +1
        # minimize the opposite average NSE
        loss = -(losssum/nsample)
        return loss

class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST)+0.1)**2)
                losssum = losssum + temp
                nsample = nsample +1
        loss = losssum/nsample
        return loss
