
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loss functions ============================================================================================================================================

## CLL ------------------------------------------------------------------------------------------------------------------------------------------------------

'''
Cite from: 
    https://github.com/takashiishida/comp/blob/master/utils_data.py
'''
def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

'''
Cite from: 
    Ishida, T., Niu, G., Menon, A., & Sugiyama, M. (2019, May). Complementary-label learning for arbitrary losses and models. In International conference on machine learning (pp. 2971-2980). PMLR.
    https://github.com/takashiishida/comp/blob/master/utils_algo.py
'''
def assump_free_loss(f, num_classes, labels, ccp, device):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, num_classes=num_classes, labels=labels, ccp=ccp, beta=np.inf, device=device)

'''
Cite from: 
    Ishida, T., Niu, G., Menon, A., & Sugiyama, M. (2019, May). Complementary-label learning for arbitrary losses and models. In International conference on machine learning (pp. 2971-2980). PMLR.
    https://github.com/takashiishida/comp/blob/master/utils_algo.py
'''
def non_negative_loss(f, num_classes, labels, ccp, beta, device):
    ccp = torch.from_numpy(ccp).float().to(device)
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(num_classes, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(num_classes).to(device)
    for k in range(num_classes):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.view(-1,1).repeat(1,num_classes)
            neglog_k = torch.masked_select(neglog, idxs).view(-1,num_classes)
            temp_loss_vector[k] = -(num_classes-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max 
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < num_classes:
        count = np.append(count, 0) # when largest label is below num_classes, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(num_classes, requires_grad=True).view(-1,1).to(device)-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)

'''
Cite from: 
    Yu, X., Liu, T., Gong, M., & Tao, D. (2018). Learning with biased complementary labels. In Proceedings of the European conference on computer vision (ECCV) (pp. 68-83).
    Ishida, T., Niu, G., Menon, A., & Sugiyama, M. (2019, May). Complementary-label learning for arbitrary losses and models. In International conference on machine learning (pp. 2971-2980). PMLR.
    https://github.com/takashiishida/comp/blob/master/utils_algo.py
'''
def forward_loss(f, num_classes, labels, device):
    Q = torch.ones(num_classes,num_classes) * 1/(num_classes-1)
    Q = Q.to(device)
    for k in range(num_classes):
        Q[k,k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long())

# CLL PC Loss (sigmoid loss)
'''
Cite from: 
    Ishida, T., Niu, G., Hu, W., & Sugiyama, M. (2017). Learning from complementary labels. Advances in neural information processing systems, 30.
    https://github.com/takashiishida/comp/blob/master/utils_algo.py
'''
def pc_loss(f, num_classes, labels):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, num_classes)
    loss_matrix = sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
    M1, M2 = num_classes*(num_classes-1)/2, num_classes-1
    pc_loss = torch.sum(loss_matrix)*(num_classes-1)/len(labels) - M1 + M2
    return pc_loss

'''
Cite from: 
    https://github.com/takashiishida/comp/blob/master/utils_algo.py
'''
def chosen_loss_c(f, num_classes, labels, ccp, meta_method, device):
    class_loss_torch = None
    if meta_method=='free':
        final_loss, class_loss_torch = assump_free_loss(f=f, num_classes=num_classes, labels=labels, ccp=ccp, device=device)
    elif meta_method=='nn':
        final_loss, class_loss_torch = non_negative_loss(f=f, num_classes=num_classes, labels=labels, beta=0, ccp=ccp, device=device)
    elif meta_method=='forward':
        final_loss = forward_loss(f=f, num_classes=num_classes, labels=labels, device=device)
    elif meta_method=='pc':
        final_loss = pc_loss(f=f, num_classes=num_classes, labels=labels)
    return final_loss, class_loss_torch


## PLL ------------------------------------------------------------------------------------------------------------------------------------------------------

# RC, CC Loss (cross entropy)
'''
Cite from: 
    Feng, L., Lv, J., Han, B., Xu, M., Niu, G., Geng, X., ... & Sugiyama, M. (2020). Provably consistent partial-label learning. Advances in neural information processing systems, 33, 10948-10960.
    https://github.com/hongwei-wen/LW-loss-for-partial-label/tree/master
    https://github.com/hongwei-wen/LW-loss-for-partial-label/blob/master/utils/utils_loss.py
'''
def rc_loss(outputs, scaled_labels):
    '''
    Args:
        outputs: torch.tensor
            logits
        scaled_labels: torch.tensor
            one-hot encoded partial labels (each sum of row is 1 (scaled))
    '''
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * scaled_labels
    average_loss = -((final_outputs).sum(dim=1)).mean()
    return average_loss

def cc_loss(outputs, labels):
    '''
    Args:
        outputs: torch.tensor
            logits
        scaled_labels: torch.tensor
            one-hot encoded partial labels (not scaled)
    '''
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * labels
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss

'''
Cite from: 
    Lv, J., Xu, M., Feng, L., Niu, G., Geng, X., & Sugiyama, M. (2020, November). Progressive identification of true labels for partial-label learning. In international conference on machine learning (pp. 6500-6510). PMLR.
    https://github.com/lvjiaqi77/PRODEN/blob/master/utils/utils_loss.py
'''
def partial_loss(output1, target):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY

    return loss, new_target