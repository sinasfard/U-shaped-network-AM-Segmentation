from torchmetrics import Dice, Precision, Recall, Accuracy, F1Score, ConfusionMatrix

cfs = ConfusionMatrix(task='BINARY').to(device)
def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    f1 = 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])
    IU = tp / (pos + res - tp)
    dice = 2 * tp / (pos+res)
    meanDice = np.mean(dice)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    # Fpr = confusion[1][0]
    # Tpr = confusion[1][1]

    # return  meanIU,meanDice,Acc,Se,Sp,IU,f1
    return  IU[1],dice[1],Acc,Se,Sp,IU,f1
