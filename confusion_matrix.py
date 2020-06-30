import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def draw_matrix(true,predict):
    sns.set()
    f,ax=plt.subplots()
    labels = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9','10')
    C2= confusion_matrix(true, predict,labels=labels)
    sns.heatmap(C2,annot=True,ax=ax) #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    f.savefig("confusion_maxtrix.jpg")