from pathlib import Path
import matplotlib.pyplot as plt
from os import listdir
import matplotlib
from IPython.display import display
from pandas import DataFrame
import seaborn as sns


# 정답 맞춘게 몇개인지 세는 것

def get_correct_count(pred, target):
    return pred.eq(target.view_as(pred)).sum().item()


# label: ['Can', 'Glass', 'Plastic', 'Paper','Vinyl', 'PET','Paperpack']
#           0,1,2,3,4,5,6
def check_correct(pred, target):
    pred = pred.view_as(target)
    pred = pred.int()
    target = target.int()
    return pred, target


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group['lr']:
            return param_group['lr']
        pass

    return None


def touch_dir(target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    pass


def show_image(img):
    display(img)
    pass


# graph 그리는 함수, 최고&최저도 같이 나타냄
def draw_result(history):
    # 도화지 생성
    fig = plt.figure()

    # 큰 도화지(fig)를 1x1로 나눈 도화지에 첫 번째 칸에 ax_acc 그래프를 추가한다
    ax_acc = fig.add_subplot(1, 1, 1)

    # 정확도 그래프 그리기
    y_range = range(len(history['val_acc']))

    ax_acc.plot(y_range, history['val_acc'], label='Accuracy(%)', color='darkred')
    # 축 이름
    plt.xlabel('num_epochs')
    plt.ylabel('Accuracy(%)')
    ax_acc.grid(linestyle='--', color='lavender')
    # ax_acc.set_ylim([20, 100])

    # ax_loos 그래프를 x축을 동일하게 사용하여 겹쳐 그리기 선언
    ax_loss = ax_acc.twinx()
    ax_loss.plot(y_range, history['val_loss'], label='Loss', color='darkblue')
    plt.ylabel('Loss')
    ax_loss.yaxis.tick_right()
    ax_loss.grid(linestyle='--', color='lavender')
    # ax_loss.set_ylim([0, 2])

    plt.legend()
    plt.savefig('result_image/graph.png', dpi=300)

    print("min Acc: ", min(history['val_acc']))
    print("max Acc: ", max(history['val_acc']))
    print("min Loss: ", min(history['val_loss']))
    print("max Loss: ", max(history['val_loss']))
    pass


def draw_heatmap(matrix):
    df = DataFrame(matrix, columns=['Can', 'Glass', 'Plastic', 'Paper', 'Vinyl', 'PET','Paperpack'],
                   index=['Can', 'Glass', 'Plastic', 'Paper', 'Vinyl', 'PET','Paperpack'])#paperpack 잠시 뺌

    hm = sns.heatmap(df, annot=True, fmt="d")

    plt.show()

    fig = hm.get_figure()

    fig.savefig('result_image/heatmap.png', dpi=300)
    pass
