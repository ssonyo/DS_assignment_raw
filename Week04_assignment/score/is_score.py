import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

# Inception Score 계산
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10):
    N = len(imgs) #생성된 이미지 수

    assert batch_size > 0
    assert N > batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size) #이미지 불러오기

    # IS score는 inception model을 기반으로 생성 이미지들의 분류 결과를 토대로 측정됩니다
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # InceptionV3는 input size가 (299, 299)
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Inception v3 모델은 ImageNet으로 학습된 모델이라 1000개 클래스로 분류 예측
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    '''
    Inception Score는 생성된 이미지의 품질, 다양성을 고려하여 계산됩니다.
    모델이 출력한 각 이미지의 클래스 확률 분포 P(y|x)와 전체 이미지의 평균 클래스 확률 분포 P(y) 간의 차이를 측정
    그 차이를 KL-Divergence로 계싼한 후, 지수 함수로 변환해서 최종 점수를 산출합니다.
    '''
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)
