from image_load import ImagePathDataset, get_transform
from is_score import inception_score
import torch

if __name__ == '__main__':

    origin_image_dir = '/root/VAE/original_generated_images' #생성된 이미지가 저장된 디렉토리 경로 넣어주세요!
    improved_image_dir = '/root/VAE/improved_generated_images'

    # 이미지 데이터셋 로드
    transform = get_transform()
    origin_generated_images_dataset = ImagePathDataset(image_dir=origin_image_dir, transform=transform)
    improved_generated_images_dataset = ImagePathDataset(image_dir=improved_image_dir, transform=transform)

    # IS Score 계산
    '''
    mean_is는 IS의 평균값으로, 생성 이미지의 품질과 다양성을 나타내고, min 1, 이론적으로 max는 없지만 실제는 10 이내 범위에 있습니다. 높을수록 좋은 값
    std_is는 IS의 표준 편차로, 평가 결과 일관성 <<뺄까요..?

    '''
    mean_is = inception_score(origin_generated_images_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    print(f"두구두구 Original Inception Score: {mean_is}")
    mean_is = inception_score(improved_generated_images_dataset, cuda=True, batch_size=32, resize=True, splits=10)
    print(f"두구두구 Improved Inception Score: {mean_is}")