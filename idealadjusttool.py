import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)

    # 이미지가 로드되지 않은 경우 오류 메시지 출력
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 밝기 조정
    brightness_adjusted = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # 알파: 대비 조정, 베타: 밝기 조정

    # 샤프닝
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 샤프닝 커널
    sharpened_image = cv2.filter2D(brightness_adjusted, -1, kernel)

    # 엣지 검출
    edges = cv2.Canny(sharpened_image, 100, 200)

    # 결과 시각화
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(brightness_adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Brightness Adjusted Image')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened Image')

    plt.subplot(2, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection Image')

    plt.tight_layout()
    plt.show()

    # 사진 히스토그램 출력
    plot_image_histogram(image, 'Original Image Histogram')
    plot_image_histogram(brightness_adjusted, 'Brightness Adjusted Image Histogram')

    # Composition recommendation
    recommend_composition(image, brightness_adjusted)

def plot_image_histogram(image, title):
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 밝기 구간별로 픽셀 수 계산
    darkest = np.sum((gray_image >= 0) & (gray_image < 51))
    dark = np.sum((gray_image >= 51) & (gray_image < 102))
    mid = np.sum((gray_image >= 102) & (gray_image < 153))
    light = np.sum((gray_image >= 153) & (gray_image < 204))
    lightest = np.sum((gray_image >= 204) & (gray_image <= 255))

    # 밝기 구간별 데이터 시각화
    labels = ['Darkest Tones', 'Dark Tones', 'Mid Tones', 'Light Tones', 'Lightest Tones']
    values = [darkest, dark, mid, light, lightest]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='gray')
    plt.title(title)
    plt.xlabel('Brightness Range')
    plt.ylabel('Number of Pixels')
    plt.show()

def recommend_composition(original_image, brightness_adjusted):
    # 원본 이미지의 밝기 및 대비 분석
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray_image)[0]
    contrast = gray_image.std()  # 대비는 표준 편차로 간주

    # 색상 통계
    mean_color = cv2.mean(original_image)[:3]  # RGB 평균 값
    color_stddev = np.std(original_image, axis=(0, 1))  # 색상 표준 편차

    # 분석 결과 출력
    print(f"Mean Brightness: {mean_brightness:.2f}")
    print(f"Contrast: {contrast:.2f}")
    print(f"Mean Color (B, G, R): {mean_color}")
    print(f"Color Standard Deviation: {color_stddev}")

    # 동적 추천 밝기 및 대비 계산
    recommended_brightness = calculate_recommended_brightness(mean_brightness)
    recommended_contrast = calculate_recommended_contrast(contrast)

    # 구성 추천 기준
    lighting_advice = generate_lighting_advice(mean_brightness)
    contrast_advice = generate_contrast_advice(contrast)

    # 분석 기반의 상세 추천
    recommendations = (
        f"Based on the analysis:\n"
        f"- Mean Brightness: {mean_brightness:.2f} suggests: {lighting_advice}\n"
        f"- Contrast: {contrast:.2f} suggests: {contrast_advice}\n"
        f"- Mean Color Values (B, G, R): {mean_color}\n"
        f"- Consider adjusting the colors based on their standard deviation: {color_stddev}. "
        f"High standard deviation might indicate a vibrant scene, while low values could suggest a more uniform color distribution."
    )

    recommendations += (
        f"\n\nFor better results, aim for a brightness of around {recommended_brightness:.2f} and "
        f"a contrast of around {recommended_contrast:.2f}."
    )

    print("Recommendations for Better Composition:")
    print(recommendations)

    # 추천에 따라 조정 적용
    adjusted_image = apply_recommendations(original_image, recommended_brightness, recommended_contrast)

    # 원본 및 조정된 이미지 비교
    compare_images(original_image, adjusted_image)

def calculate_recommended_brightness(mean_brightness):
    # 평균 밝기를 기반으로 동적 추천 밝기 계산
    if mean_brightness < 100:
        return 110  # 낮은 밝기 증가
    elif mean_brightness >= 100 and mean_brightness <= 130:
        return mean_brightness + 5  # 중간 밝기 약간 증가
    else:
        return mean_brightness - 20

def calculate_recommended_contrast(contrast):
    # 현재 대비를 기반으로 동적 추천 대비 계산
    if contrast < 50:
        return 70  # 낮은 대비 증가
    else:
        return contrast + 15  # 적절한 대비에서 약간 증가

def generate_lighting_advice(mean_brightness):
    # 평균 밝기에 따른 조명 조언 생성
    if mean_brightness < 100:
        return "Increase the lighting. Consider using a flash or additional light sources."
    elif mean_brightness >= 100 and mean_brightness < 130:
        return "Lighting is adequate, but a slight increase might enhance the image."
    else:
        return "Lighting is optimal. You may maintain current conditions."

def generate_contrast_advice(contrast):
    # 대비에 따른 조언 생성
    if contrast < 50:
        return "Increase contrast by adjusting the camera settings or post-processing."
    else:
        return "Contrast levels are satisfactory. Good for capturing details."

def apply_recommendations(original_image, recommended_brightness, recommended_contrast):
    # 이미지 복사
    adjusted_image = original_image.copy()

    # LAB 색 공간으로 변환하여 대비 조정
    lab = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 적용 (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # LAB 색 공간에서 다시 병합
    adjusted_image = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_LAB2BGR)

    # HSV 색 공간으로 변환하여 하늘 영역 마스킹
    hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])   # 하늘색 범위 설정
    upper_blue = np.array([140, 255, 255])
    sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 하늘 영역 조정
    sky_region = cv2.bitwise_and(adjusted_image, adjusted_image, mask=sky_mask)
    sky_region = cv2.convertScaleAbs(sky_region, alpha=1.05, beta=5)  # 하늘 색상 강도 및 대비 조정

    # 건물 및 다른 요소 마스킹
    building_mask = cv2.bitwise_not(sky_mask)
    building_region = cv2.bitwise_and(adjusted_image, adjusted_image, mask=building_mask)
    building_region = cv2.convertScaleAbs(building_region, alpha=1.05, beta=0)  # 건물의 색상 및 대비 조정

    # 최종 이미지 결합
    final_image = cv2.add(sky_region, building_region)

    # 색상 균형 조정
    final_image = cv2.convertScaleAbs(final_image, alpha=1.0, beta=0)

    return final_image

def local_brightness_correction(image):
    # 이미지의 LAB 색 공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 밝기 조정
    l = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 밝기에 따른 영역 마스크 생성
    bright_mask = (l > 180).astype(np.uint8)  # 밝은 영역 (예: 건물)
    dark_mask = (l < 100).astype(np.uint8)    # 어두운 영역 (예: 하늘)

    # 밝기 조정
    if np.any(bright_mask):
        l[bright_mask == 1] = np.clip(l[bright_mask == 1] - 20, 0, 255)  # 밝은 영역에서 밝기 감소
    if np.any(dark_mask):
        l[dark_mask == 1] = np.clip(l[dark_mask == 1] + 30, 0, 255)      # 어두운 영역에서 밝기 증가

    # 채널 병합 후 BGR로 변환
    adjusted_lab = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    return adjusted_image

def compare_images(original_image, adjusted_image):
    # 원본 및 조정된 이미지를 나란히 표시
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
    plt.title('Adjusted Image')

    plt.show()

# 사용 예시
analyze_image('/content/Building.jpg')  # 이미지를 불러올 경로를 지정하세요