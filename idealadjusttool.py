import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    # Grayscale로 변환
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
    # Analyze brightness and contrast of the original image
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray_image)[0]
    contrast = gray_image.std()  # Contrast considered as standard deviation

    # Color statistics
    mean_color = cv2.mean(original_image)[:3]  # RGB mean values
    color_stddev = np.std(original_image, axis=(0, 1))  # Color standard deviation for each channel

    # Print analysis results
    print(f"Mean Brightness: {mean_brightness:.2f}")
    print(f"Contrast: {contrast:.2f}")
    print(f"Mean Color (B, G, R): {mean_color}")
    print(f"Color Standard Deviation: {color_stddev}")

    # Composition recommendation criteria based on the original image
    if mean_brightness < 100:
        lighting_advice = "Increase the lighting. Consider using a flash or additional light sources."
    elif mean_brightness >= 100 and contrast >= 50:
        lighting_advice = "Lighting is optimal. You may maintain current conditions."
    else:
        lighting_advice = "Adjust the lighting conditions for better shooting results."

    if contrast < 50:
        contrast_advice = "Increase contrast by adjusting the camera settings or post-processing."
    else:
        contrast_advice = "Contrast levels are satisfactory. Good for capturing details."

    # Detailed recommendation based on analysis
    recommendations = (
        f"Based on the analysis:\n"
        f"- Mean Brightness: {mean_brightness:.2f} suggests: {lighting_advice}\n"
        f"- Contrast: {contrast:.2f} suggests: {contrast_advice}\n"
        f"- Mean Color Values (B, G, R): {mean_color}\n"
        f"- Consider adjusting the colors based on their standard deviation: {color_stddev}. "
        f"High standard deviation might indicate a vibrant scene, while low values could suggest a more uniform color distribution."
    )

    print("Recommendations for Better Composition:")
    print(recommendations)

    # Apply adjustments based on recommendations and show results
    adjusted_image = apply_recommendations(original_image, mean_brightness, contrast)

    # Show original and adjusted images side by side
    compare_images(original_image, adjusted_image)

def apply_recommendations(original_image, mean_brightness, contrast):
    # Adjust brightness and contrast naturally while preserving highlights
    adjusted_image = original_image.copy()

    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for natural contrast adjustment without brightening highlights
    lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # Apply CLAHE to the L-channel

    lab_adjusted = cv2.merge((cl, a, b))
    adjusted_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

    # Apply mild sharpening to enhance details without affecting highlights
    kernel = np.array([[0, -0.3, 0], [-0.3, 2, -0.3], [0, -0.3, 0]])
    adjusted_image = cv2.filter2D(adjusted_image, -1, kernel)

    return adjusted_image

def compare_images(original_image, adjusted_image):
    # Display original and adjusted images side by side
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
    plt.title('Adjusted Image Based on Recommendations')

    plt.tight_layout()
    plt.show()

# Example usage
analyze_image('/content/Building.jpg')