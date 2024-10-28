import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class ImageEnhancementAgent:
    def __init__(self):
        self.brightness_range = (0.8, 1.2)  # Factor range for brightness adjustment
        self.contrast_range = (0.5, 1.5)  # Factor range for contrast adjustment
        self.learning_rate = 0.1
        self.q_table = {}  # Simple Q-table for Q-learning
        self.last_action = None
        self.last_state = None

    def choose_action(self, explore=True):
        if explore or random.random() < 0.5:  # 50% chance to explore
            return (random.uniform(*self.brightness_range), random.uniform(*self.contrast_range))
        else:
            best_action = max(self.q_table.get(self.last_state, {}), key=self.q_table[self.last_state].get)
            return best_action

    def update_q_table(self, state, action, reward):
        # Update Q-table based on the action taken and the reward received
        if state not in self.q_table:
            self.q_table[state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        # Q-learning update rule
        self.q_table[state][action] += self.learning_rate * (reward - self.q_table[state][action])
        print(f"Updated Q-table for state {state}: {self.q_table[state]}")

    def evaluate_image(self, image):
        # A simple evaluation metric for image quality: mean brightness
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray_image)[0]

def analyze_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # Initialize the RL agent
    agent = ImageEnhancementAgent()

    # Initial state evaluation
    initial_quality = agent.evaluate_image(image)

    # RL loop to enhance the image
    best_action = None
    best_quality = initial_quality
    quality_history = []

    for _ in range(100):  # Increase iterations to give the agent more opportunities to learn
        action = agent.choose_action()
        brightness_factor, contrast_factor = action

        # Adjust brightness and contrast
        brightness_adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=(brightness_factor - 1) * 100)

        # Evaluate the enhanced image
        enhanced_quality = agent.evaluate_image(brightness_adjusted)

        # Calculate reward based on improvement
        reward = enhanced_quality - initial_quality
        agent.update_q_table(initial_quality, action, reward)

        # Log the action and reward
        print(f"Action taken: {action}, Reward: {reward}, Enhanced Quality: {enhanced_quality}")

        # Update best action if the current quality is better
        if enhanced_quality > best_quality:
            best_quality = enhanced_quality
            best_action = action

        # Update initial quality for the next iteration
        initial_quality = enhanced_quality

        # Store the quality history
        quality_history.append(best_quality)

    # Final adjustment using the best action found
    final_brightness, final_contrast = best_action  # Use the best action found
    final_image = cv2.convertScaleAbs(image, alpha=final_contrast, beta=(final_brightness - 1) * 100)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    sharpened_image = cv2.filter2D(final_image, -1, kernel)

    # Edge detection
    edges = cv2.Canny(sharpened_image, 100, 200)

    # Result visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Final Enhanced Image')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened Image')

    plt.subplot(2, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection Image')

    plt.tight_layout()
    plt.show()

    # Plot quality improvements over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(quality_history, label='Best Quality over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Quality')
    plt.title('Quality Improvement Over Time')
    plt.legend()
    plt.show()

    # Image histograms
    plot_image_histogram(image, 'Original Image Histogram')
    plot_image_histogram(final_image, 'Final Enhanced Image Histogram')

    # Composition recommendation
    recommend_composition(image, final_image)

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

    # Calculate dynamic recommended brightness and contrast
    recommended_brightness = calculate_recommended_brightness(mean_brightness)
    recommended_contrast = calculate_recommended_contrast(contrast)

    # Composition recommendation criteria based on the original image
    lighting_advice = generate_lighting_advice(mean_brightness)
    contrast_advice = generate_contrast_advice(contrast)

    # Detailed recommendation based on analysis
    recommendations = (
        f"Based on the analysis:\n"
        f"- Mean Brightness: {mean_brightness:.2f} suggests: {lighting_advice}\n"
        f"- Contrast: {contrast:.2f} suggests: {contrast_advice}\n"
        f"- Mean Color Values (B, G, R): {mean_color}\n"
        f"- Consider adjusting the colors based on their standard deviation: {color_stddev}. "
        f"High standard deviation might indicate a vibrant scene, while low values could suggest a more uniform color distribution."
    )

    # Adding recommendations for optimal settings
    recommendations += (
        f"\n\nFor better results, aim for a brightness of around {recommended_brightness:.2f} and "
        f"a contrast of around {recommended_contrast:.2f}."
    )

    print("Recommendations for Better Composition:")
    print(recommendations)

    # Apply adjustments based on recommendations and show results
    adjusted_image = apply_recommendations(original_image, recommended_brightness, recommended_contrast)

    # Show original and adjusted images side by side
    compare_images(original_image, adjusted_image)

def calculate_recommended_brightness(mean_brightness):
    # Dynamic calculation of recommended brightness based on mean brightness
    if mean_brightness < 100:
        return 110  # Increase for low brightness
    elif mean_brightness >= 100 and mean_brightness <= 130:
        return mean_brightness + 5  # Minor increase for moderate brightness
    else:
        return mean_brightness - 20  # Reduce for high brightness

def calculate_recommended_contrast(contrast):
    # Dynamic calculation of recommended contrast based on current contrast
    if contrast < 50:
        return 70  # Increase for low contrast
    else:
        return contrast + 15  # Moderate increase for adequate contrast

def generate_lighting_advice(mean_brightness):
    if mean_brightness < 100:
        return "Increase the lighting. Consider using a flash or additional light sources."
    elif mean_brightness >= 100 and mean_brightness < 130:
        return "Lighting is adequate, but a slight increase might enhance the image."
    else:
        return "Lighting is optimal. You may maintain current conditions."

def generate_contrast_advice(contrast):
    if contrast < 50:
        return "Increase contrast by adjusting the camera settings or post-processing."
    else:
        return "Contrast levels are satisfactory. Good for capturing details."

# def apply_recommendations(original_image, recommended_brightness, recommended_contrast):
#     # 이미지 복사
#     adjusted_image = original_image.copy()

#     # LAB 색 공간으로 변환
#     lab = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     # CLAHE 적용
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     # 밝기 및 대비 조정
#     adjusted_image = cv2.merge((l, a, b))
#     adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_LAB2BGR)
#     adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=1.1, beta=-15)  # 밝기 감소

#     # 색상 보정
#     b, g, r = cv2.split(adjusted_image)
#     b = cv2.addWeighted(b, 1.05, 0, 0, 0)  # 파란색 강조
#     r = cv2.addWeighted(r, 0.9, 0, 0, 0)  # 빨간색 약간 감소
#     adjusted_image = cv2.merge((b, g, r))

#     # 검정색 건물 부분의 디테일 보존
#     black_building_mask = cv2.inRange(l, 0, 100)  # 검정색 건물 마스크
#     adjusted_image[black_building_mask > 0] = cv2.convertScaleAbs(adjusted_image[black_building_mask > 0], alpha=1.2, beta=15)  # 밝기 조정

#     # 나무 부분의 디테일 보존을 위한 추가 조정
#     mask_trees = cv2.inRange(g, 50, 150)  # 나무 색상을 기준으로 마스크 생성
#     adjusted_image[mask_trees > 0] = cv2.convertScaleAbs(adjusted_image[mask_trees > 0], alpha=1.1, beta=10)  # 나무 부분 밝기 조정

#     # 샤프닝 필터 적용
#     sharpened_image = cv2.filter2D(adjusted_image, -1, np.array([[0, -0.3, 0], [-0.3, 2, -0.3], [0, -0.3, 0]]))
#     final_image = cv2.addWeighted(adjusted_image, 0.7, sharpened_image, 0.3, 0)

#     return final_image

def apply_recommendations(original_image, recommended_brightness, recommended_contrast):
    # 이미지 복사
    adjusted_image = original_image.copy()

    # LAB 색 공간으로 변환
    lab = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 밝기 및 대비 조정
    adjusted_image = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_LAB2BGR)
    adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=1.05, beta=-10)  # 밝기 감소

    # 구름 영역 조정
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    cloud_mask = (gray_image < 100).astype(np.uint8)  # 어두운 구름 영역
    adjusted_image[cloud_mask > 0] = cv2.convertScaleAbs(adjusted_image[cloud_mask > 0], alpha=0.9, beta=0)  # 구름 밝기 조정

    # 색상 보정
    b, g, r = cv2.split(adjusted_image)
    b = cv2.addWeighted(b, 1.05, 0, 0, 0)  # 파란색 강조
    r = cv2.addWeighted(r, 0.95, 0, 0, 0)  # 빨간색 약간 감소
    adjusted_image = cv2.merge((b, g, r))

    # 검정색 건물 부분의 디테일 보존
    black_building_mask = cv2.inRange(l, 0, 100)  # 검정색 건물 마스크
    adjusted_image[black_building_mask > 0] = cv2.convertScaleAbs(adjusted_image[black_building_mask > 0], alpha=1.1, beta=10)  # 밝기 조정

    return adjusted_image

def local_brightness_correction(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance contrast and brightness based on area conditions
    l = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Create masks for different areas based on brightness
    bright_mask = (l > 180).astype(np.uint8)  # Bright areas (e.g., buildings)
    dark_mask = (l < 100).astype(np.uint8)    # Dark areas (e.g., sky)

    # Adjust brightness for bright and dark areas
    if np.any(bright_mask):
        l[bright_mask == 1] = np.clip(l[bright_mask == 1] - 20, 0, 255)  # Decrease brightness in bright areas (건물)
    if np.any(dark_mask):
        l[dark_mask == 1] = np.clip(l[dark_mask == 1] + 30, 0, 255)      # Increase brightness in dark areas (하늘)

    # Merge channels back and convert to BGR
    adjusted_lab = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    return adjusted_image

def compare_images(original_image, adjusted_image):
    # Display original and adjusted images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))
    plt.title('Adjusted Image')

    plt.show()

# Example usage
analyze_image('<Put image in here>.jpg')  # Replace with your image path
