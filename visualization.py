import cv2
import os
import random

# 参数设置
image_folder = "./datasets/dancetrack/dancetrack0004/img1"
tracking_file = "./datasets/dancetrack/dancetrack0004.txt"
output_video = "output_tracking.mp4"
save_video = True

# 要截取并保存的帧编号（从 1 开始计数），含标注框
extract_frames_nums = [380, 381]
extract_out_dir = "./extracted_frames"
os.makedirs(extract_out_dir, exist_ok=True)

# 加载追踪数据
def load_tracking_data(file_path):
    tracking = {}
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            frame_id = int(items[0])
            target_id = int(items[1])
            x, y = float(items[2]), float(items[3])
            w, h = float(items[4]), float(items[5])
            tracking.setdefault(frame_id, []).append((target_id, x, y, w, h))
    return tracking

# 获取排序后的图像帧路径
def get_sorted_frame_paths(folder):
    frames = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    return [os.path.join(folder, f) for f in frames]

# 随机颜色生成（BGR），同 ID 同色
def get_color(target_id, color_dict):
    if target_id not in color_dict:
        random.seed(target_id)
        color_dict[target_id] = tuple(random.randint(0, 255) for _ in range(3))
    return color_dict[target_id]

# 在单张帧图上添加所有追踪框
def draw_tracking_on_frame(img, frame_idx, tracking_data, id_colors):
    # frame_idx 从 1 开始
    for tid, x, y, w, h in tracking_data.get(frame_idx, []):
        color = get_color(tid, id_colors)
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"ID:{tid}", (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# 提取并保存含标注框的帧
def extract_and_save_tracked_frames(frame_paths, frame_numbers, tracking_data, out_dir):
    id_colors = {}
    for num in frame_numbers:
        idx = num - 1
        if idx < 0 or idx >= len(frame_paths):
            print(f"帧 {num} 超出范围，跳过")
            continue
        img = cv2.imread(frame_paths[idx])
        if img is None:
            print(f"无法读取第 {num} 帧：{frame_paths[idx]}")
            continue
        # 绘制标注框
        img_tracked = draw_tracking_on_frame(img, num, tracking_data, id_colors)
        save_path = os.path.join(out_dir, f"frame_{num}_tracked.png")
        cv2.imwrite(save_path, img_tracked)
        print(f"已保存含标注框的帧 {num} 到 {save_path}")

# 视频可视化函数
def visualize(tracking_data, frame_paths):
    writer = None
    paused = False
    current_idx = 0
    total_frames = len(frame_paths)
    id_colors = {}

    while True:
        if not paused:
            frame = cv2.imread(frame_paths[current_idx])
            frame = draw_tracking_on_frame(frame, current_idx + 1, tracking_data, id_colors)

            # 顶部状态栏
            info_height = 40
            frame = cv2.copyMakeBorder(frame, info_height, 0, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(50, 50, 50))
            cv2.putText(frame, f"Frame: {current_idx + 1}/{total_frames}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Targets: {len(tracking_data.get(current_idx + 1, []))}", (220, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)
            cv2.putText(frame, "[SPACE] Pause/Play  [back/forward] Jump  [Q] Quit", (450, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # 显示及写入视频
            cv2.imshow("Tracking Viewer", frame)
            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_video, fourcc, 20, (w, h))
                writer.write(frame)

            current_idx = (current_idx + 1) % total_frames

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 81:  # ←
            current_idx = max(0, current_idx - 10)
        elif key == 83:  # →
            current_idx = min(total_frames - 1, current_idx + 10)

    if writer:
        writer.release()
    cv2.destroyAllWindows()

# 主流程
if __name__ == "__main__":
    tracking_data = load_tracking_data(tracking_file)
    frame_paths = get_sorted_frame_paths(image_folder)

    # 步骤1：提取并保存含标注框的第 585 和 586 帧
    extract_and_save_tracked_frames(frame_paths, extract_frames_nums, tracking_data, extract_out_dir)

    # 步骤2：启动视频可视化
    visualize(tracking_data, frame_paths)
