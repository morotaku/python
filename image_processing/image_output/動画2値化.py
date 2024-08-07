import cv2

def binarize_video(input_file, output_file):
    # 動画ファイルを読み込む
    video = cv2.VideoCapture(input_file)
    
    # 出力する動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), 0)
    
    # 動画フレームごとに処理
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
    
        # 2値化処理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
        # 出力動画にフレームを追加
        output.write(binary)
    
    # リソースの解放
    video.release()
    output.release()

# 使用例
input_file = 'AuJR-2_w=35um_t=40um_200ns_x9.2_5mW_HWP185.75_1_001.avi'
output_file = 'b.mp4'
binarize_video(input_file, output_file)