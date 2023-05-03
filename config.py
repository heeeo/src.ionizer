
model = {
    "yolo-weights": "./Yolov5_StrongSORT_OSNet/yolov5/runs/train/20230424.phc/best.pt"
}

app = {
    "source": "./test_video/230418_1515_1540_Trim.mp4",
    "device": '0', # gpu device for run() in track_custom.py
    "imgsz": (1280, 1280)
}

mariadb = {
    'on': False
}

process = {
    'save_vid': True,
    'conf_thres': 0.2, 
    'stt_thres': 40,
    'end_thres': 20,
    'ion_cont_thres': 10,
    'led_conf_thres': 0.2,
    'volt_conf_thres': 0.2,
    'led_cnt_thres': 5,
    'volt_cnt_thres': 3,
    
}

image = {
    'i_w': 1920,
    'i_h': 1080,
    'x_min': 1130, # 작업영역 왼쪽 라인
    'x_max': 1290  # 작업영역 오른쪽 라인
}

border_color = {
    'ready': "#718093", 
    'start': "#00a8ff", 
    'good': "#4cd137", 
    'complete': "#27ae60", 
    'warn': "#e84118"
}

sound = {
    'warn': './sound/mixkit-classic-short-alarm-993.wav',
    'good': './sound/mixkit-correct-answer-tone-2870.wav',
    'invalid': './sound/questioning-vocal-tone-huh-92720.wav'
}