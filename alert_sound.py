import threading
import winsound as sd
import config as cfg

# detect 결과에 따른 사운드 출력을 위한 클래스
class AlertSound:
    def __init__(self):
        self.src_wrn = cfg.sound['warn']
        self.src_pss  = cfg.sound['good']
        self.src_inv = cfg.sound['invalid']
    
    def run(self, src):
        sd.PlaySound(src, sd.SND_FILENAME)

    def warn(self): # 경고음
        thr = threading.Thread(target=self.run, args=(self.src_wrn,))
        thr.start()

    def ok(self): # 정상
        thr = threading.Thread(target=self.run, args=(self.src_pss,))
        thr.start()

    def invalid(self): # 비정상
        thr = threading.Thread(target=self.run, args=(self.src_inv,))
        thr.start()
