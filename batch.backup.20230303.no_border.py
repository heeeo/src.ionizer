import math
from pathlib import Path
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
import tkinter.scrolledtext as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from Yolov5_StrongSORT_OSNet.track_custom import run as run_track
from alert_sound import AlertSound
import config as cfg


class DetectObject:
    def __init__(self, det):  # {'cls': _cls, 'id': _id, 'bboxes': bboxes}
        # self.last_fidx = frame_idx
        self.cls = det['cls']
        self.id = det['id']
        self.xyxy = det['bboxes']
        self.center = self.get_center(self.xyxy)
    
    def __str__(self):
        return f"object = id: {self.id}, cls: {self.cls}, xyxy: {self.xyxy}"
    
    def get_center(self, xyxy):
        xl = xyxy[0]
        yl = xyxy[1]
        xr = xyxy[2]
        yr = xyxy[3]
        xc = xl + ((xr-xl)/2)
        yc = yl + ((yr-yl)/2)
        return (xc, yc)

    def update(self, frame_idx, bboxes):
        self.last_fidx = frame_idx
        self.xyxy = bboxes
        self.center = self.get_center(bboxes)


class Ionizer(DetectObject):
    def __init__(self, frame_idx, xyxy, id, cls):
        super().__init__(frame_idx, xyxy, id, cls)
        # self.lock  = threading.Lock()
        # self.working = False
            

class ProcState:
    def __init__(self):
        self.process_on = False
        self.process_complete = False
    
    def process_started(self, app, ionizer):
        self.process_on = True
        self.stt_time = time.time()
        app.insert_log(f'\n\n*** Process Started (working ionizer=[{ionizer.id}]) !\n\n')
        # print('\n\n*** Process Started !\n\n')

    def process_ended(self, app):
        self.process_on = False
        end_time = time.time()
        app.insert_log(f'\n\n*** Process Ends (in {(end_time-self.stt_time):.2f} s) ...\n\n')
        # print(f'\n\n*** Process Ends (in {(end_time-self.stt_time):.2f} s) ...\n\n')
                

class App:
    def __init__(self, video_source):
        self.alert = AlertSound()
        self.window = tk.Tk()
        self.window.title("Live Cam Viewer")
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        print('loading video to get metainfo ...')
        cap = cv2.VideoCapture(video_source)
        print('done.')
        self.img_width = cfg.image['i_w']
        self.img_height = cfg.image['i_h']
        self.det_area_x_min = cfg.image['x_min']
        self.det_area_x_max = cfg.image['x_max']
        self.vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        print('cam frame size :', self.vid_width,',', self.vid_height)
        print('screen size :', screen_width,',', screen_height)

        self.window.geometry(f"{screen_width}x{screen_height-30}")
        self.window.resizable(False, False)
        self.frame = tk.Frame(self.window)
        self.frame.grid(sticky='nswe')
        self.frame.grid_rowconfigure(0, weight=30)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        self.cam_view = tk.Label(self.frame) # , borderwidth=1, relief="solid"
        self.cam_view.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        self.log_area = st.ScrolledText(self.frame, fg='white', bg='black')
        self.log_area.grid(row=1, column=0, sticky='nsew')
        self.log_area.insert(tk.INSERT, "log will be displayed here.\n")
        self.log_area.configure(state='disabled')

        self.stop_track = False
        self.curr_objects = {}
        self.curr_process = self.init_curr_process()

        # self.frame.pack(expand=True)
        self.window.after(10, self.fit_camview_size)
        self.window.after(100, self.start_track, video_source, Path(cfg.model["yolo-weights"]), cfg.app["device"])

        self.window.protocol("WM_DELETE_WINDOW", self.on_win_close)
        self.window.mainloop()
        pass
    
    def fit_camview_size(self):
        self.window.update()
        width = self.cam_view.winfo_width()
        height = self.cam_view.winfo_height()
        print('current camview size :', width,',', height)

        img = Image.open('./dataset/test/images/1000_55_54.jpg')
        img_wid, img_hei = img.size

        rhei = height
        rwid = int(rhei * (img_wid/img_hei))

        self.vid_width = rwid
        self.vid_height = rhei
        print('adjust cam frame size :', self.vid_width,',', self.vid_height)
        # imr = img.resize((rwid, rhei), Image.ANTIALIAS)
        # tk_img = ImageTk.PhotoImage(imr)
        # self.cam_view.config(image=tk_img)
        # self.cam_view.image = tk_img
        return
    
    def on_win_close(self):
        self.stop_track = True
        print('waiting for stoping track ...')
        self.window.after(500, self.window.destroy)
    
    def run_track(self, source, weight, device):
        run_track(self, source=source, yolo_weights=weight, imgsz=(1920,1080), device=device, show_vid=True, nosave=True)

    def start_track(self, source, weight, device):
        threading.Thread(target=App.run_track, args=(self, source, weight, device)).start()

    def imshow(self, img):
        # draw 3 sections lines
        i_w = self.img_width
        i_h = self.img_height
        # print(f"****** i_w: {i_w}, i_h: {i_h}\n")
        x_l = self.det_area_x_min
        x_r = self.det_area_x_max
        cv2.line(img, (x_l, 0), (x_l, i_h), (255,0,0), thickness=4, lineType=cv2.LINE_AA)
        cv2.line(img, (x_r, 0), (x_r, i_h), (255,0,0), thickness=4, lineType=cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((self.vid_width, self.vid_height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image = img, master=self.window)
        self.cam_view.config(image=photo)
        self.cam_view.image = photo
    
    def insert_log(self, log):
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log = f"[{nowtime}] {log}"
        log_area = self.log_area
        log_area.configure(state='normal')
        log_area.insert(tk.END, log + '\n')
        log_area.see(tk.END)
        log_area.configure(state='disabled')
        print(log)
        return


    def init_curr_process(self):
        return {
            'stat': 0, # stat: 0(종료), 1(시작)
            'stt_cnt': 0, # 공정시작 전 1영역(detect area)에서 detect되는 ionizer 누적 frame 수 (start event trigger)
            'det_area_ids': set(), # 공정시작 후 1영역에서 발견되는 모든 track id
            'led_check': False,
            'volt_check': False
        }

    def conclude(self):
        conclude = {}
        missed = []

        led_check = self.curr_process['led_check']
        volt_check = self.curr_process['volt_check']

        if not led_check: missed.append('LED')
        if not volt_check: missed.append('VOLT')

        conclude['result'] = 0 if led_check and volt_check else 1
        conclude['missed'] = missed

        result = '정상' if conclude['result'] == 0 else '누락'
        self.insert_log("\n******* 결과 *******")
        self.insert_log(f" - 정/불 : '{result}'")
        if conclude['result'] == 1:
            self.insert_log(f" - 누락 : {missed}")

        return conclude


    def update(self, det_data):
        cls_set = det_data['cls_set']
        cls_cnts = {}
        ionizers = {}
        # print("******* det_data per frame : ", det_data)

        for cls in cls_set:
            cls_det_list = cls_set[cls]
            cls_cnts[cls] = len(cls_det_list)

            if cls == 0: # ionizer
                for ion_det in cls_det_list:
                    ion_obj = DetectObject(ion_det)
                    center  = ion_obj.center
                    xc      = center[0]
                    yc      = center[1]

                    # 공정시작 trigger 이벤트
                    if (self.curr_process['stat'] == 0) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        stt_cnt = self.curr_process['stt_cnt']
                        self.curr_process['stt_cnt'] = stt_cnt + 1
                        self.insert_log(f"stt_cnt : {self.curr_process['stt_cnt']}")
                    # 공정중 det_area_id set update
                    elif (self.curr_process['stat'] == 1) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        det_area_ids = self.curr_process['det_area_ids']
                        det_area_ids.add(ion_obj.id)
                        self.insert_log(f"det_area_ids : {det_area_ids}")
                    
                    # 공정종료 이벤트
                    if (self.curr_process['stat'] == 1) and (self.det_area_x_max <= xc):
                        det_area_ids = self.curr_process['det_area_ids']
                        if ion_obj.id in det_area_ids:
                            self.insert_log("********************** Process End !!! ***********************")
                            conc = self.conclude()
                            self.curr_process = self.init_curr_process()
                # 공정시작 이벤트
                if (self.curr_process['stat'] == 0) and (self.curr_process['stt_cnt'] >= cfg.process['stt_thres']):
                    self.insert_log("********************** Process Start !!! ***********************")
                    self.curr_process['stat'] = 1
                
            elif cls == 1 or cls == 2: # 1: led_check or 2: volt_check
                for chk_det in cls_det_list:
                    chk_obj = DetectObject(chk_det)
                    center  = chk_obj.center
                    xc      = center[0]
                    yc      = center[1]

                    # 공정시작 trigger 이벤트
                    if (self.curr_process['stat'] == 0) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        stt_cnt = self.curr_process['stt_cnt']
                        self.curr_process['stt_cnt'] = stt_cnt + 1
                        self.insert_log(f"stt_cnt : {self.curr_process['stt_cnt']}")
                    # 공정중 det_area_id set update 
                    elif (self.curr_process['stat'] == 1) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        det_area_ids = self.curr_process['det_area_ids']
                        det_area_ids.add(chk_obj.id)
                        self.insert_log(f"det_area_ids : {det_area_ids}")

                        if cls == 1:
                            if self.curr_process['led_check'] == False: 
                                self.insert_log("************** LED Checked ....... !!!")
                            self.curr_process['led_check'] = True
                        elif cls == 2:
                            if self.curr_process['volt_check'] == False:
                                self.insert_log("************** Volt Checked ....... !!!")
                            self.curr_process['volt_check'] = True
                    

        # print("******* cls_cnts : ", cls_cnts)

        return


def main():
    App(video_source=cfg.app["source"])
    

if __name__ == "__main__":
    main()