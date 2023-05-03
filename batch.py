import math
import os
from pathlib import Path
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
import tkinter.scrolledtext as st
from apscheduler.schedulers.background import BackgroundScheduler
import warnings
import mariadb
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from Yolov5_StrongSORT_OSNet.track_custom import run as run_track
from alert_sound import AlertSound
import config as cfg


class DetectObject:
    def __init__(self, det):  # {'cls': _cls, 'id': _id, 'bboxes': bboxes}
        # self.last_fidx = frame_idx
        self.conf = det['conf']
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
                
class DB:
    def __init__(self):
        try:
            print('connecting ...')
            dbconf = cfg.mariadb
            self.conn = mariadb.connect(
                user= dbconf['user'],
                password=dbconf['password'],
                host=dbconf['host'],
                port=dbconf['port'],
                database=dbconf['database'],
            )
            self.cur = self.conn.cursor()
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB platfomr: {e}")

    def insert2io(self, result):
        try:
            self.cur.execute("INSERT INTO FCT_IO (s_date, e_date, result) VALUES (?, ?, ?)", (result[0], result[1], result[2]))
            self.conn.commit()
            print(f"Last Inserted ID: {self.cur.lastrowid}")
        except mariadb.Error as e:
            self.conn.rollback()
            print(f"Error: {e}")
    
    def select_last_row(self):
        self.cur.execute("SELECT * from FCT_IO ORDER BY SEQ DESC LIMIT 1")
        row = self.cur.fetchone()
        return row

    def close(self):
        print('closing db connect ...')
        self.cur.close()
        self.conn.close()

class App:
    def __init__(self, video_source):
        # Set App Visual Elements
        self.color = cfg.border_color
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
        
        self.border = tk.Frame(self.frame, background=self.color['ready'])
        self.border.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        self.cam_view = tk.Label(self.frame) # , borderwidth=1, relief="solid"
        self.cam_view.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.log_area = st.ScrolledText(self.frame, fg='white', bg='black')
        self.log_area.grid(row=1, column=0, sticky='nsew')
        self.log_area.insert(tk.INSERT, "log will be displayed here.\n")
        self.log_area.configure(state='disabled')

        # Set Process Args
        self.stop_track = False
        self.process_idx = 0
        self.curr_objects = {}
        self.curr_process = self.init_curr_process()
        self.log_file = None
        self.sched = None
        self.pause = False

        # self.frame.pack(expand=True)
        self.init_process_log()
        self.init_log_scheduler()
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
        # self.sched.shutdown()
        if self.log_file is not None:
            self.log_file.close()
        self.window.after(500, self.window.destroy)

    def init_process_log(self):
        today = datetime.now().strftime('%Y-%m-%d')

        Path("./process_log").mkdir(parents=True, exist_ok=True)
        log_path = f'./process_log/{today}.log.csv'
        log_exists = os.path.isfile(log_path)
        file_mode  = 'a+' if log_exists else 'w'

        # log_file = open(log_path, file_mode)

        if not log_exists:
            if self.log_file is not None:
                self.log_file.close()
            self.log_file = open(log_path, file_mode)
            self.log_file.write('index, 공정시작시간, 공정종료시간, result, 누락공정\n')
            self.log_file.flush()
            self.process_idx = 0
            
            nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f'\n[{nowtime}] Log Recharge Scheduler: Recharged log file.\n')

        if self.log_file is None:
            self.log_file = open(log_path, file_mode)


    def recharge_log_file(self):
        self.init_process_log()
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f'\n[{nowtime}] Log Recharge Scheduler: check log file.\n')

    def sche_test_print(self):
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f'\n[{nowtime}] Log Recharge Scheduler: Test Print ...\n')


    def init_log_scheduler(self):
        self.sched = BackgroundScheduler()
        self.sched.start()

        self.sched.add_job(self.recharge_log_file, 'cron', second='00', id='recharge_log')  # every midnight  // hour='00', minute='00',
        return

    def add_process_log(self, conc):
        # file format (.csv)
        # index, 공정시작시간, 공정종료시간, result, 누락공정
        # 0    , 2023-03-03 13:29:05.959, 2023-03-03 13:29:05.959, 0, led
        line = f"{self.process_idx}, {self.curr_process['stt_time']}, {self.curr_process['end_time']}, {conc['result']}, {conc['missed']}\n"
        self.log_file.write(line)
        self.log_file.flush()

        self.process_idx += 1
        pass
    
    def run_track(self, source, weight, device):
        run_track(
                  self, 
                  source=source, 
                  yolo_weights=weight,
                  imgsz=(1280,1280), 
                  conf_thres=cfg.process['conf_thres'],
                  save_vid=cfg.process['save_vid'],
                  device=device, 
                  show_vid=True, 
                  nosave=True
        )

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
            'end_cnt': 0, 
            'stt_time': '',
            'end_time': '',
            'det_area_ids': set(), # 공정시작 후 1영역에서 발견되는 모든 track id
            'led_chk_cnt': 0,
            'volt_chk_cnt': 0,
            'led_check': False,
            'volt_check': False
        }
    
    def update_end_cnt(self, cls_set):
        if self.curr_process['stat'] == 0:
            return

        size = len(cls_set)
        if size == 0:
            self.curr_process['end_cnt'] += 1
            self.insert_log(f"end_cnt: {self.curr_process['end_cnt']}")
        else:
            self.curr_process['end_cnt'] = 0

        if cfg.process['end_thres'] <= self.curr_process['end_cnt']:
            self.finish_process()


    def select_last_row(self):
        db = DB()
        row = db.select_last_row()
        db.close()
        return row

    def insert_result_2_db(self, result):
        db = DB()
        db.insert2io(result)
        db.close()


    def check_user_select_error_type(self):  ### 🔴
        # fetch DB to check last row's error column is not null
        row = self.select_last_row()
        if row is None:
            return False

        error_val = row[4]
        print(f"\n****************************** error_val : {error_val} \n")
        if error_val is not None:
            return True
        return False

    def check_user_select_error_type_loop(self):  # run for a thread    ### 🔴
        time.sleep(2)

        while True:
            if self.check_user_select_error_type():
                self.pause = False
                self.insert_log("\n************* process Unpaused **************\n")
                return
            time.sleep(0.5)
    

    def finish_process(self):
        self.insert_log("********************** Process End !!! ***********************")
        self.curr_process['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        conc = self.conclude()
        result = conc['result']
        if result == 0: # 정상
            self.border.config(background=self.color['complete'])
        else: # 누락
            self.border.config(background=self.color['warn'])
        self.add_process_log(conc)

        stt_time = self.curr_process['stt_time']
        end_time = self.curr_process['end_time']

        self.curr_process = self.init_curr_process()

        if result == 1: ### 🔴
            self.pause = True
            self.insert_log("\n************* process paused **************\n")
            threading.Thread(target=self.check_user_select_error_type_loop, args=()).start()

        # DB 입력
        self.insert_result_2_db([stt_time, end_time, conc['result']])
        self.insert_log(f"*** DB Insert : result({result})")
    
        time.sleep(2)
        self.border.config(background=self.color['ready'])


    def conclude(self):
        conclude = {}
        missed = ''

        led_check = self.curr_process['led_check']
        volt_check = self.curr_process['volt_check']

        if not led_check: missed = missed + 'led '
        if not volt_check: missed = missed + 'volt'

        conclude['result'] = 0 if led_check and volt_check else 1
        conclude['missed'] = missed

        result = '정상' if conclude['result'] == 0 else '누락'
        self.insert_log("\n******* 결과 *******")
        self.insert_log(f" - 정/불 : '{result}'")
        if conclude['result'] == 1:
            self.insert_log(f" - 누락 : {missed}")

        return conclude


    def update(self, det_data):
        if self.pause: ### 🔴
            return
        
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
                        self.insert_log(
                            f"det_area_ids : {det_area_ids}, led_cnt: {self.curr_process['led_chk_cnt']}, volt_cnt: {self.curr_process['volt_chk_cnt']}, led_check: {self.curr_process['led_check']}, volt_check: {self.curr_process['volt_check']}"
                        )
                    
                    # 공정종료 이벤트
                    if (self.curr_process['stat'] == 1) and (self.det_area_x_max < xc ):  # or xc < self.det_area_x_min
                        det_area_ids = self.curr_process['det_area_ids']
                        if ion_obj.id in det_area_ids:
                            self.finish_process()

                # 공정시작 이벤트
                if (self.curr_process['stat'] == 0) and (self.curr_process['stt_cnt'] >= cfg.process['stt_thres']):
                    self.insert_log("********************** Process Start !!! ***********************")
                    self.curr_process['stat'] = 1
                    self.curr_process['stt_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self.border.config(background=self.color['start'])
                
            elif cls == 1 or cls == 2: # 1: led_check or 2: volt_check
                for chk_det in cls_det_list:
                    chk_obj = DetectObject(chk_det)
                    center  = chk_obj.center
                    xc      = center[0]
                    yc      = center[1]
                    # print(f"id: {chk_obj.id}, class: {cls}, xc: {xc}")

                    # 공정시작 trigger 이벤트
                    if (self.curr_process['stat'] == 0) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        stt_cnt = self.curr_process['stt_cnt']
                        self.curr_process['stt_cnt'] = stt_cnt + 1
                        self.insert_log(f"stt_cnt : {self.curr_process['stt_cnt']}")
                    # 공정중 det_area_id set update 
                    elif (self.curr_process['stat'] == 1) and (self.det_area_x_min <= xc and xc <= self.det_area_x_max):
                        det_area_ids = self.curr_process['det_area_ids']
                        det_area_ids.add(chk_obj.id)

                        if cls == 1:
                            if chk_obj.conf >= cfg.process['led_conf_thres']:
                                self.curr_process['led_chk_cnt'] += 1
                            if (self.curr_process['led_check'] == False) and (self.curr_process['led_chk_cnt'] >= cfg.process['led_cnt_thres']):
                                self.curr_process['led_check'] = True
                                self.insert_log("************** LED Checked ....... !!!")
                        elif cls == 2:
                            if chk_obj.conf >= cfg.process['volt_conf_thres']:
                                self.curr_process['volt_chk_cnt'] += 1
                            if (self.curr_process['volt_check'] == False) and (self.curr_process['volt_chk_cnt'] >= cfg.process['volt_cnt_thres']):
                                self.curr_process['volt_check'] = True
                                self.insert_log("************** Volt Checked ....... !!!")

                        self.insert_log(
                            f"det_area_ids : {det_area_ids}, led_cnt: {self.curr_process['led_chk_cnt']}, volt_cnt: {self.curr_process['volt_chk_cnt']}, led_check: {self.curr_process['led_check']}, volt_check: {self.curr_process['volt_check']}"
                        )

                    # 공정종료 이벤트
                    if (self.curr_process['stat'] == 1) and (self.det_area_x_max < xc ):  # or xc < self.det_area_x_min
                        det_area_ids = self.curr_process['det_area_ids']
                        if chk_obj.id in det_area_ids:
                            self.finish_process()

                # 공정시작 이벤트
                if (self.curr_process['stat'] == 0) and (self.curr_process['stt_cnt'] >= cfg.process['stt_thres']):
                    self.insert_log("********************** Process Start !!! ***********************")
                    self.curr_process['stat'] = 1
                    self.curr_process['stt_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self.border.config(background=self.color['start'])

        # print("******* cls_cnts : ", cls_cnts)

        # self.update_end_cnt(cls_set)

        return


def main():
    App(video_source=cfg.app["source"])
    

if __name__ == "__main__":
    main()