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
    def __init__(self, frame_idx, xyxy, id, cls):
        self.last_fidx = frame_idx
        self.xyxy = xyxy
        self.center = self.get_center(xyxy)
        self.id = id
        self.cls = cls
    
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
        self.volt_check_needed = False
        self.volt_check = False
        self.led_check = False
        self.port_lans = {}
    
    def renew(self, frame_idx, xyxy, id):
        self.last_fidx = frame_idx
        self.xyxy = xyxy
        self.center = self.get_center(xyxy)
        self.id = id

    def has_center_in_box(self, obejct:DetectObject):
        obj_center_x = obejct.center[0]
        obj_center_y = obejct.center[1]
        xl, yl, xr, yr = self.xyxy[0], self.xyxy[1], self.xyxy[2], self.xyxy[3]
        if (xl <= obj_center_x and obj_center_x <= xr) and (yl <= obj_center_y and obj_center_y <= yr):
            return True
        return False
    
    def show_portlans(self, app):
        app.insert_log(f"current portlan set of [ionizer id {self.id}] : {list(self.port_lans.keys())} ({len(self.port_lans)})")
        # print(f"current portlan set of [ionizer id {self.id}] : {list(self.port_lans.keys())} ({len(self.port_lans)})")


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
        self.state = ProcState()
        self.ionizers = {"all": {}, "working": None}
        # self.port_lans = {}

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

    def show_current_ionizers(self):
        all_ionizers = self.ionizers["all"]
        current_ionizer_ids = list(all_ionizers.keys())
        self.insert_log(f'current ionizers : {current_ionizer_ids}, ({len(current_ionizer_ids)})\n')

    def get_nearest_portlan(self, port_lan: DetectObject) -> Ionizer:
        # 이 port_lan을 포함하는 ionizer 모두 찾기
        including_ionizers = []
        ionizer :Ionizer
        for ion_id, ionizer in self.ionizers["all"].items():
            if ionizer.has_center_in_box(port_lan):
                including_ionizers.append(ionizer)
        if len(including_ionizers) == 0:
            return None
        if len(including_ionizers) == 1:
            return including_ionizers[0]
        # 각 ionizer와의 거리(중점to중점) 구하기
        dist_to_ionizers = []
        for ionizer in including_ionizers:
            dist = math.dist([ionizer.center[0], ionizer.center[1]], [port_lan.center[0], port_lan.center[1]])
            dist_to_ionizers.append(dist)
        # 가장 가까운 거리의 ionizer 구하기
        index_min = np.argmin(dist_to_ionizers)
        return including_ionizers[index_min]

    def is_successful(self, work_izr: Ionizer):
        led_checked = work_izr.led_check
        volt_checked = False
        if work_izr.volt_check_needed:
            volt_checked = work_izr.volt_check
        else:
            volt_checked = True
        return (led_checked and volt_checked)

    def update(self, frame_idx, bboxes, id, cls):
        id, cls = int(id), int(cls)

        if cls == 0: # ionizer
            a_ionizer = Ionizer(frame_idx, bboxes, id, cls)
            work_izr:Ionizer = self.ionizers["working"]
            all_ionizers = self.ionizers["all"]
            if id not in all_ionizers: 
                if (work_izr is not None) and (work_izr.id not in all_ionizers) and a_ionizer.has_center_in_box(work_izr):
                    # 공정작업 중인 ionizer가 가려져서 detect 안됐다가 다시 됐을때
                    # 중점이 공정작업 중인 ionizer에 들어가는지 체크 -> true면 해당 id 및 정보로 renew
                    prev_id = work_izr.id
                    work_izr.renew(frame_idx, bboxes, id)
                    self.insert_log(f"working ionizer Renewed : id [{prev_id}] -> [{id}]")
                # else:
                #    all_ionizers[id] = a_ionizer # add new ionizer
                #    self.insert_log(f"ionizer added : id [{id}]")
                #    self.show_current_ionizers()
                all_ionizers[id] = a_ionizer # add new ionizer
                self.insert_log(f"ionizer added : id [{id}]")
                self.show_current_ionizers()
            else:
                all_ionizers[id].update(frame_idx, bboxes)

        if cls == 1: # port_lan
            port_lan = DetectObject(frame_idx, bboxes, id, cls)
            work_izr:Ionizer = self.ionizers["working"]
            # 이 port_lan을 포함하면서 가장 근접한 ionizer를 구한다.
            ionizer: Ionizer = self.get_nearest_portlan(port_lan)
            # 해당 ionizer에 이 port_lan을 추가하든, 업뎃하든 작업한다.
            if ionizer is not None:
                if id in ionizer.port_lans:
                    ionizer.port_lans[id].update(frame_idx, bboxes)
                else:
                    ionizer.port_lans[id] = port_lan    # 해당 ionizer 객체에 port_lan 등록
                    self.insert_log(f"port_lan({id}) added to ionizer({ionizer.id})")
                    ionizer.show_portlans(self)
                    # volt_check_needed 수정
                    if len(ionizer.port_lans) >= 3 and ionizer.volt_check_needed == False:
                        ionizer.volt_check_needed = True
                        self.insert_log(f"ionizer volt check type changed to 'needed' : ionizer(id={ionizer.id})")
                        
                if work_izr is None: # 프로세스 시작되고 첫 공정시작
                    self.ionizers["working"] = ionizer
                    self.state.process_started(self, ionizer)
                else:
                    if work_izr.id != ionizer.id: 
                        # 기존 공정작업 중인 ionizer가 아닌 다른 ionizer의 port에 lan 꼽았을 때
                        # 공정종료 처리
                        self.state.process_ended(self)
                        
                        # Alert according to result !!!
                        if self.is_successful(work_izr):
                            self.insert_log(f"\n\nSuccessfully Done Process !\n\n")
                            self.alert.ok()
                        else:
                            missed = ''
                            if not work_izr.led_check:
                                missed += 'LED-Check'
                            if work_izr.volt_check_needed and not work_izr.volt_check:
                                missed += '; Volt-Check'
                            self.insert_log(f"\n\nMissed Stage Detected ! : [{missed}]\n\n")
                            self.alert.warn()

                        # 새 공정시작
                        self.ionizers["working"] = ionizer
                        self.state.process_started(self, ionizer)

        if cls == 2: # led_check
            led_check_obj = DetectObject(frame_idx, bboxes, id, cls)
            work_izr:Ionizer = self.ionizers["working"]
            if (work_izr is not None) and (not work_izr.led_check) and work_izr.has_center_in_box(led_check_obj):
                work_izr.led_check = True
                self.insert_log(f"\n\nLED Check is Done ! : ionizer({work_izr.id})\n\n")

        if cls == 3: # volt_check
            volt_check_obj = DetectObject(frame_idx, bboxes, id, cls)
            work_izr:Ionizer = self.ionizers["working"]
            if (work_izr is not None) and (not work_izr.volt_check) and work_izr.has_center_in_box(volt_check_obj):
                work_izr.volt_check = True
                self.insert_log(f"\n\nVolt Check is Done ! : ionizer({work_izr.id})\n\n")

        ### backup ###
        # if cls == 1: # port_lan
        #     port_lan = DetectObject(frame_idx, bboxes, id, cls)
        #     work_izr:Ionizer = self.ionizers["working"]
        #     ionizer :Ionizer
        #     for ion_id, ionizer in self.ionizers["all"].items():
        #         if id in ionizer.port_lans:
        #             ionizer.port_lans[id].update(frame_idx, bboxes)
        #             # break
        #         if ionizer.has_center_in_box(port_lan): # 기존 등록된 ionizer들 중에서 port에 lan이 꼽힌 것 있음
        #             ionizer.port_lans[id] = port_lan    # 해당 ionizer 객체에 port_lan 등록
        #             self.insert_log(f"port_lan({id}) added to ionizer({ion_id})")
        #             ionizer.show_portlans(self)
        # 
        #             if work_izr is None: # 프로세스 시작되고 첫 공정시작
        #                 self.ionizers["working"] = ionizer
        #                 self.state.process_started(self, ionizer)
        #             else:
        #                 if work_izr.id == ionizer.id:
        #                     pass
        #                 else: # 기존 공정작업 중인 ionizer가 아닌 다른 ionizer의 port에 lan 꼽았을 때
        #                       # 공정종료 처리 --> 새 공정시작  ----##############
        #                     self.state.process_ended(self)
        #                     # Alert according to result !!!
        # 
        #                     # 새 공정시작: state.process_started()
        # 
        #                     self.ionizers["working"] = ionizer
        #                     self.state.process_started(self, ionizer)
        #             break
        ###################
        return
    
    def update_ionizer_set(self, ionizer_set):
        all_ionizers = self.ionizers["all"]
        ids_to_remove = [id for id in all_ionizers if id not in ionizer_set]
        for id in ids_to_remove:
            all_ionizers.pop(id, None)
        
        # current_ionizer_ids = list(all_ionizers.keys())
        # self.insert_log(f'current ionizers : {current_ionizer_ids}, ({len(current_ionizer_ids)})\n')
        if len(ids_to_remove) > 0:
            self.insert_log(f"ionizer removed : ids={ids_to_remove}")
            self.show_current_ionizers()

    def update_portlan_set(self, portlan_set):
        all_ionizers = self.ionizers["all"]
        ionizer :Ionizer
        for ion_id, ionizer in all_ionizers.items():
            # with ionizer.lock:
            port_lans = ionizer.port_lans
            if len(port_lans) > 0:
                ids_to_remove = []
                for pl_id, port_lan in port_lans.items():
                    if pl_id not in portlan_set:
                        ids_to_remove.append(pl_id)
                for pl_id in ids_to_remove:
                    port_lans.pop(pl_id, None)
                    self.insert_log(f"port_lan removed : [{pl_id}] in ionizer({ion_id})")
                    ionizer.show_portlans(self)
        # volt_check_needed 수정
        # work_izr: Ionizer = self.ionizers["working"] 
        # if work_izr is None:
        #     return
        # if len(work_izr.port_lans) >= 3 and work_izr.volt_check_needed == False:
        #     work_izr.volt_check_needed = True
        #     self.insert_log(f"ionizer volt check type changed to 'needed' : ionizer(id={work_izr.id})")
    
    def update_after_frame(self, ionizer_set, portlan_set):
        self.update_ionizer_set(ionizer_set)
        self.update_portlan_set(portlan_set)


def main():
    App(video_source=cfg.app["source"])
    

if __name__ == "__main__":
    main()