# K230 手势识别+人脸检测 按键切换
# 放在/sdcard/main.py，上电自动跑
# 之前试过帧级交替但是两个模型冲突了，改成按键切换

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *
from media.sensor import *
import os, gc, time, aidemo
import nncase_runtime as nn
import ulab.numpy as np
from machine import Pin, PWM, FPIOA

# ==========================================
# GPIO
# ==========================================
fpioa = FPIOA()
fpioa.set_function(62, FPIOA.GPIO62); fpioa.set_function(20, FPIOA.GPIO20)
fpioa.set_function(63, FPIOA.GPIO63); fpioa.set_function(43, FPIOA.PWM1)
fpioa.set_function(53, FPIOA.GPIO53)

LED_R = Pin(62, Pin.OUT, drive=7)
LED_G = Pin(20, Pin.OUT, drive=7)
LED_B = Pin(63, Pin.OUT, drive=7)
buzzer = PWM(1); buzzer.freq(4000)
BUTTON = Pin(53, Pin.IN, Pin.PULL_DOWN)
LED_R.value(1); LED_G.value(1); LED_B.value(1); buzzer.duty_u16(0)

# ==========================================
# 人脸检测类（独立AIBase，一次只存在一个）
# ==========================================
class FaceAI(AIBase):
    def __init__(self, kmodel_path, anchors, rgb888p_size, display_size):
        super().__init__(kmodel_path, [320, 320], rgb888p_size, 0)
        self.conf_thresh = 0.5
        self.nms_thresh = 0.2
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_pp(self):
        top, bottom, left, right, _ = letterbox_pad_param(self.rgb888p_size, [320, 320])
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, 3, self.rgb888p_size[1], self.rgb888p_size[0]], [1, 3, 320, 320])

    def postprocess(self, results):
        ret = aidemo.face_det_post_process(self.conf_thresh, self.nms_thresh, 320, self.anchors, self.rgb888p_size, results)
        return ret[0] if len(ret) > 0 else []

# ==========================================
# 模型管理
# ==========================================
GESTURE_LABELS = {0: 'fist', 1: 'five', 2: 'gun', 3: 'one', 4: 'thumbUp', 5: 'yeah'}
pl = PipeLine(rgb888p_size=[640, 360], display_size=[640, 480], display_mode="virt")
pl.create(sensor=Sensor(id=2, width=1920, height=1080))
display_size = pl.get_display_size()

# 人脸锚点
face_anchors = np.fromfile("/sdcard/examples/utils/prior_data_320.bin", dtype=np.float)
face_anchors = face_anchors.reshape((4200, 4))

gesture_model = None  # YOLO11
face_model = None     # FaceAI
active = "gesture"    # 当前激活

# 预加载两个模型
gesture_model = YOLO11(task_type="detect", mode="video",
    kmodel_path="/sdcard/yolo11n_det_320.kmodel",
    labels=GESTURE_LABELS,
    rgb888p_size=[640, 360], model_input_size=[320, 320],
    display_size=display_size,
    conf_thresh=0.6, nms_thresh=0.45, max_boxes_num=50, debug_mode=0)
gesture_model.config_preprocess()

face_model = FaceAI("/sdcard/examples/kmodel/face_detection_320.kmodel",
                    face_anchors, [640, 360], display_size)
face_model.config_pp()

# ==========================================
# 状态
# ==========================================
MODE_GESTURE, MODE_FACE = 0, 1
mode = MODE_GESTURE
last_key, pending_key = "?", "?"
debounce_count = 0
btn_last = 0
face_timer = 0

def flash(n):
    for _ in range(n):
        LED_R.value(0); LED_G.value(0); LED_B.value(0); time.sleep_ms(30)
        LED_R.value(1); LED_G.value(1); LED_B.value(1); time.sleep_ms(20)

flash(1)
print("Gesture mode")

# ==========================================
# 主循环
# ==========================================
while True:
    os.exitpoint()
    img = pl.get_frame()
    if img is None:
        time.sleep_ms(10)
        continue

    # ===== 按键 =====
    btn = BUTTON.value()
    if btn == 1 and btn_last == 0:
        mode = 1 - mode
        active = "gesture" if mode == MODE_GESTURE else "face"
        flash(mode + 1)
        last_key = "?"; pending_key = "?"; debounce_count = 0
        time.sleep_ms(200)  # 等 KPU 切换稳定
    btn_last = btn

    # ===== 推理 =====
    key = "?"

    if active == "gesture" and gesture_model:
        res = gesture_model.run(img)
        gesture_model.draw_result(res, pl.osd_img)
        if res and len(res) >= 2 and len(res[0]) > 0:
            idx, score = res[1][0], res[2][0]
            name = GESTURE_LABELS.get(idx, "?")
            if name == "five" and score < 0.75: pass
            elif name == "gun" and score < 0.65: pass
            elif name == "one" and score < 0.75: pass
            elif name == "yeah" and score < 0.75: pass
            else: key = name

    elif active == "face" and face_model:
        pl.osd_img.clear()
        res = face_model.run(img)
        if res:
            for det in res:
                x, y, w, h = map(lambda v: int(round(v, 0)), det[:4])
                x = x * display_size[0] // 640
                y = y * display_size[1] // 360
                w = w * display_size[0] // 640
                h = h * display_size[1] // 360
                pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 255), thickness=2)
            key = "face"

    # ===== 防抖 =====
    if key == pending_key: debounce_count += 1
    else: pending_key = key; debounce_count = 1

    if debounce_count >= 2 and pending_key != last_key:
        LED_R.value(1); LED_G.value(1); LED_B.value(1)
        if last_key != "face": buzzer.duty_u16(0)
        k = pending_key
        if k == "fist": LED_G.value(0)
        elif k == "five": LED_R.value(0)
        elif k == "yeah": LED_B.value(0)
        elif k == "gun": buzzer.duty_u16(32768)
        elif k == "thumbUp": LED_R.value(0); LED_G.value(0)
        elif k == "one": LED_B.value(0); buzzer.duty_u16(32768)
        elif k == "face": buzzer.duty_u16(32768); face_timer = 3
        last_key = k

    # 人脸短蜂鸣
    if face_timer > 0:
        face_timer -= 1
        if face_timer == 0:
            buzzer.duty_u16(0)

    # gun 持续蜂鸣
    if pending_key == "gun" and debounce_count >= 2:
        buzzer.duty_u16(32768)

    # ===== 显示 =====
    if active == "gesture":
        pl.osd_img.draw_string_advanced(10, 10, 28, "Gesture", color=(0,255,0))
    else:
        pl.osd_img.draw_string_advanced(10, 10, 28, "Face", color=(0,255,0))
    pl.osd_img.draw_string_advanced(10, 42, 28, key, color=(255,255,0))

    pl.show_image()
    gc.collect()
    time.sleep_ms(5)
