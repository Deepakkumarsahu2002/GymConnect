import streamlit as st
import requests
import av
import os
import sys
import cv2
import tempfile
import time
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder

# --- Configuration ---
st.set_page_config(layout="centered")

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Main body and app container styling */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Buttons with rounded corners and subtle shadow */
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #005A8D;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Input fields with a subtle style */
    .stTextInput>div>div>input, .stForm>div>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border: 1px solid #3A3A3A;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #0072B5;
        box-shadow: 0 0 0 2px #0072B5;
    }

    /* Headings for a cleaner look */
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
    }
    h1 {
        border-bottom: 2px solid #0072B5;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Top navigation bar styling */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent !important;
        color: #E0E0E0 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 2px solid #0072B5 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Re-using utility functions ---
# These functions were in your other files. We'll copy them here to make the app self-contained.
def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):
    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w), angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)
    cv2.ellipse(img, (x2 - w, y1 + w), (w, w), angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)
    cv2.ellipse(img, (x1 + w, y2 - w), (w, w), angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)
    cv2.ellipse(img, (x2 - w, y2 - w), (w, w), angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)
    return img
def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0
    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)
    return frame
def draw_text(img, msg, width = 8, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=1, font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0), box_offset=(20, 10)):
    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)
    cv2.putText(img, msg, (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return text_size
def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt
    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    degree = int(180 / np.pi) * theta
    return int(degree)
def get_landmark_array(pose_landmark, key, frame_width, frame_height):
    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)
    return np.array([denorm_x, denorm_y])
def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):
    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)
    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)
        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")
def get_mediapipe_pose(static_image_mode = False, model_complexity = 1, smooth_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
    pose = mp.solutions.pose.Pose(static_image_mode = static_image_mode, model_complexity = model_complexity, smooth_landmarks = smooth_landmarks, min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
    return pose
def get_thresholds_beginner():
    _ANGLE_HIP_KNEE_VERT = {'NORMAL' : (0, 32), 'TRANS' : (35, 65), 'PASS' : (70, 95)}
    thresholds = {'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT, 'HIP_THRESH': [10, 50], 'ANKLE_THRESH': 45, 'KNEE_THRESH': [50, 70, 95], 'OFFSET_THRESH': 35.0, 'INACTIVE_THRESH': 15.0, 'CNT_FRAME_THRESH': 50}
    return thresholds
def get_thresholds_pro():
    _ANGLE_HIP_KNEE_VERT = {'NORMAL' : (0, 32), 'TRANS' : (35, 65), 'PASS' : (80, 95)}
    thresholds = {'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT, 'HIP_THRESH': [15, 50], 'ANKLE_THRESH': 30, 'KNEE_THRESH': [50, 80, 95], 'OFFSET_THRESH': 35.0, 'INACTIVE_THRESH': 15.0, 'CNT_FRAME_THRESH': 50}
    return thresholds
class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.radius = 20
        self.COLORS = {'blue': (0, 127, 255), 'red': (255, 50, 50), 'green': (0, 255, 127), 'light_green': (100, 233, 127), 'yellow': (255, 255, 0), 'magenta': (255, 0, 255), 'white': (255,255,255), 'cyan': (0, 255, 255), 'light_blue': (102, 204, 255)}
        self.dict_features = {}
        self.left_features = {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27, 'foot': 31}
        self.right_features = {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28, 'foot': 32}
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0
        self.state_tracker = {'state_seq': [], 'start_inactive_time': time.perf_counter(), 'start_inactive_time_front': time.perf_counter(), 'INACTIVE_TIME': 0.0, 'INACTIVE_TIME_FRONT': 0.0, 'DISPLAY_TEXT': np.full((4,), False), 'COUNT_FRAMES': np.zeros((4,), dtype=np.int64), 'LOWER_HIPS': False, 'INCORRECT_POSTURE': False, 'prev_state': None, 'curr_state':None, 'SQUAT_COUNT': 0, 'IMPROPER_SQUAT':0}
        self.FEEDBACK_ID_MAP = {0: ('BEND BACKWARDS', 215, (0, 153, 255)), 1: ('BEND FORWARD', 215, (0, 153, 255)), 2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)), 3: ('SQUAT TOO DEEP', 125, (255, 80, 80))}

    def _get_state(self, knee_angle):
        knee = None        
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3
        return f's{knee}' if knee else None
    def _update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(state)
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)
    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        if lower_hips_disp:
            draw_text(frame, 'LOWER YOUR HIPS', pos=(30, 80), text_color=(0, 0, 0), font_scale=0.6, text_color_bg=(255, 255, 0))  
        for idx in np.where(c_frame)[0]:
            draw_text(frame, dict_maps[idx][0], pos=(30, dict_maps[idx][1]), text_color=(255, 255, 230), font_scale=0.6, text_color_bg=dict_maps[idx][2])
        return frame
    def process(self, frame: np.array, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape
        keypoints = pose.process(frame)
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks
            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)
            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)
            if offset_angle > self.thresholds['OFFSET_THRESH']:
                display_inactivity = False
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time
                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True
                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                draw_text(frame, 'CAMERA NOT ALIGNED PROPERLY!!!', pos=(30, frame_height-60), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                draw_text(frame, 'OFFSET ANGLE: '+str(offset_angle), pos=(30, frame_height-30), text_color=(255, 255, 230), font_scale=0.65, text_color_bg=(255, 153, 0)) 
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]
                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None
                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord
                    multiplier = -1
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord
                    multiplier = 1
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, color = self.COLORS['white'], thickness = 3, lineType = self.linetype)
                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])
                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)
                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])
                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30), angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle, color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)
                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)
                if current_state == 's1':
                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT']+=1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False
                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and self.state_tracker['state_seq'].count('s2')==1:
                            self.state_tracker['DISPLAY_TEXT'][1] = True
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['LOWER_HIPS'] = True
                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                display_inactivity = False
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:
                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time
                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True
                else:
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])
                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
                draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state
        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)
            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
            display_inactivity = False
            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True
            self.state_tracker['start_inactive_time'] = end_time
            draw_text(frame, "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), pos=(int(frame_width*0.68), 30), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(18, 185, 0))  
            draw_text(frame, "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), pos=(int(frame_width*0.68), 80), text_color=(255, 255, 230), font_scale=0.7, text_color_bg=(221, 0, 0))  
            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
        return frame, play_sound


# --- Constants ---
ADMIN_USER = "admin"
ADMIN_PASS = "adminpass" 
BACKEND_URL = "http://localhost:5000"

# --- Streamlit Session State Initialization ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# --- Page Functions ---
def show_home_page():
    st.title('GymConnect-AI Fitness Trainer')
    if st.session_state['authenticated']:
        st.success(f"Welcome back, {st.session_state['username']}!")
        
        # Display the demo video for all users
        recorded_file = 'output_sample.mp4'
        st.video(recorded_file)
        
        # Admin-specific content
        if st.session_state['user_type'] == 'Admin':
            st.markdown("---")
            st.subheader("Admin Panel")
            st.write("Click below to view all user data and submitted questions.")
            st.link_button("Go to Admin Panel", f"{BACKEND_URL}/view_admin_panel")
    else:
        st.subheader("Login to your Account")
        with st.form("login_form", clear_on_submit=True):
            user_type = st.selectbox("I am a:", ('User', 'Admin'))
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if user_type == 'Admin':
                    if username == ADMIN_USER and password == ADMIN_PASS:
                        st.session_state['authenticated'] = True
                        st.session_state['user_type'] = 'Admin'
                        st.session_state['username'] = ADMIN_USER
                        st.session_state['page'] = 'Home'
                        st.rerun()
                    else:
                        st.error("Invalid admin credentials.")
                else: # User login
                    try:
                        response = requests.post(f"{BACKEND_URL}/login", json={"username": username, "password": password})
                        if response.status_code == 200:
                            st.session_state['authenticated'] = True
                            st.session_state['user_type'] = 'User'
                            st.session_state['username'] = username
                            st.session_state['page'] = 'Home'
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the backend server. Please ensure the Flask app is running.")
        st.markdown("---")
        st.subheader("Don't have an account? Sign up!")
        with st.form("signup_form", clear_on_submit=True):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            signup_button = st.form_submit_button("Sign Up")
            if signup_button:
                try:
                    response = requests.post(f"{BACKEND_URL}/signup", json={"username": new_username, "password": new_password})
                    if response.status_code == 201:
                        st.success("✅ Account created successfully! You can now log in.")
                    else:
                        st.error(f"❌ Error: {response.json().get('message')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Please ensure the Flask app is running.")

def show_live_stream_page():
    st.title('DO YOUR LIVE SESSION HERE')
    mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True, key='live_mode_radio')
    thresholds = None 
    if mode == 'Beginner':
        thresholds = get_thresholds_beginner()
    elif mode == 'Pro':
        thresholds = get_thresholds_pro()
    live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
    pose = get_mediapipe_pose()
    if 'download' not in st.session_state:
        st.session_state['download'] = False
    output_video_file = f'output_live.flv'
    def video_frame_callback(frame: av.VideoFrame):
        frame = frame.to_ndarray(format="rgb24")
        frame, _ = live_process_frame.process(frame, pose)
        return av.VideoFrame.from_ndarray(frame, format="rgb24")
    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(output_video_file)
    ctx = webrtc_streamer(key="Squats-pose-analysis", video_frame_callback=video_frame_callback, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False}, video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False), out_recorder_factory=out_recorder_factory)
    download_button = st.empty()
    if os.path.exists(output_video_file):
        with open(output_video_file, 'rb') as op_vid:
            download = download_button.download_button('Download Video', data = op_vid, file_name='output_live.flv')
            if download:
                st.session_state['download'] = True
    if os.path.exists(output_video_file) and st.session_state['download']:
        os.remove(output_video_file)
        st.session_state['download'] = False
        download_button.empty()
def show_upload_video_page():
    st.title('UPLOAD YOUR VIDEO HERE')
    mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True, key='upload_mode_radio')
    thresholds = None 
    if mode == 'Beginner':
        thresholds = get_thresholds_beginner()
    elif mode == 'Pro':
        thresholds = get_thresholds_pro()
    upload_process_frame = ProcessFrame(thresholds=thresholds)
    pose = get_mediapipe_pose()
    download = None
    if 'download' not in st.session_state:
        st.session_state['download'] = False
    output_video_file = f'output_recorded.mp4'
    if os.path.exists(output_video_file):
        os.remove(output_video_file)
    with st.form('Upload', clear_on_submit=True):
        up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
        uploaded = st.form_submit_button("Upload")
    stframe = st.empty()
    ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'
    warning_str = '<p style="font-family:Helvetica; font-weight: bold; color: Red; font-size: 17px;">Please Upload a Video first!!!</p>'
    warn = st.empty()
    download_button = st.empty()
    if up_file and uploaded:
        download_button.empty()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        try:
            warn.empty()
            tfile.write(up_file.read())
            vf = cv2.VideoCapture(tfile.name)
            fps = int(vf.get(cv2.CAP_PROP_FPS))
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
            txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)
            ip_video = st.sidebar.video(tfile.name)
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_frame, _ = upload_process_frame.process(frame, pose)
                stframe.image(out_frame)
                video_output.write(out_frame[...,::-1])
            vf.release()
            video_output.release()
            stframe.empty()
            ip_video.empty()
            txt.empty()
            tfile.close()
        except AttributeError:
            warn.markdown(warning_str, unsafe_allow_html=True)   
    if os.path.exists(output_video_file):
        with open(output_video_file, 'rb') as op_vid:
            download = download_button.download_button('Download Video', data = op_vid, file_name='output_recorded.mp4')
        if download:
            st.session_state['download'] = True
    if os.path.exists(output_video_file) and st.session_state['download']:
        os.remove(output_video_file)
        st.session_state['download'] = False
        download_button.empty()
def show_exercise_guide_images():
    st.title("EXERCISE GUIDE")
    instructions = """
    Are you not aware of the correct Technique!  
    ✅We are here to guide you.
    See the Images and Videos and know about your exercise.
    """
    st.markdown(instructions)
    assets_folder = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_folder):
        st.error(f"❌ 'assets' folder not found in {os.getcwd()}")
        return
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = [f for f in os.listdir(assets_folder) if os.path.splitext(f)[1].lower() in image_extensions]
    if image_files:
        cols = st.columns(3)
        for idx, img in enumerate(image_files):
            with open(os.path.join(assets_folder, img), "rb") as file:
                img_bytes = file.read()
            cols[idx % 3].image(img_bytes, use_container_width=True)
    else:
        st.warning("⚠️ No image files found in assets/ folder.")
def show_exercise_guide_videos():
    st.title("EXERCISE VIDEOS")
    instructions = """
    Watch the videos carefully to perform your exercises correctly.  
    ✅ Follow the proper technique to avoid injuries.
    """
    st.markdown(instructions)
    assets_folder = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_folder):
        st.error(f"❌ 'assets' folder not found in {os.getcwd()}")
        return
    video_extensions = [".mp4", ".avi", ".mov"]
    video_files = [f for f in os.listdir(assets_folder) if os.path.splitext(f)[1].lower() in video_extensions]
    if video_files:
        cols = st.columns(2)
        for idx, vid in enumerate(video_files):
            with open(os.path.join(assets_folder, vid), "rb") as video_file:
                video_bytes = video_file.read()
            cols[idx % 2].video(video_bytes, format="video/mp4", start_time=0)
            cols[idx % 2].caption(vid)
    else:
        st.warning("⚠️ No video files found in assets/ folder.")
def show_contact_trainer():
    st.title("📩 Contact Our Certified Trainer")
    exercise_categories = ["Select","Chest","Back","Legs","Arms","Abs","Other"]

    # Add the instructions here
    st.info("Our certified trainer will personally contact you through your provided contact information to address your question and problems.")

    with st.form("contact_form"):
        name = st.text_input("Full Name *")
        email = st.text_input("Your Email *")
        phone = st.text_input("Phone Number")
        category = st.selectbox("Exercise Category *", exercise_categories)
        question = st.text_area("Your Question / Message *")
        submit = st.form_submit_button("Send Message")
    if submit:
        payload = {"name": name, "email": email, "phone": phone, "category": category, "question": question}
        try:
            response = requests.post(f"{BACKEND_URL}/submit_question", json=payload)
            if response.status_code == 200:
                st.success("✅ Your question has been submitted successfully!")
            else:
                st.error(f"❌ Error: {response.json().get('message')}")
        except Exception as e:
            st.error(f"❌ Could not connect to backend. Error: {e}")

# --- Main App Logic ---
if st.session_state['authenticated']:
    # Top bar navigation
    page = st.tabs(["Home", "Live Stream", "Upload Video", "Exercise Guide (Images)", "Exercise Guide (Videos)", "Contact Trainer"])
    
    with page[0]:
        show_home_page()
    with page[1]:
        show_live_stream_page()
    with page[2]:
        show_upload_video_page()
    with page[3]:
        show_exercise_guide_images()
    with page[4]:
        show_exercise_guide_videos()
    with page[5]:
        show_contact_trainer()
    
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())
    
else:
    show_home_page()