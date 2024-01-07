import ultralytics
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

import cv2
# import pafy
from pytube import YouTube

import settings

def load_model(model_path):
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)
       
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

# pafyから変更！　pytubeへ
def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    if st.sidebar.button('Detect Objects'):
        try:
            # YouTubeオブジェクトを作成
            video = YouTube(source_youtube)

            # 最高画質のストリームを選択
            best = video.streams.filter(file_extension='mp4').order_by('resolution').desc().first()

            # ビデオのURLを取得
            vid_cap = cv2.VideoCapture(best.url)
            st_frame = st.empty()

            # ビデオをフレームごとに処理
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                            model,
                                            st_frame,
                                            image,
                                            )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


#【追加】カメラからの入力
def play_webcam_video(conf, model):
    st.header("Webcam Live Feed")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        # モデルを使って画像を解析
        res = model.predict(image, conf=conf)
        res_plotted = res[0].plot()
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

    webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={  # この設定を足す
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
