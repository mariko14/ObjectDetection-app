import streamlit as st
import PIL

#ページレイアウトの設定
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

#サイドバー
with st.sidebar:
    st.header("Image/Video Config")
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

st.title("Object Detection using YOLOv8")
col1, col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)


        st.image(source_img,
                 caption="Upliaded Image",
                 use_column_width=True
                 )
        


# streamlit run ObjectDetection/image_upload_app.py