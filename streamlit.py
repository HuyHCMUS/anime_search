import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# URL của FastAPI server
FASTAPI_URL = "http://localhost:8000/search/"

# Tiêu đề cho ứng dụng
st.title("Anime Search Engine")

# Khởi tạo biến `load_count` để đếm số anime đã hiển thị
if 'load_count' not in st.session_state:
    st.session_state.load_count = 0  # Khởi tạo với 0 anime hiển thị

# Tạo phần tải lên ảnh
uploaded_file = st.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])

# Khi người dùng tải ảnh lên
if uploaded_file is not None:
    # Hiển thị ảnh người dùng vừa tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Chuyển đổi ảnh sang dạng bytes để gửi về server
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    # Tạo nút "Tìm kiếm"
    if st.button("Tìm kiếm"):
        # Gửi yêu cầu POST tới FastAPI server
        files = {'file': uploaded_file.getvalue()}
        with st.spinner("Đang tìm kiếm..."):
            response = requests.post(FASTAPI_URL, files=files)

        # Nếu yêu cầu thành công
        if response.status_code == 200:
            # Load dữ liệu JSON trả về từ server và lưu vào session state
            st.session_state.data = pd.DataFrame(response.json())
            
            # Đếm số nhân vật khác nhau và tổng số ảnh cho mỗi anime
            anime_stats = st.session_state.data.groupby('anime').agg(
                count_distinct=('character', 'nunique'),
                count=('character', 'size')
            ).reset_index().sort_values(
                by=['count_distinct', 'count'], ascending=[False, False]
            )

            # Lưu `anime_stats` vào `st.session_state` để sử dụng khi bấm "Tải thêm"
            st.session_state.anime_data = anime_stats
            st.session_state.load_count = 1  # Bắt đầu với 1 lô đầu tiên

# Kiểm tra nếu anime_data đã tồn tại trong `st.session_state` trước khi truy xuất
if 'anime_data' in st.session_state:
    st.success("Tìm kiếm thành công!")
    st.write(st.session_state.anime_data['anime'])
    st.subheader("Một số hình ảnh tương tự:")

    # Hiển thị từng đợt 3 anime
    for load_index in range(st.session_state.load_count):
        start = load_index * 3
        end = start + 3
        batch_data = st.session_state.anime_data.iloc[start:end]

        # Lọc và hiển thị 3 anime hiện tại
        cols = st.columns(3)  # Chia thành 3 cột
        count = 0

        for _, anime_row in batch_data.iterrows():
            anime_name = anime_row['anime']
            anime_data = st.session_state.data[st.session_state.data['anime'] == anime_name]
            
            # Lấy các ảnh từ nhân vật khác nhau trước, tối đa 3 ảnh
            unique_characters = anime_data.drop_duplicates(subset=['character']).nsmallest(3, 'distance')
            
            # Nếu số ảnh từ nhân vật khác nhau ít hơn 3, bổ sung thêm các ảnh của các nhân vật hiện có
            if len(unique_characters) < 3:
                remaining_images = anime_data[~anime_data.index.isin(unique_characters.index)]
                additional_images = remaining_images.nsmallest(3 - len(unique_characters), 'id')
                final_images = pd.concat([unique_characters, additional_images])
            else:
                final_images = unique_characters
            
            # Hiển thị các ảnh theo thứ tự ưu tiên trong các cột
            for _, row in final_images.iterrows():
                col = cols[count % 3]  # Chọn cột tương ứng
                col.image(row['similar_images'], use_column_width=True)
                col.markdown(f"**ID {row['id']}: {row['character']}**<br>Anime: {row['anime']}", unsafe_allow_html=True)
                count += 1

    # Cập nhật nút "Tải thêm" để hiển thị các lô mới bên dưới
    if end < len(st.session_state.anime_data):
        if st.button("Tải thêm"):
            st.session_state.load_count += 1  # Tăng biến đếm lên để hiển thị đợt anime tiếp theo
