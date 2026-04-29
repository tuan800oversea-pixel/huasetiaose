import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from texture_mockup_core import process_uploaded_files


st.set_page_config(page_title="套图调色", page_icon="🎨", layout="wide")

st.title("套图调色")
st.caption("上传底图、蒙版、花型和参考图，自动搜索调色参数，并导出 JPG / PSD。")


def ensure_bytesio_download(path: str):
    with open(path, "rb") as f:
        return f.read()


def build_zip_from_dir(output_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                arcname = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)


with st.sidebar:
    st.subheader("参数")
    delta_e_threshold = st.slider("命中阈值 Delta E", 1.0, 20.0, 7.0, 0.5)
    export_hit_psd = st.checkbox("每个命中结果都导出 PSD", value=True)

    st.subheader("说明")
    st.write("`JPG` 适合预览，`PSD` 保留背景、蒙版、花型、光影分层。")
    st.write("如果命中结果很多，导出 PSD 会更慢。")


col1, col2 = st.columns(2)
with col1:
    original_file = st.file_uploader("上传底图", type=["jpg", "jpeg", "png"])
    mask_file = st.file_uploader("上传蒙版", type=["png", "jpg", "jpeg"])
with col2:
    pattern_file = st.file_uploader("上传花型贴图", type=["png", "jpg", "jpeg"])
    ref_file = st.file_uploader("上传参考图", type=["png", "jpg", "jpeg"])


run = st.button("开始生成", type="primary", use_container_width=True)

if run:
    missing = [
        name
        for name, uploaded in [
            ("底图", original_file),
            ("蒙版", mask_file),
            ("花型贴图", pattern_file),
            ("参考图", ref_file),
        ]
        if uploaded is None
    ]

    if missing:
        st.error("请先上传: " + "、".join(missing))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "inputs")
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            with st.spinner("正在搜索最优调色参数并导出结果..."):
                result = process_uploaded_files(
                    original_file=original_file,
                    mask_file=mask_file,
                    pattern_file=pattern_file,
                    ref_file=ref_file,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    delta_e_threshold=delta_e_threshold,
                    export_hit_psd=export_hit_psd,
                )

            st.success("处理完成")

            st.write(
                f"最佳色差: `{result['best_diff']:.2f}`，命中数量: `{result['hit_count']}`，耗时: `{result['elapsed_seconds']:.1f}` 秒"
            )

            preview_path = result.get("best_preview_path")
            if preview_path and os.path.exists(preview_path):
                st.image(preview_path, caption="最佳结果预览", use_container_width=True)

            best_jpg = result.get("best_jpg_path")
            best_psd = result.get("best_psd_path")

            if best_jpg and os.path.exists(best_jpg):
                st.download_button(
                    "下载最佳 JPG",
                    data=ensure_bytesio_download(best_jpg),
                    file_name=Path(best_jpg).name,
                    mime="image/jpeg",
                    use_container_width=True,
                )

            if best_psd and os.path.exists(best_psd):
                st.download_button(
                    "下载最佳 PSD",
                    data=ensure_bytesio_download(best_psd),
                    file_name=Path(best_psd).name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )

            zip_path = os.path.join(tmpdir, "套图调色输出.zip")
            build_zip_from_dir(output_dir, zip_path)
            st.download_button(
                "下载全部结果 ZIP",
                data=ensure_bytesio_download(zip_path),
                file_name="套图调色输出.zip",
                mime="application/zip",
                use_container_width=True,
            )
