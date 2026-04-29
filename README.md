# 套图调色

这是一个适合放到 GitHub 和 Streamlit 的版本。

功能：

- 上传底图、蒙版、花型贴图、参考图
- 自动搜索 `Scale / Hue / Saturation / Value`
- 输出最佳 `JPG`
- 输出分层 `PSD`
- 可下载全部结果 `ZIP`

## 文件说明

- `app_deploy.py`: Streamlit 页面入口
- `texture_mockup_core.py`: 核心算法与 PSD 导出
- `requirements.txt`: Python 依赖
- `.streamlit/config.toml`: Streamlit 配置

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app_deploy.py
```

## 发布到 GitHub

把下面这些文件提交到仓库根目录：

- `app_deploy.py`
- `texture_mockup_core.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`

## 发布到 Streamlit

1. 把代码推到 GitHub
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 选择你的仓库
4. Main file path 填 `app_deploy.py`
5. 部署即可

## 注意

- 为了更适合云端部署，这里使用的是 `opencv-python-headless`
- 如果一次命中很多结果，并且勾选“每个命中结果都导出 PSD”，处理时间会明显变长
- Streamlit 云端对单次上传大小、运行时长、内存都有一定限制，超大图片建议先压缩后再传
