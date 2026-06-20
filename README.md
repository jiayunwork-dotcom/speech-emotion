# 🎙️ 语音情感识别与说话人分离分析平台

一个基于Python的语音情感识别与说话人分离后端分析平台，支持音频上传、自动分析、可视化展示等功能。

## ✨ 功能特性

### 核心功能
- **音频输入处理**: 支持WAV/MP3格式，最大50MB，最长30分钟，自动转换为16kHz单声道
- **语音活动检测(VAD)**: 智能静音切分，自动合并间隔过短的语音片段
- **说话人分离**: 基于MFCC特征和层次聚类，自动识别最多10个说话人
- **情感识别**: 6种情感分类（中性、开心、愤怒、悲伤、恐惧、惊讶），支持随机森林模型和规则兜底
- **时间轴标注**: 输出JSON和SRT格式的标注结果
- **对话统计分析**: 说话人占比、情感分布、打断次数、平均发言时长等
- **批量处理**: 支持最多10个文件批量上传，并行处理（最多3个同时处理）

### REST API接口
- `POST /upload` - 上传音频文件，返回任务ID
- `GET /task/{task_id}/status` - 查询任务状态
- `GET /task/{task_id}/result` - 获取完整分析结果（JSON）
- `GET /task/{task_id}/summary` - 获取统计摘要
- `GET /task/{task_id}/srt` - 下载SRT字幕
- `GET /task/{task_id}/audio` - 下载处理后的音频
- `POST /batch/upload` - 批量上传音频文件
- `GET /batch/{batch_id}/status` - 查询批次处理进度

### Streamlit可视化面板
- 音频波形图（带说话人分段色块）
- 情感时间轴（与波形图对齐）
- 统计图表（饼图+柱状图）
- 详细标注表格（支持筛选）
- 片段播放功能

## 📁 项目结构

```
speech-emotion/
├── src/
│   ├── api/                    # FastAPI REST API
│   │   ├── __init__.py
│   │   └── main.py             # API主入口
│   ├── core/                   # 核心配置和模型
│   │   ├── __init__.py
│   │   ├── config.py           # 配置参数
│   │   └── models.py           # Pydantic数据模型
│   ├── audio/                  # 音频处理模块
│   │   ├── __init__.py
│   │   └── processor.py        # 音频加载、转换、VAD
│   ├── speaker/                # 说话人分离模块
│   │   ├── __init__.py
│   │   └── separation.py       # 特征提取、层次聚类
│   ├── emotion/                # 情感识别模块
│   │   ├── __init__.py
│   │   └── recognizer.py       # 特征提取、随机森林分类
│   ├── output/                 # 结果输出模块
│   │   ├── __init__.py
│   │   └── formatter.py        # JSON/SRT格式化
│   ├── stats/                  # 统计分析模块
│   │   ├── __init__.py
│   │   └── analyzer.py         # 统计指标计算
│   └── tasks/                  # 任务管理模块
│       ├── __init__.py
│       ├── manager.py          # 任务队列、批量处理
│       └── pipeline.py         # 分析流水线
├── streamlit_app/              # Streamlit可视化面板
│   └── app.py                  # 前端应用
├── models/                     # 预训练模型目录
│   └── README.md               # 模型说明
├── data/                       # 数据目录
│   ├── raw/                    # 原始上传文件
│   └── processed/              # 处理后文件和结果
├── tests/                      # 单元测试
│   ├── test_audio_processor.py
│   ├── test_speaker_separation.py
│   ├── test_emotion_recognizer.py
│   └── test_stats_analyzer.py
├── docker/                     # Docker配置
│   └── Dockerfile              # 镜像构建文件
├── docker-compose.yml          # Docker编排
├── requirements.txt            # Python依赖
├── start_api.py                # API启动脚本
├── start_streamlit.py          # Streamlit启动脚本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 方式一：本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动API服务**
```bash
python start_api.py
```
API将在 http://localhost:8000 启动

3. **启动Streamlit面板**（另开一个终端）
```bash
python start_streamlit.py
```
面板将在 http://localhost:8501 启动

4. **访问API文档**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 方式二：Docker运行

```bash
# 构建并启动服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

服务将在以下地址访问：
- API: http://localhost:8000
- Streamlit面板: http://localhost:8501

## 🧪 运行测试

```bash
# 安装pytest
pip install pytest

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_audio_processor.py -v
```

## 📊 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| Web框架 | FastAPI | 0.115.0 |
| 异步服务 | Uvicorn | 0.30.6 |
| 可视化面板 | Streamlit | 1.38.0 |
| 音频处理 | Librosa | 0.10.2 |
| 机器学习 | Scikit-learn | 1.5.2 |
| 科学计算 | NumPy, SciPy | 1.26.4, 1.13.1 |
| 数据处理 | Pandas | 2.2.2 |
| 图表 | Plotly, Matplotlib | 5.24.1, 3.9.2 |
| 容器化 | Docker | - |

## 🔧 核心算法说明

### 说话人分离
1. **特征提取**: 每个语音片段分成1.5秒窗口（50%重叠），提取40维MFCC（20个系数+一阶差分）
2. **嵌入向量**: 所有窗口特征的均值+标准差拼接为80维向量
3. **聚类算法**: 层次聚类（凝聚聚类），余弦距离，平均链接
4. **自动估计说话人数**: 距离阈值0.6，最多10个说话人

### 情感识别
1. **特征提取**: 35维声学特征
   - F0均值、标准差、范围（3维）
   - 语速（1维）
   - 能量均值、范围（2维）
   - MFCC前13维的均值和标准差（26维）
   - 频谱质心均值、频谱滚降频率均值（2维）
   - 总计: 34维（实际35维）

2. **分类器**: 随机森林（pickle序列化模型）
3. **规则兜底**: 当模型文件不存在时使用
   - 基频标准差大且能量高 → 愤怒
   - 基频均值高且变化大 → 开心
   - 基频低且能量低 → 悲伤
   - 其他 → 中性

## 📝 API使用示例

### 上传音频
```python
import requests

url = "http://localhost:8000/upload"
files = {"file": open("audio.wav", "rb")}
response = requests.post(url, files=files)
task_id = response.json()["task_id"]
print(f"任务ID: {task_id}")
```

### 查询任务状态
```python
import requests
import time

task_id = "your-task-id"
while True:
    response = requests.get(f"http://localhost:8000/task/{task_id}/status")
    status = response.json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"失败: {status['error_message']}")
        break
    print(f"进度: {status['progress']*100:.0f}%")
    time.sleep(2)
```

### 获取分析结果
```python
import requests

task_id = "your-task-id"
response = requests.get(f"http://localhost:8000/task/{task_id}/result")
result = response.json()

for seg in result["segments"]:
    print(f"[{seg['start_ms']/1000:.1f}s - {seg['end_ms']/1000:.1f}s] "
          f"{seg['speaker']}: {seg['emotion']} ({seg['emotion_confidence']:.2f})")
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License
