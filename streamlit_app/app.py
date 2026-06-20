import streamlit as st
import requests
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import io
import librosa
import soundfile as sf

import os
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

SPEAKER_COLORS = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
EMOTION_COLORS = {
    "neutral": "#808080",
    "happy": "#FFD700",
    "angry": "#FF4500",
    "sad": "#4169E1",
    "fear": "#9932CC",
    "surprise": "#32CD32"
}

EMOTION_LABELS_CN = {
    "neutral": "中性",
    "happy": "开心",
    "angry": "愤怒",
    "sad": "悲伤",
    "fear": "恐惧",
    "surprise": "惊讶"
}

st.set_page_config(
    page_title="语音情感识别与说话人分离平台",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 语音情感识别与说话人分离分析平台")
st.markdown("---")

def upload_audio(file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_BASE_URL}/upload", files=files)
    return response.json()

def get_task_status(task_id):
    response = requests.get(f"{API_BASE_URL}/task/{task_id}/status")
    return response.json()

def get_task_result(task_id):
    response = requests.get(f"{API_BASE_URL}/task/{task_id}/result")
    return response.json()

def get_task_summary(task_id):
    response = requests.get(f"{API_BASE_URL}/task/{task_id}/summary")
    return response.json()

def get_audio_data(task_id):
    response = requests.get(f"{API_BASE_URL}/task/{task_id}/audio")
    return response.content

def get_task_quality(task_id):
    response = requests.get(f"{API_BASE_URL}/task/{task_id}/quality")
    if response.status_code == 200:
        return response.json()
    return None

def list_completed_tasks():
    response = requests.get(f"{API_BASE_URL}/tasks", params={"status": "completed"})
    return response.json()

def compare_tasks(task1_id, task2_id):
    response = requests.get(f"{API_BASE_URL}/compare", params={"task1_id": task1_id, "task2_id": task2_id})
    return response.json()

def get_conclusion_color(conclusion):
    if conclusion == "task1占优":
        return "color: #28a745; font-weight: bold;"
    elif conclusion == "task2占优":
        return "color: #dc3545; font-weight: bold;"
    else:
        return "color: #6c757d; font-weight: bold;"

def load_audio_from_bytes(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    return y, sr

def generate_srt(result_data):
    lines = []
    for i, seg in enumerate(result_data['segments'], 1):
        start_ms = seg['start_ms']
        end_ms = seg['end_ms']
        
        def format_time(ms):
            seconds = ms / 1000.0
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        speaker = seg.get('speaker', 'Unknown')
        emotion = EMOTION_LABELS_CN.get(seg.get('emotion', 'neutral'), seg.get('emotion', 'neutral'))
        confidence = f"{seg.get('emotion_confidence', 0):.2f}"
        
        lines.append(str(i))
        lines.append(f"{format_time(start_ms)} --> {format_time(end_ms)}")
        lines.append(f"[{speaker}] [{emotion}] (置信度: {confidence})")
        lines.append("")
    
    return "\n".join(lines)

def plot_waveform_with_speakers(y, sr, segments):
    fig = go.Figure()
    
    times = np.arange(len(y)) / sr
    fig.add_trace(go.Scatter(
        x=times,
        y=y,
        mode='lines',
        name='波形',
        line=dict(color='rgba(100, 100, 100, 0.5)', width=1),
        hoverinfo='none'
    ))
    
    speaker_list = sorted(set(seg['speaker'] for seg in segments if seg.get('speaker')))
    speaker_color_map = {speaker: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] 
                        for i, speaker in enumerate(speaker_list)}
    
    for seg in segments:
        if seg.get('speaker'):
            start_sec = seg['start_ms'] / 1000
            end_sec = seg['end_ms'] / 1000
            speaker = seg['speaker']
            color = speaker_color_map.get(speaker, '#888888')
            
            fig.add_shape(
                type="rect",
                x0=start_sec,
                x1=end_sec,
                y0=-np.max(np.abs(y)) * 1.1,
                y1=np.max(np.abs(y)) * 1.1,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                name=speaker
            )
    
    for speaker, color in speaker_color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=speaker,
            showlegend=True
        ))
    
    fig.update_layout(
        title="音频波形图 (说话人分段)",
        xaxis_title="时间 (秒)",
        yaxis_title="振幅",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_emotion_timeline(segments, duration_ms):
    fig = go.Figure()
    
    speaker_list = sorted(set(seg['speaker'] for seg in segments if seg.get('speaker')))
    speaker_color_map = {speaker: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] 
                        for i, speaker in enumerate(speaker_list)}
    
    for seg in segments:
        start_sec = seg['start_ms'] / 1000
        end_sec = seg['end_ms'] / 1000
        emotion = seg.get('emotion', 'neutral')
        color = EMOTION_COLORS.get(emotion, '#808080')
        speaker = seg.get('speaker', 'Unknown')
        
        emotion_score = {
            "neutral": 0,
            "happy": 2,
            "angry": -2,
            "sad": -1,
            "fear": -1.5,
            "surprise": 1.5
        }.get(emotion, 0)
        
        mid_time = (start_sec + end_sec) / 2
        
        fig.add_trace(go.Scatter(
            x=[start_sec, end_sec, end_sec, start_sec],
            y=[emotion_score - 0.4, emotion_score - 0.4, emotion_score + 0.4, emotion_score + 0.4],
            fill="toself",
            fillcolor=color,
            mode="lines",
            line=dict(width=0),
            name=f"{speaker} - {EMOTION_LABELS_CN.get(emotion, emotion)}",
            text=f"{speaker}<br>{EMOTION_LABELS_CN.get(emotion, emotion)}<br>置信度: {seg.get('emotion_confidence', 0):.2f}",
            hoverinfo="text"
        ))
    
    for emotion, color in EMOTION_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=EMOTION_LABELS_CN.get(emotion, emotion),
            showlegend=True
        ))
    
    fig.update_layout(
        title="情感时间轴",
        xaxis_title="时间 (秒)",
        yaxis_title="情感得分",
        yaxis=dict(
            tickvals=[-2, -1.5, -1, 0, 1.5, 2],
            ticktext=["愤怒", "恐惧", "悲伤", "中性", "惊讶", "开心"],
            range=[-2.5, 2.5]
        ),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_speaker_pie(speaker_stats):
    labels = [s['speaker'] for s in speaker_stats]
    values = [s['total_duration_ms'] for s in speaker_stats]
    colors = [SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i in range(len(labels))]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate="%{label}<br>时长: %{value/1000:.1f}秒<br>占比: %{percent}"
    )])
    
    fig.update_layout(
        title="说话人发言时长占比",
        height=400
    )
    
    return fig

def plot_emotion_distribution(speaker_emotions):
    emotions = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
    emotion_labels_cn = [EMOTION_LABELS_CN[e] for e in emotions]
    
    fig = go.Figure()
    
    for i, se in enumerate(speaker_emotions):
        speaker = se['speaker']
        counts = [ed['count'] for ed in se['emotions']]
        color = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]
        
        fig.add_trace(go.Bar(
            name=speaker,
            x=emotion_labels_cn,
            y=counts,
            marker_color=color,
            text=counts,
            textposition='auto'
        ))
    
    fig.update_layout(
        title="各说话人情感分布",
        xaxis_title="情感类别",
        yaxis_title="片段数",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_segments_table(segments):
    data = []
    for i, seg in enumerate(segments):
        data.append({
            "序号": i + 1,
            "开始时间": f"{seg['start_ms']/1000:.2f}s",
            "结束时间": f"{seg['end_ms']/1000:.2f}s",
            "时长": f"{(seg['end_ms'] - seg['start_ms'])/1000:.2f}s",
            "说话人": seg.get('speaker', 'Unknown'),
            "情感": EMOTION_LABELS_CN.get(seg.get('emotion', 'neutral'), seg.get('emotion', 'neutral')),
            "置信度": f"{seg.get('emotion_confidence', 0):.2f}"
        })
    return pd.DataFrame(data)

tab1, tab2, tab3 = st.tabs(["📊 单文件分析", "📦 批量处理", "🔍 对比分析"])

with tab1:
    st.header("上传音频文件")
    
    uploaded_file = st.file_uploader(
        "选择音频文件 (WAV/MP3, 最大50MB, 最长30分钟)",
        type=['wav', 'mp3'],
        key="single_upload"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.audio(uploaded_file)
        
        if st.button("开始分析", type="primary", key="analyze_btn"):
            with st.spinner("正在上传并创建分析任务..."):
                try:
                    result = upload_audio(uploaded_file)
                    task_id = result['task_id']
                    st.session_state['task_id'] = task_id
                    st.success(f"✅ 任务创建成功! 任务ID: {task_id}")
                except Exception as e:
                    st.error(f"上传失败: {str(e)}")
    
    if 'task_id' in st.session_state:
        task_id = st.session_state['task_id']
        
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status = get_task_status(task_id)
        while status['status'] in ['pending', 'processing']:
            status = get_task_status(task_id)
            progress = status.get('progress', 0)
            
            status_text = {
                'pending': '⏳ 等待处理...',
                'processing': '🔄 正在分析...'
            }.get(status['status'], status['status'])
            
            status_placeholder.info(f"{status_text} ({progress*100:.0f}%)")
            progress_bar.progress(progress)
            
            if status.get('error_message'):
                st.error(f"❌ 处理失败: {status['error_message']}")
                break
            
            time.sleep(2)
        
        if status['status'] == 'completed':
            status_placeholder.success("✅ 分析完成!")
            progress_bar.progress(1.0)
            
            with st.spinner("正在加载分析结果..."):
                try:
                    result_data = get_task_result(task_id)
                    summary_data = get_task_summary(task_id)
                    audio_bytes = get_audio_data(task_id)
                    y, sr = load_audio_from_bytes(audio_bytes)
                    quality_data = result_data.get('quality_assessment')
                    if not quality_data:
                        quality_data = get_task_quality(task_id)
                except Exception as e:
                    st.error(f"加载结果失败: {str(e)}")
                    st.stop()

            st.markdown("---")

            if quality_data:
                grade_colors = {
                    "优秀": {"bg": "#d4edda", "border": "#28a745", "text": "#155724", "badge": "#28a745"},
                    "良好": {"bg": "#d1ecf1", "border": "#17a2b8", "text": "#0c5460", "badge": "#17a2b8"},
                    "一般": {"bg": "#fff3cd", "border": "#ffc107", "text": "#856404", "badge": "#fd7e14"},
                    "较差": {"bg": "#f8d7da", "border": "#dc3545", "text": "#721c24", "badge": "#dc3545"}
                }
                grade = quality_data.get('grade', '一般')
                colors = grade_colors.get(grade, grade_colors["一般"])
                overall_score = quality_data.get('overall_score', 0)

                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {colors['bg']};
                            border: 2px solid {colors['border']};
                            border-radius: 12px;
                            padding: 16px 20px;
                            margin-bottom: 16px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                                <div style="display: flex; align-items: center; gap: 16px;">
                                    <div style="
                                        font-size: 32px;
                                        font-weight: bold;
                                        color: {colors['text']};
                                        min-width: 120px;
                                    ">
                                        {overall_score:.1f} 分
                                    </div>
                                    <div>
                                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                            <span style="
                                                background-color: {colors['badge']};
                                                color: white;
                                                padding: 4px 14px;
                                                border-radius: 20px;
                                                font-weight: bold;
                                                font-size: 16px;
                                            ">
                                                {grade}
                                            </span>
                                            <span style="font-size: 18px; font-weight: bold; color: {colors['text']};">
                                                🎵 音频质量评估
                                            </span>
                                        </div>
                                        <div style="color: {colors['text']}; font-size: 14px; opacity: 0.85;">
                                            综合评估音频可分析程度，帮助判断分析结果的可靠程度
                                        </div>
                                    </div>
                                </div>
                                <div style="width: 220px;">
                                    <div style="height: 12px; background: rgba(255,255,255,0.6); border-radius: 6px; overflow: hidden;">
                                        <div style="
                                            height: 100%;
                                            width: {min(max(overall_score, 0), 100)}%;
                                            background: linear-gradient(90deg, {colors['border']}, {colors['badge']});
                                            border-radius: 6px;
                                            transition: width 0.5s ease;
                                        "></div>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: 4px; color: {colors['text']}; opacity: 0.7;">
                                        <span>0</span><span>50</span><span>100</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("📊 查看各维度详情和改善建议", expanded=False):
                        dim_order = ['snr', 'clipping', 'speech_ratio', 'sample_rate_fitness']
                        dim_labels = {
                            'snr': '信噪比(SNR)',
                            'clipping': '削波检测',
                            'speech_ratio': '有效语音占比',
                            'sample_rate_fitness': '采样率适配度'
                        }
                        weights = {
                            'snr': '40%',
                            'clipping': '20%',
                            'speech_ratio': '25%',
                            'sample_rate_fitness': '15%'
                        }

                        for dim_key in dim_order:
                            dim = quality_data.get(dim_key)
                            if dim:
                                label = dim_labels.get(dim_key, dim.get('name', dim_key))
                                score = dim.get('score', 0)
                                raw_val = dim.get('raw_value')
                                unit = dim.get('unit', '')
                                weight = weights.get(dim_key, '')

                                if score >= 90:
                                    bar_color = "#28a745"
                                elif score >= 70:
                                    bar_color = "#17a2b8"
                                elif score >= 50:
                                    bar_color = "#fd7e14"
                                else:
                                    bar_color = "#dc3545"

                                val_text = ""
                                if raw_val is not None:
                                    if unit == 'dB':
                                        val_text = f" · {raw_val:.1f} {unit}"
                                    elif unit == '%':
                                        val_text = f" · {raw_val:.1f} {unit}"
                                    elif unit == 'Hz':
                                        val_text = f" · {int(raw_val)} {unit}"
                                    else:
                                        val_text = f" · {raw_val} {unit}"

                                st.markdown(
                                    f"""
                                    <div style="padding: 10px 4px; border-bottom: 1px solid #eee;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                            <div style="font-weight: 500; color: #333;">
                                                {label} <span style="font-size: 12px; color: #888; margin-left: 6px;">(权重 {weight})</span>
                                            </div>
                                            <div style="font-weight: bold; color: {bar_color};">
                                                {score:.0f} 分{val_text}
                                            </div>
                                        </div>
                                        <div style="height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden;">
                                            <div style="
                                                height: 100%;
                                                width: {score:.0f}%;
                                                background: {bar_color};
                                                border-radius: 4px;
                                                transition: width 0.4s ease;
                                            "></div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        suggestions = quality_data.get('suggestions', [])
                        if suggestions:
                            st.markdown("---")
                            st.markdown("#### 💡 改善建议")
                            for i, s in enumerate(suggestions, 1):
                                dim_name = s.get('dimension', '')
                                problem = s.get('problem', '')
                                suggestion = s.get('suggestion', '')
                                is_overall = dim_name == '综合评估'

                                if is_overall:
                                    st.warning(
                                        f"**⚠️ 警告：{problem}**\n\n👉 {suggestion}"
                                    )
                                else:
                                    st.info(
                                        f"**【{dim_name}】{problem}**\n\n👉 {suggestion}"
                                    )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总时长", f"{result_data['duration_ms']/1000:.1f} 秒")
            with col2:
                st.metric("语音片段数", f"{len(result_data['segments'])} 段")
            with col3:
                st.metric("说话人数", f"{len(summary_data['speaker_stats'])} 人")
            
            st.markdown("### 📈 可视化分析")
            
            fig_wave = plot_waveform_with_speakers(y, sr, result_data['segments'])
            st.plotly_chart(fig_wave, use_container_width=True)
            
            fig_emotion = plot_emotion_timeline(result_data['segments'], result_data['duration_ms'])
            st.plotly_chart(fig_emotion, use_container_width=True)
            
            st.markdown("### 📊 统计图表")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = plot_speaker_pie(summary_data['speaker_stats'])
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                fig_bar = plot_emotion_distribution(summary_data['speaker_emotions'])
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("### 📋 详细标注表格")
            
            df_segments = create_segments_table(result_data['segments'])
            
            col1, col2 = st.columns(2)
            with col1:
                speaker_filter = st.multiselect(
                    "筛选说话人",
                    options=sorted(set(df_segments['说话人'])),
                    default=sorted(set(df_segments['说话人']))
                )
            with col2:
                emotion_filter = st.multiselect(
                    "筛选情感",
                    options=sorted(set(df_segments['情感'])),
                    default=sorted(set(df_segments['情感']))
                )
            
            filtered_df = df_segments[
                df_segments['说话人'].isin(speaker_filter) & 
                df_segments['情感'].isin(emotion_filter)
            ]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("### 🎵 片段播放")
            
            selected_idx = st.selectbox(
                "选择要播放的片段序号",
                options=range(len(result_data['segments'])),
                format_func=lambda x: f"片段 {x+1}: {result_data['segments'][x].get('speaker', 'Unknown')} - {EMOTION_LABELS_CN.get(result_data['segments'][x].get('emotion', 'neutral'), result_data['segments'][x].get('emotion', 'neutral'))}"
            )
            
            if selected_idx is not None:
                seg = result_data['segments'][selected_idx]
                start_sample = int(seg['start_ms'] / 1000 * sr)
                end_sample = int(seg['end_ms'] / 1000 * sr)
                segment_audio = y[start_sample:end_sample]
                
                segment_bytes = io.BytesIO()
                sf.write(segment_bytes, segment_audio, sr, format='WAV')
                segment_bytes.seek(0)
                
                st.audio(segment_bytes, format='audio/wav')
                st.info(f"片段 {selected_idx + 1}: {seg.get('speaker', 'Unknown')} | {EMOTION_LABELS_CN.get(seg.get('emotion', 'neutral'), seg.get('emotion', 'neutral'))} | 置信度: {seg.get('emotion_confidence', 0):.2f}")
            
            st.markdown("### 📥 导出结果")
            
            col1, col2 = st.columns(2)
            with col1:
                json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "下载 JSON 结果",
                    data=json_str,
                    file_name=f"{task_id}_result.json",
                    mime="application/json"
                )
            with col2:
                srt_content = generate_srt(result_data)
                st.download_button(
                    "下载 SRT 字幕",
                    data=srt_content,
                    file_name=f"{task_id}_result.srt",
                    mime="text/plain"
                )
        
        elif status['status'] == 'failed':
            st.error(f"❌ 任务失败: {status.get('error_message', 'Unknown error')}")

with tab2:
    st.header("批量上传音频文件")
    
    uploaded_files = st.file_uploader(
        "选择多个音频文件 (最多10个, WAV/MP3, 单个最大50MB)",
        type=['wav', 'mp3'],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件")
        
        if len(uploaded_files) > 10:
            st.warning("⚠️ 最多只能上传10个文件，请减少选择")
        else:
            if st.button("开始批量分析", type="primary", key="batch_analyze_btn"):
                with st.spinner("正在上传批量文件..."):
                    try:
                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                        response = requests.post(f"{API_BASE_URL}/batch/upload", files=files)
                        batch_data = response.json()
                        batch_id = batch_data['batch_id']
                        st.session_state['batch_id'] = batch_id
                        st.success(f"✅ 批次任务创建成功! 批次ID: {batch_id}")
                    except Exception as e:
                        st.error(f"批量上传失败: {str(e)}")
    
    if 'batch_id' in st.session_state:
        batch_id = st.session_state['batch_id']
        
        st.markdown("### 批次处理进度")
        
        status_placeholder = st.empty()
        
        while True:
            try:
                batch_status = requests.get(f"{API_BASE_URL}/batch/{batch_id}/status").json()
                tasks = batch_status['tasks']
                
                status_df = pd.DataFrame([
                    {
                        "任务ID": t['task_id'],
                        "状态": {
                            'pending': '⏳ 等待中',
                            'processing': '🔄 处理中',
                            'completed': '✅ 已完成',
                            'failed': '❌ 失败'
                        }.get(t['status'], t['status']),
                        "进度": f"{t.get('progress', 0)*100:.0f}%",
                        "错误": t.get('error_message', '-')
                    }
                    for t in tasks
                ])
                
                status_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
                
                all_done = all(t['status'] in ['completed', 'failed'] for t in tasks)
                if all_done:
                    break
                
                time.sleep(2)
            except Exception as e:
                st.error(f"获取批次状态失败: {str(e)}")
                break

with tab3:
    st.header("对话对比分析")
    st.markdown("选择两个已完成的分析任务进行横向对比，生成结构化对比报告。")
    
    try:
        completed_tasks = list_completed_tasks()
    except Exception as e:
        st.error(f"获取任务列表失败: {str(e)}")
        completed_tasks = []

    if not completed_tasks:
        st.info("暂无已完成的任务，请先在「单文件分析」或「批量处理」中完成至少两个任务。")
    else:
        task_options = {f"{t['filename']} ({t['task_id'][:8]}...)": t['task_id'] for t in completed_tasks}
        
        col1, col2 = st.columns(2)
        with col1:
            selected_label1 = st.selectbox(
                "选择任务1",
                options=list(task_options.keys()),
                key="compare_task1"
            )
        with col2:
            selected_label2 = st.selectbox(
                "选择任务2",
                options=list(task_options.keys()),
                key="compare_task2"
            )

        if st.button("开始对比分析", type="primary", key="compare_btn"):
            task1_id = task_options[selected_label1]
            task2_id = task_options[selected_label2]

            if task1_id == task2_id:
                st.warning("⚠️ 请选择两个不同的任务进行对比")
            else:
                with st.spinner("正在生成对比报告..."):
                    try:
                        report = compare_tasks(task1_id, task2_id)
                    except Exception as e:
                        st.error(f"对比分析失败: {str(e)}")
                        st.stop()

                st.markdown("---")
                st.subheader("📋 对比报告")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**任务1**: {report['task1_filename']}\n\nID: `{report['task1_id']}`")
                with col2:
                    st.info(f"**任务2**: {report['task2_filename']}\n\nID: `{report['task2_id']}`")

                st.markdown("---")

                st.markdown("### 1️⃣ 发言主导度对比")
                sd = report['speaker_dominance']
                speakers_data = []
                for sp in sd['speakers']:
                    speakers_data.append({
                        "说话人": sp['speaker'],
                        "任务1占比(%)": f"{sp['percentage_task1']:.2f}" if sp['percentage_task1'] is not None else "-",
                        "任务2占比(%)": f"{sp['percentage_task2']:.2f}" if sp['percentage_task2'] is not None else "-"
                    })
                st.table(pd.DataFrame(speakers_data))
                st.markdown(f"""
                - 任务1发言最多者: **{sd['max_speaker_task1']}** ({sd['max_percentage_task1']:.2f}%)
                - 任务2发言最多者: **{sd['max_speaker_task2']}** ({sd['max_percentage_task2']:.2f}%)
                - 结论: <span style="{get_conclusion_color(sd['conclusion'])}">{sd['conclusion']}</span>
                """, unsafe_allow_html=True)

                st.markdown("---")

                st.markdown("### 2️⃣ 情感倾向对比")
                se = report['sentiment']
                sentiment_df = pd.DataFrame([
                    {"维度": "情感净值 (-1~1)", "任务1": f"{se['sentiment_score_task1']:.4f}", "任务2": f"{se['sentiment_score_task2']:.4f}"}
                ])
                st.table(sentiment_df)
                st.markdown(f"""
                - 结论: <span style="{get_conclusion_color(se['conclusion'])}">{se['conclusion']}</span>
                """, unsafe_allow_html=True)

                st.markdown("---")

                st.markdown("### 3️⃣ 对话活跃度对比")
                ac = report['activity']
                activity_df = pd.DataFrame([
                    {"指标": "总片段数", "任务1": ac['total_segments_task1'], "任务2": ac['total_segments_task2']},
                    {"指标": "平均每轮时长(ms)", "任务1": ac['avg_segment_duration_ms_task1'], "任务2": ac['avg_segment_duration_ms_task2']}
                ])
                st.table(activity_df)
                st.markdown(f"""
                - 结论: <span style="{get_conclusion_color(ac['conclusion'])}">{ac['conclusion']}</span>
                (片段数多且平均时长短视为节奏更快)
                """, unsafe_allow_html=True)

                st.markdown("---")

                st.markdown("### 4️⃣ 打断频率对比")
                ic = report['interruption']
                interruption_df = pd.DataFrame([
                    {"指标": "打断次数", "任务1": ic['interruption_count_task1'], "任务2": ic['interruption_count_task2']},
                    {"指标": "总片段数", "任务1": ic['total_segments_task1'], "任务2": ic['total_segments_task2']},
                    {"指标": "打断比率", "任务1": f"{ic['interruption_rate_task1']:.4f}", "任务2": f"{ic['interruption_rate_task2']:.4f}"}
                ])
                st.table(interruption_df)
                st.markdown(f"""
                - 结论: <span style="{get_conclusion_color(ic['conclusion'])}">{ic['conclusion']}</span>
                (比率高表示对话冲突更激烈)
                """, unsafe_allow_html=True)

                st.markdown("---")

                st.subheader("📊 结论汇总")
                summary_data = [
                    {"对比维度": "发言主导度", "结论": sd['conclusion']},
                    {"对比维度": "情感倾向", "结论": se['conclusion']},
                    {"对比维度": "对话活跃度", "结论": ac['conclusion']},
                    {"对比维度": "打断频率", "结论": ic['conclusion']}
                ]
                summary_df = pd.DataFrame(summary_data)

                def highlight_conclusion(val):
                    color = "#28a745" if val == "task1占优" else ("#dc3545" if val == "task2占优" else "#6c757d")
                    return f"color: {color}; font-weight: bold;"

                styled_summary = summary_df.style.map(highlight_conclusion, subset=["结论"])
                st.dataframe(styled_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("语音情感识别与说话人分离分析平台 | Powered by FastAPI + Streamlit")
