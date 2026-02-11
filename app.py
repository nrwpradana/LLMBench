import streamlit as st
import subprocess
import json
import pandas as pd
from pathlib import Path
import time
import threading
from queue import Queue, Empty
import plotly.express as px
import os

st.set_page_config(page_title="SEA-HELM Bench", layout="wide")
st.title("🌏 SEA-HELM Evaluation Dashboard")
st.markdown("**Benchmark LLM untuk bahasa-bahasa Southeast Asia**")

# Sidebar
with st.sidebar:
    st.header("🚀 Evaluation Settings")
    model_name = st.text_input("Model Name / HF ID", value="aisingapore/SEA-LION-7B-Instruct", help="Contoh: meta-llama/Llama-3.1-8B-Instruct")
    output_dir = st.text_input("Output Directory", value="./results", help="Folder tempat simpan hasil")
    tasks = st.multiselect("Tasks", ["seahelm", "vision"], default=["seahelm"])
    model_type = st.selectbox("Model Type", ["vllm", "hf", "openai"], index=0)
    tensor_parallel = st.slider("Tensor Parallel Size (GPU)", 1, 8, value="auto")
    run_number = st.number_input("Run Number (0-7)", 0, 7, 0)
    
    st.header("🔑 API Keys (Required)")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Untuk GPT-4 sebagai judge di MT-Bench.")
    hf_token = st.text_input("HuggingFace Token", type="password", help="Untuk akses gated models.")
    
    if st.button("🔥 Start Evaluation", type="primary", use_container_width=True):
        if not model_name or not openai_key:
            st.error("Masukkan nama model & OpenAI API Key!")
        else:
            # Set env vars
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["HF_TOKEN"] = hf_token if hf_token else ""
            
            st.session_state["running"] = True
            st.session_state["log_queue"] = Queue()
            st.session_state["output_dir"] = output_dir
            st.session_state["model_name"] = model_name

            def run_evaluation():
                cmd = [
                    "uv", "run", "seahelm_evaluation.py",
                    "--tasks", *tasks,
                    "--output_dir", output_dir,
                    "--model_type", model_type,
                    "--model_name", model_name,
                    "--model_args", f"enable_prefix_caching=True,tensor_parallel_size={tensor_parallel}",
                    "--run_number", str(run_number)
                ]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=Path("..").resolve()  # kembali ke root SEA-HELM
                )
                for line in process.stdout:
                    st.session_state["log_queue"].put(line.strip())
                process.wait()
                st.session_state["running"] = False
                st.session_state["finished"] = True

            threading.Thread(target=run_evaluation, daemon=True).start()

# Main area
tab1, tab2, tab3 = st.tabs(["📊 Live Evaluation", "📋 Results", "🏆 Leaderboard"])

with tab1:
    if st.session_state.get("running", False):
        st.info("Evaluation sedang berjalan... (bisa memakan waktu 30 menit – beberapa jam tergantung model & GPU)")
        
        log_container = st.empty()
        while st.session_state.get("running", False):
            try:
                while True:
                    line = st.session_state["log_queue"].get_nowait()
                    log_container.text(line)
            except Empty:
                time.sleep(0.5)
                st.rerun()
    elif st.session_state.get("finished", False):
        st.success("✅ Evaluation selesai!")
        st.balloons()
    else:
        st.info("Klik 'Start Evaluation' di sidebar untuk memulai.")

with tab2:
    if Path(output_dir).exists():
        st.subheader("Hasil Terbaru")
        # Load semua JSON hasil
        results = []
        for run_dir in Path(output_dir).glob("run_*"):
            for json_file in run_dir.glob("**/*.json"):
                if "results" in json_file.name.lower():
                    with open(json_file) as f:
                        data = json.load(f)
                        # Ambil summary score (sesuaikan dengan struktur SEA-HELM JSON)
                        avg_score = data.get("overall_score", 0) or data.get("average", 0)
                        results.append({
                            "Model": st.session_state.get("model_name", "Unknown"),
                            "Run": run_dir.name,
                            "Task": data.get("task", "seahelm"),
                            "Avg Score": round(avg_score, 4),
                            "Details": json_file.name
                        })
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Chart per task
            fig = px.bar(df, x="Task", y="Avg Score", color="Run", title="Score per Task")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Belum ada hasil. Jalankan evaluation dulu.")
    else:
        st.info("Output directory belum ada. Jalankan evaluation dulu.")

with tab3:
    st.subheader("Local Leaderboard")
    st.caption("Bandingkan dengan official SEA-HELM Leaderboard")
    
    # Load semua model yang sudah pernah dievaluasi (simpan di file JSON sederhana)
    leaderboard_file = Path("local_leaderboard.json")
    if leaderboard_file.exists():
        lb = pd.read_json(leaderboard_file)
    else:
        lb = pd.DataFrame(columns=["Model", "Avg SEA Score", "Burmese", "Filipino", "Indonesian", "Malay", "Tamil", "Thai", "Vietnamese", "English", "Run Date"])
    
    # Tambah hasil terbaru ke leaderboard (contoh logic sederhana – expand sesuai struktur JSON SEA-HELM)
    if st.session_state.get("finished", False):
        # Parse hasil terbaru (asumsi satu run, expand untuk multi)
        new_entry = {
            "Model": st.session_state.get("model_name", "Unknown"),
            "Avg SEA Score": sum(r["Avg Score"] for r in results) / len(results) if results else 0,  # Contoh agregasi
            # Tambah per bahasa kalau ada di JSON: data["scores"]["burmese"] dll.
            "Run Date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        lb = pd.concat([lb, pd.DataFrame([new_entry])], ignore_index=True)
        lb = lb.drop_duplicates(subset=["Model"])
        lb.to_json(leaderboard_file, orient="records", indent=2)
        st.session_state["finished"] = False  # Reset
    
    if not lb.empty:
        lb_sorted = lb.sort_values("Avg SEA Score", ascending=False)
        st.dataframe(lb_sorted.style.highlight_max(axis=0, subset=["Avg SEA Score"]), use_container_width=True)
    
    st.markdown("**Official Leaderboard** → [leaderboard.sea-lion.ai](https://leaderboard.sea-lion.ai/)")
    st.info("Untuk submit ke leaderboard resmi, jalankan 8 run (dengan run_number 0-7) lalu kirim aggregated JSON ke AISG (lihat docs di repo).")

# Footer
st.divider()
st.caption("Built with ❤️ for SEA-HELM • Powered by Streamlit + vLLM")
