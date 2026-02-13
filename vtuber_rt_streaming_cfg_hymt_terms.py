# vtuber_rt_streaming_cfg.py
# Win + Conda: Demucs (quasi-realtime vocals) + ufal Whisper-Streaming + faster-whisper
# OBS: read obs_final.txt / obs_partial.txt via "Read from file"

import time
import queue
import threading
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio
from rich.console import Console
from scipy.signal import resample_poly
import torch

# Demucs-infer (python API)
from demucs_infer.pretrained import get_model as demucs_get_model
from demucs_infer.apply import apply_model as demucs_apply_model
from sympy import false

# ufal whisper_streaming
from whisper_online import FasterWhisperASR, OnlineASRProcessor

from transformers import AutoTokenizer, AutoModelForCausalLM

console = Console()

# =========================
# CONFIG (EDIT HERE)
# =========================

# --- Audio capture (WASAPI loopback) ---
DEVICE_INDEX = 28          # <<< set this from list_loopback.py
SRC_SR = 48000             # loopback sample rate (usually 48000)
SRC_CHANNELS = 2
FRAMES_PER_BUFFER_SEC = 0.05   # 0.05~0.20; smaller = lower latency but higher CPU overhead

# --- Enable Demucs vocals separation (recommended for BGM-heavy streams) ---
ENABLE_DEMUCS = True
DEMUCS_MODEL_NAME = "mdx_q"     # try: "mdx_q" (faster) or "htdemucs_ft" (better, heavier)
DEMUCS_CHUNK_SEC = 8.0          # separation window size (seconds)
DEMUCS_HOP_SEC = 1.0            # how often we produce new vocals audio (seconds)
DEMUCS_DEVICE = "cuda"          # "cuda" or "cpu" (auto-fallback if cuda not available)

# --- Whisper-Streaming (ufal) ---
WHISPER_MODEL_NAME = "large-v3" # e.g. "large-v3", "medium", "distil-large-v3"
LANG = "ja"                     # source language
TASK = "transcribe"             # "transcribe" or "translate" (translate is Whisper direct)
MIN_CHUNK_SIZE_SEC = 1.0        # how often we feed audio into online processor

# Buffer trimming (only applied if OnlineASRProcessor exposes attributes)
BUFFER_TRIMMING = "segment"     # "segment" or "sentence"
BUFFER_TRIMMING_SEC = 16.0

# --- Output files for OBS ---
OUT_FINAL = "obs_final.txt"
OUT_PARTIAL = "obs_partial.txt"

# --- Logging ---
PRINT_EVERY_SEC = 1.0           # console print cadence

# --- Translation model ---
# --- Sentence-level translation (CTranslate2 + NLLB) ---
# --- Translation model ---
ENABLE_ZH_TRANSLATION = True

# Use LLM translator (HY-MT1.5)
HYMT_MODEL_ID = "tencent/HY-MT1.5-1.8B"   # 先用 1.8B；你之后可改 "tencent/HY-MT1.5-7B"（需量化更合适）
HYMT_DEVICE = "cuda"                      # "cuda" or "cpu"
HYMT_MAX_NEW_TOKENS = 256
HYMT_TEMPERATURE = 0.2
HYMT_TOP_P = 0.7

HYMT_PROMPT = (
    "Translate the following Japanese into Simplified Chinese.\n"
    "Requirements:\n"
    "- Keep the original meaning; do not add or omit information.\n"
    "- Keep names (people/organizations) as-is.\n"
    # "- Keep all placeholders like __TERM_000__ exactly unchanged.\n"
    "- Use natural spoken Chinese suitable for live subtitles.\n"
    "- Output ONLY the translation.\n\n"
    "Japanese:\n{jp}\n\nChinese:"
)

ZH_OUT_FINAL = "obs_zh_final.txt"
ZH_OUT_PART = "obs_zh_part.txt"

# 句子提交条件：仅靠标点+长度（你不想用VAD gate）
SENT_END_PUNCS = ("。", "！", "？", "!", "?", ".", "…")
SENT_MAX_CHARS = 30
SENT_MIN_CHARS = 3




# --- Force terms / glossary (Iroha-focused) ---
# Strategy: protect matched JP terms with placeholders before LLM translation,
# then restore placeholders to the desired ZH terms after translation.
ENABLE_FORCE_TERMS = False

FORCE_TERMS = {
    # ===== 風真いろは / holoX core =====
    "風真いろは": "风真伊吕波",
    "風真伊呂波": "风真伊吕波",
    "かざまいろは": "风真伊吕波",
    "いろはちゃん": "伊吕波酱",
    "いろは": "伊吕波",  # optional; if too aggressive, delete this line

    # Persona / address style (keep some JP flavor)
    "ござる": "ござる",
    "でござる": "でござる",
    "皆殿": "诸位",
    "拙者": "在下",
    "用心棒": "护卫",
    "侍": "武士",
    "忍者": "忍者",

    # Greetings / common phrases
    "こんいろは": "晚安伊吕波",
    "おはござる": "早安ござる",
    "こんござる": "晚上好ござる",
    "おつござる": "辛苦了ござる",
    "おつかれさまでござる": "辛苦了ござる",
    "おつかれさまでした": "辛苦了",
    "おつかれさま": "辛苦了",
    "おつかれ": "辛苦了",
    "おつ": "辛苦了",

    # holoX org
    "秘密結社holoX": "秘密结社holoX",
    "秘密結社 holoX": "秘密结社holoX",
    "holoX": "holoX",
    "ホロックス": "holoX",
    "ホロＸ": "holoX",

    # Fanbase / nicknames
    "かざま隊": "かざま队",
    "かざま隊員": "かざま队队员",
    "ﾁｬｷ丸": "チャキ丸",
    "チャキ丸": "チャキ丸",
    "ぽこべぇ": "ぽこべぇ",
    "ママ殿": "ママ殿",
    "パパ殿": "パパ殿",

    # holoX members (common CN names; adjust to your preferred naming)
    "ラプラス・ダークネス": "拉普拉斯·暗黑",
    "La+ Darknesss": "拉普拉斯·暗黑",
    "鷹嶺ルイ": "鹰岭琉依",
    "Takane Lui": "鹰岭琉依",
    "博衣こより": "博衣小夜璃",
    "Hakui Koyori": "博衣小夜璃",
    "沙花叉クロヱ": "沙花叉克萝伊",
    "Sakamata Chloe": "沙花叉克萝伊",

    # Hololive / streaming common terms
    "ホロライブ": "Hololive",
    "hololive": "Hololive",
    "ホロメン": "Hololive成员",
    "配信": "直播",
    "生放送": "直播",
    "雑談": "杂谈",
    "歌枠": "歌回",
    "耐久": "耐久",
    "同時視聴": "同看",
    "切り抜き": "切片",
    "アーカイブ": "回放",
    "概要欄": "简介栏",
    "コメント": "评论",
    "チャット": "聊天",

    # Donations / membership
    "スパチャ": "SC",
    "スーパーチャット": "SC",
    "メンバーシップ": "会员",
    "メン限": "会员限定",

    # Engagement
    "高評価": "点赞",
    "チャンネル登録": "订阅",
    "通知": "通知",

    # Radio / letters
    "ラジオネーム": "电台昵称",
    "お便り": "来信",
    "成人式": "成人礼",
    "成人": "成年",
    "お年玉": "压岁钱",
    "お正月": "新年",
    "正月": "新年",
    "リーク": "爆料",
}

# =========================
# END CONFIG
# =========================


# =========================
# Term protection helpers
# =========================
import re as _re

_TERM_PATTERN = None
_FORCE_KEYS_SORTED = None

def _build_term_pattern(force_terms: dict[str, str]):
    # Longest-first to avoid partial matches (e.g., おつかれさまでした before おつ)
    keys = sorted(force_terms.keys(), key=len, reverse=True)
    escaped = [_re.escape(k) for k in keys if k]
    if not escaped:
        return None, keys
    return _re.compile("|".join(escaped)), keys

def protect_terms_with_placeholders(jp: str):
    """
    Replace FORCE_TERMS matches in JP with placeholders __TERM_000__, __TERM_001__, ...
    Returns (protected_text, placeholder_to_zh)
    """
    global _TERM_PATTERN, _FORCE_KEYS_SORTED
    if not ENABLE_FORCE_TERMS or not FORCE_TERMS:
        return jp, {}

    if _TERM_PATTERN is None:
        _TERM_PATTERN, _FORCE_KEYS_SORTED = _build_term_pattern(FORCE_TERMS)

    if _TERM_PATTERN is None:
        return jp, {}

    placeholder_to_zh: dict[str, str] = {}
    used = 0

    def repl(m: _re.Match):
        nonlocal used
        k = m.group(0)
        zh = FORCE_TERMS.get(k, k)
        ph = f"__TERM_{used:03d}__"
        used += 1
        placeholder_to_zh[ph] = zh
        return ph

    protected = _TERM_PATTERN.sub(repl, jp)
    return protected, placeholder_to_zh

def restore_placeholders(text: str, placeholder_to_zh: dict[str, str]) -> str:
    for ph, zh in placeholder_to_zh.items():
        text = text.replace(ph, zh)
    return text



def to_float32(x):
    return x.astype(np.float32, copy=False)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    return x.mean(axis=1)


def resample_any(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)

    # common exact ratios
    if sr_in == 48000 and sr_out == 44100:
        return resample_poly(x, 147, 160).astype(np.float32)
    if sr_in == 44100 and sr_out == 16000:
        return resample_poly(x, 160, 441).astype(np.float32)
    if sr_in == 48000 and sr_out == 16000:
        return resample_poly(x, 1, 3).astype(np.float32)

    # generic approx (works fine but not perfect for all pairs)
    up = 100
    down = int(round(sr_in * up / sr_out))
    return resample_poly(x, up, down).astype(np.float32)


class LoopbackCapture:
    def __init__(self, device_index: int, rate=48000, channels=2, frames_per_buffer=4800):
        self.device_index = device_index
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer

        self.p = pyaudio.PyAudio()
        self.q = queue.Queue(maxsize=120)
        self.stream = None
        self._stop = threading.Event()

    def start(self):
        def callback(in_data, frame_count, time_info, status):
            if self._stop.is_set():
                return (None, pyaudio.paComplete)
            try:
                self.q.put_nowait(in_data)
            except queue.Full:
                pass
            return (None, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=callback,
        )
        self.stream.start_stream()

    def read(self, timeout=1.0):
        try:
            data = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        x = np.frombuffer(data, dtype=np.float32)
        x = x.reshape(-1, self.channels)
        return x

    def stop(self):
        self._stop.set()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


class DemucsStreamer:
    """
    Quasi-realtime Demucs:
    - ring buffer at demucs_sr (usually 44100)
    - every hop_sec, run separation on last chunk_sec
    - output last hop_sec of vocals (stereo)
    """
    def __init__(self, model_name="mdx_q", device="cuda", chunk_sec=8.0, hop_sec=0.8):
        self.model = demucs_get_model(model_name)
        self.model.eval()

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.model.to("cuda")
        else:
            self.device = "cpu"
            self.model.to("cpu")

        self.sr = getattr(self.model, "samplerate", 44100)

        self.chunk_sec = float(chunk_sec)
        self.hop_sec = float(hop_sec)
        self.chunk_n = int(self.chunk_sec * self.sr)
        self.hop_n = int(self.hop_sec * self.sr)

        self.ring = deque()
        self.max_n = self.chunk_n
        self._hop_acc = 0

        self.vocals_idx = -1
        if hasattr(self.model, "sources"):
            try:
                self.vocals_idx = list(self.model.sources).index("vocals")
            except Exception:
                self.vocals_idx = -1

    def push_mix(self, mix_stereo: np.ndarray):
        for v in mix_stereo:
            self.ring.append(v)
        while len(self.ring) > self.max_n:
            self.ring.popleft()
        self._hop_acc += mix_stereo.shape[0]

    def ready(self):
        return len(self.ring) >= self.chunk_n and self._hop_acc >= self.hop_n

    @torch.no_grad()
    def pop_vocals_hop(self) -> np.ndarray | None:
        if not self.ready():
            return None
        self._hop_acc = 0

        buf = np.array(self.ring, dtype=np.float32)     # [T,2]
        wav = torch.from_numpy(buf.T).unsqueeze(0)      # [1,2,T]

        sources = demucs_apply_model(self.model, wav, device=self.device)  # [1,S,C,T]
        if sources.ndim != 4:
            return None

        if self.vocals_idx >= 0:
            v = sources[0, self.vocals_idx]  # [C,T]
        else:
            v = sources[0, -1]               # fallback: last stem

        v = v.detach().cpu().numpy().T       # [T,2]
        out = v[-self.hop_n:, :]
        return out.astype(np.float32, copy=False)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    # Basic checks
    if DEVICE_INDEX is None:
        raise RuntimeError("Set DEVICE_INDEX in CONFIG.")

    frames_per_buffer = int(SRC_SR * FRAMES_PER_BUFFER_SEC)

    # 1) Loopback capture
    cap = LoopbackCapture(
        device_index=int(DEVICE_INDEX),
        rate=int(SRC_SR),
        channels=int(SRC_CHANNELS),
        frames_per_buffer=frames_per_buffer,
    )

    # 2) Demucs streamer
    demucs = None
    if ENABLE_DEMUCS:
        console.print(f"[cyan]Loading Demucs: {DEMUCS_MODEL_NAME} ...[/cyan]")
        demucs = DemucsStreamer(
            model_name=DEMUCS_MODEL_NAME,
            device=DEMUCS_DEVICE,
            chunk_sec=DEMUCS_CHUNK_SEC,
            hop_sec=DEMUCS_HOP_SEC,
        )
        console.print(f"[cyan]Demucs sr={demucs.sr} chunk={DEMUCS_CHUNK_SEC}s hop={DEMUCS_HOP_SEC}s device={demucs.device}[/cyan]")

    # 3) Whisper-Streaming + faster-whisper backend
    console.print("[cyan]Init Whisper-Streaming (ufal) + faster-whisper backend...[/cyan]")
    asr = FasterWhisperASR(LANG, WHISPER_MODEL_NAME)
    if TASK == "translate":
        asr.set_translate_task()

    online = OnlineASRProcessor(asr)

    hymt_tok = None
    hymt_model = None
    zh_segments = []
    sentence_buf = ""

    if ENABLE_ZH_TRANSLATION:
        console.print("[cyan]Init HY-MT (LLM) for JA->ZH...[/cyan]")
        hymt_tok = AutoTokenizer.from_pretrained(HYMT_MODEL_ID, trust_remote_code=True)
        hymt_model = AutoModelForCausalLM.from_pretrained(
            HYMT_MODEL_ID,
            device_map="auto" if HYMT_DEVICE == "cuda" and torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if HYMT_DEVICE == "cuda" and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        hymt_model.eval()
        write_text(ZH_OUT_FINAL, "")

    def is_sentence_complete(s: str) -> bool:
        t = s.strip()
        if not t:
            return False
        if len(t) >= SENT_MAX_CHARS:
            return True
        return t.endswith(SENT_END_PUNCS)


    def hymt_ja2zh(text: str) -> str:
        if hymt_tok is None or hymt_model is None:
            return ""
        jp = (text or "").strip()
        if not jp:
            return ""

        # --- protect forced terms with placeholders ---
        protected_jp, ph_map = protect_terms_with_placeholders(jp)

        prompt = HYMT_PROMPT.format(jp=protected_jp)
        messages = [{"role": "user", "content": prompt}]

        # Most robust: template -> text -> tokenize -> tensors
        prompt_text = hymt_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = hymt_tok(prompt_text, return_tensors="pt").to(hymt_model.device)

        gen_kwargs = dict(
            max_new_tokens=HYMT_MAX_NEW_TOKENS,
            temperature=HYMT_TEMPERATURE,
            top_p=HYMT_TOP_P,
            do_sample=(HYMT_TEMPERATURE > 0),
        )

        with torch.no_grad():
            out = hymt_model.generate(**inputs, **gen_kwargs)

        decoded = hymt_tok.decode(out[0], skip_special_tokens=True)

        # Extract only after "Chinese:" (avoid echoing prompt/Japanese)
        marker = "\nChinese:"
        if marker in decoded:
            zh = decoded.split(marker, 1)[1].strip()
        elif "Chinese:" in decoded:
            zh = decoded.split("Chinese:", 1)[1].strip()
        else:
            zh = decoded.strip().splitlines()[-1].strip()

        # --- restore forced terms ---
        if ph_map:
            zh = restore_placeholders(zh, ph_map)

        return zh

    # optional trimming knobs
    if hasattr(online, "buffer_trimming"):
        online.buffer_trimming = BUFFER_TRIMMING
    if hasattr(online, "buffer_trimming_sec"):
        online.buffer_trimming_sec = BUFFER_TRIMMING_SEC

    # 4) Output init
    write_text(OUT_FINAL, "")
    write_text(OUT_PARTIAL, "")

    # 5) Streaming loop
    cap.start()
    console.print("[green]Running. Ctrl+C to stop.[/green]")

    feed_sr = 16000
    pending_16k = np.zeros((0,), dtype=np.float32)
    min_chunk_n = int(MIN_CHUNK_SIZE_SEC * feed_sr)

    final_text = ""
    final_segments = []          # list[str]
    seen_keys = set()            # set[(beg,end,text)]
    last_print = time.time()

    try:
        while True:
            x = cap.read(timeout=1.0)
            if x is None:
                continue

            mix = to_stereo(to_float32(x))  # [n,2] @ SRC_SR

            if demucs is not None:
                mix_ds = resample_any(mix, SRC_SR, demucs.sr)
                demucs.push_mix(mix_ds)

                if not demucs.ready():
                    continue

                vocals = demucs.pop_vocals_hop()
                if vocals is None:
                    continue

                v_mono = to_mono(vocals)
                chunk_16k = resample_any(v_mono, demucs.sr, feed_sr)
            else:
                m_mono = to_mono(mix)
                chunk_16k = resample_any(m_mono, SRC_SR, feed_sr)

            pending_16k = np.concatenate([pending_16k, chunk_16k])

            if pending_16k.shape[0] < min_chunk_n:
                continue

            a = pending_16k[:min_chunk_n]
            pending_16k = pending_16k[min_chunk_n:]

            online.insert_audio_chunk(a)
            o = online.process_iter()
            if o is None:
                continue

            # ---- 1) normalize output to list of (beg,end,text)
            items = []

            def push_item(it):
                if not isinstance(it, tuple) or len(it) < 3:
                    return
                beg, end, txt = it[0], it[1], it[2]
                if beg is None or end is None:
                    return
                txt = (txt or "").strip()
                if not txt:
                    return
                items.append((float(beg), float(end), txt))

            if isinstance(o, tuple):
                # could be (beg,end,text) OR (emission_time,beg,end,text) depending on version
                if len(o) >= 4 and isinstance(o[1], (float, np.floating)) and isinstance(o[2], (float, np.floating)):
                    # (t, beg, end, text)
                    push_item((o[1], o[2], o[3]))
                elif len(o) >= 3:
                    push_item((o[0], o[1], o[2]))
            elif isinstance(o, list):
                for it in o:
                    if isinstance(it, tuple) and len(it) >= 4 and isinstance(it[1], (float, np.floating)) and isinstance(it[2], (float, np.floating)):
                        push_item((it[1], it[2], it[3]))  # (t,beg,end,text)
                    elif isinstance(it, tuple) and len(it) >= 3:
                        push_item((it[0], it[1], it[2]))
            else:
                # unknown shape; ignore safely
                items = []

            # ---- 2) append only NEW committed segments (time-keyed)
            appended_any = False
            for beg, end, txt in items:
                key = (round(beg, 2), round(end, 2), txt)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                final_segments.append(txt)
                appended_any = True

                # ====== NEW: sentence buffering for translation ======
                if ENABLE_ZH_TRANSLATION:
                    # 累积稳定片段到句子缓冲
                    if sentence_buf:
                        sentence_buf += " " + txt
                    else:
                        sentence_buf = txt
                    sentence_buf = sentence_buf.strip()

                    # 句子完成 or 太长 -> 翻译提交
                    if is_sentence_complete(sentence_buf) and len(sentence_buf) >= SENT_MIN_CHARS:
                        zh = hymt_ja2zh(sentence_buf)
                        if zh:
                            zh_segments.append(zh)
                            write_text(ZH_OUT_FINAL, "\n".join(zh_segments))
                            write_text(ZH_OUT_PART,zh)
                        sentence_buf = ""

            # ---- 3) build outputs
            final_text = " ".join(final_segments).strip()

            # PART: show latest returned text (even if already seen), else empty
            if items:
                partial = items[-1][2]
            else:
                partial = ""

            write_text(OUT_FINAL, final_text)
            write_text(OUT_PARTIAL, partial)


            now = time.time()
            if now - last_print >= PRINT_EVERY_SEC:
                console.print(f"[white]FINAL:[/white] {final_text[-120:]}")
                console.print(f"[yellow]PART:[/yellow] {partial[-120:]}")
                last_print = now

    except KeyboardInterrupt:
        console.print("\n[red]Stopping...[/red]")
    finally:
        cap.stop()
        try:
            online.finish()
        except Exception:
            pass
        write_text(OUT_PARTIAL, "")


if __name__ == "__main__":
    main()
