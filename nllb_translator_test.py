import os
import re
import time
import pyperclip
import ctranslate2
from transformers import AutoTokenizer

# =====================
# CONFIG (edit here)
# =====================
CT2_DIR = os.path.join(os.path.dirname(__file__), "nllb_ct2")  # default: ./nllb_ct2
HF_TOKENIZER = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "jpn_Jpan"
TGT_LANG = "zho_Hans"

DEVICE = "cuda"   # "cuda" or "cpu"
MAX_LEN = 384     # max decoding length (increase if you paste very long text)
# =====================

def normalize_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s

def split_sentences_ja(text: str):
    """
    Very lightweight sentence split:
    - split on 。！？ plus newlines
    """
    text = text.replace("\r\n", "\n")
    # keep delimiter
    parts = re.split(r"([。！？!?])", text)
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if p in ("。", "！", "？", "!", "?"):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())

    # also split big chunks by newline (but keep non-empty)
    final = []
    for s in out:
        for line in s.split("\n"):
            line = line.strip()
            if line:
                final.append(line)
    return final

def translate_nllb(translator, tokenizer, text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    tokenizer.src_lang = SRC_LANG
    enc = tokenizer(text, return_tensors=None)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    res = translator.translate_batch(
        [tokens],
        target_prefix=[[TGT_LANG]],
        max_decoding_length=MAX_LEN,
    )
    out_tokens = res[0].hypotheses[0]
    zh = tokenizer.convert_tokens_to_string(out_tokens)
    return normalize_text(zh)

def main():
    print("=== NLLB CT2 quick test ===")
    print(f"CT2_DIR: {CT2_DIR}")
    print("Copy Japanese text to clipboard, then press Enter.\n"
          "Or type/paste into console and end with an empty line.\n")

    try:
        input("Press Enter to read from clipboard (recommended), or type 'manual' then Enter: ")
        mode = "clip"
    except Exception:
        mode = "clip"

    text = ""
    if mode == "clip":
        text = pyperclip.paste()
        text = normalize_text(text)
        if not text:
            print("Clipboard is empty. Switch to manual input.")
            mode = "manual"

    if mode == "manual":
        print("Paste text now. End with an empty line:")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        text = normalize_text("\n".join(lines))

    if not text:
        print("No input text. Exit.")
        return

    print("\n--- INPUT (first 300 chars) ---")
    print(text[:300] + ("..." if len(text) > 300 else ""))

    print("\nLoading tokenizer & translator...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER)
    translator = ctranslate2.Translator(CT2_DIR, device=DEVICE)
    print(f"Loaded in {time.time()-t0:.2f}s\n")

    # 1) Whole paragraph translation
    print("=== WHOLE TEXT TRANSLATION (zho_Hans) ===")
    t0 = time.time()
    zh_all = translate_nllb(translator, tokenizer, text)
    print(zh_all)
    print(f"\n[time] {time.time()-t0:.2f}s")

    # 2) Sentence-by-sentence (to compare segmentation effect)
    sents = split_sentences_ja(text)
    if len(sents) >= 2:
        print("\n=== SENTENCE-BY-SENTENCE (compare) ===")
        for i, s in enumerate(sents, 1):
            t0 = time.time()
            zh = translate_nllb(translator, tokenizer, s)
            dt = time.time() - t0
            print(f"\n[{i}] JA: {s}")
            print(f"    ZH: {zh}")
            print(f"    (time {dt:.2f}s)")
    else:
        print("\n(Only one sentence detected; segmentation compare skipped.)")

    print("\nDone.")

if __name__ == "__main__":
    main()
