# hymt_test.py
# Local HY-MT1.5 translation test (Windows + Conda + CUDA)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# CONFIG (edit here)
# =====================
MODEL_ID = "tencent/HY-MT1.5-1.8B"     # start with 1.8B
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SRC_TEXT = """元気をチャージする ラジオとなって おります 寝る前の 30分 お布団に入ってくつろぎ ながら 明日のスタッフを しながら またまた 勉強を頑張る 学生さんに も楽しんでもら えると っても嬉しいです ハッシュタグ は ハッシュタグ寝る前ラジオで SNSに 感想など 投稿していただけると 嬉しいです そして 本配信の スーパーチャットや メンバーシップの加入 メッセージ等は 配信後 風間が寝る前にお布団でゆっくり読ませていただきますので 配信上で の読み上げはございません あらかじめ ご了承ください いつもスーパーチャット そしてメンバーシップの加入 に ありがとうございます 始まりました ございます皆さんこんばんは…"""

# 输出控制：强烈建议用“只翻译，不解释”的提示词
PROMPT = (
    "Translate the following Japanese into Simplified Chinese.\n"
    "Requirements:\n"
    "- Keep the original meaning; do not add or omit information.\n"
    "- Keep names (people/organizations) as-is.\n"
    "- Use natural spoken Chinese suitable for live subtitles.\n"
    "- Output ONLY the translation.\n\n"
    "Japanese:\n{jp}\n\nChinese:"
)

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.7
# =====================

def main():
    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    prompt = PROMPT.format(jp=SRC_TEXT.strip())
    messages = [{"role": "user", "content": prompt}]

    # chat template（HY 系列通常支持）
    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 再编码成模型需要的张量（包含 input_ids / attention_mask）
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=(TEMPERATURE > 0),
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)

    # 简单截断：取最后一段（通常是回答）
    # 如果输出里带了 prompt，可手动 print(text) 看一下再微调
    print("\n=== RAW OUTPUT ===\n")
    print(text)

if __name__ == "__main__":
    main()
