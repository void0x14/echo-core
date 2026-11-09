"""Sunucunun chat şablonu = GGUF'taki resmi şablon (transformers jinja ile) mi?
Kullanım: python benchmarks/sablon_test.py  (echo-core-zig dizininden; sunucuyu kendisi açar)
Yapısal durumlar: tools, tool rolü, tool_calls geçmişi, düşünce blokları, iki system, parçalı içerik."""
import copy, json, os, subprocess, sys, tempfile, time, urllib.request
import gguf
from transformers.utils.chat_template_utils import render_jinja_template

MODEL = "../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf"
PORT = 8093
r = gguf.GGUFReader(MODEL)
f = r.fields["tokenizer.chat_template"]
TMPL = bytes(f.parts[f.data[0]]).decode()

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Hava durumu \"şehir\" için\nçok satır",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1},
                    "ratio": {"type": "number", "default": 0.5}, "flags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "noop", "description": "", "parameters": {"type": "object", "properties": {}}}},
]
CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather",
        "arguments": {"city": "İzmir", "days": 3, "flags": ["a", "b"], "opts": {"x": True, "y": None}, "ok": False}}}
CALL2 = {"id": "c2", "type": "function", "function": {"name": "noop", "arguments": {}}}

DURUMLAR = {
    "sade": ([{"role": "user", "content": "a"}], None, False),
    "system": ([{"role": "system", "content": " s \n"}, {"role": "user", "content": " a "}], None, False),
    "iki_system": ([{"role": "system", "content": "s1"}, {"role": "developer", "content": "s2"}, {"role": "user", "content": "a"}], None, False),
    "think_acik": ([{"role": "user", "content": "a"}], None, True),
    "parcali": ([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}], None, False),
    "gecmis_think": ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>\nd\n</think>\n\nc"},
                      {"role": "user", "content": "b"}], None, False),
    "tools": ([{"role": "user", "content": "a"}], TOOLS, False),
    "tools_system": ([{"role": "system", "content": "s"}, {"role": "user", "content": "a"}], TOOLS, False),
    "tool_dongusu": ([{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                      {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]},
                      {"role": "tool", "tool_call_id": "c1", "content": "{\"t\": 21}"},
                      {"role": "tool", "tool_call_id": "c2", "content": "ok"}], TOOLS, False),
    "tool_sonra_cevap": ([{"role": "user", "content": "a"},
                          {"role": "assistant", "content": "bakıyorum", "tool_calls": [CALL]},
                          {"role": "tool", "content": "21"},
                          {"role": "assistant", "content": "<think>\nr\n</think>\n\n21 derece"},
                          {"role": "user", "content": "b"}], TOOLS, True),
    "reasoning_alani": ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "c", "reasoning_content": " r "}], None, False),
    "null_icerik": ([{"role": "user", "content": "a"}, {"role": "assistant", "content": None, "tool_calls": [CALL]},
                     {"role": "tool", "content": "x"}], TOOLS, False),
}


def istek(msgs, tools, think, yol):
    body = {"messages": msgs, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()
    return open(yol, encoding="utf-8").read()


def main():
    yol = os.path.join(tempfile.mkdtemp(), "sablon.txt")
    env = dict(os.environ, ECHO_SABLON_YAZ=yol)
    srv = subprocess.Popen(["zig-out/bin/echo-core", MODEL, "--server", "--port", str(PORT)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        bozuk = 0
        for ad, (msgs, tools, think) in DURUMLAR.items():
            ref = render_jinja_template([copy.deepcopy(msgs)], tools=tools, chat_template=TMPL,
                                        add_generation_prompt=True, enable_thinking=think)[0][0]
            # sunucuya OpenAI biçimi: arguments JSON metni
            gonder = copy.deepcopy(msgs)
            for m in gonder:
                for tc in m.get("tool_calls", []):
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
            biz = istek(gonder, tools, think, yol)
            ok = biz == ref
            bozuk += not ok
            print(f"{'AYNI ' if ok else 'FARKLI'} {ad}")
            if not ok:
                i = next((k for k in range(min(len(biz), len(ref))) if biz[k] != ref[k]), min(len(biz), len(ref)))
                print("  ref:", repr(ref[max(0, i - 40):i + 40]))
                print("  biz:", repr(biz[max(0, i - 40):i + 40]))
        print(f"{len(DURUMLAR) - bozuk}/{len(DURUMLAR)} aynı")
        sys.exit(1 if bozuk else 0)
    finally:
        srv.terminate()


main()
