"""Uçtan uca araç çağırma: openai istemcisi → echo-core sunucu → tool_calls → araç sonucu → cevap.
Kullanım: python benchmarks/arac_test.py [ek sunucu bayrakları, ör. --fast]  (echo-core-zig dizininden)"""
import json, subprocess, sys, time, urllib.request
from openai import OpenAI

MODEL = "../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf"
PORT = 8094
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
SORU = [{"role": "user", "content": "What is the weather in Paris right now?"}]


def kontrol(ad, calls):
    assert calls, f"{ad}: tool_calls yok"
    c = calls[0]
    args = json.loads(c.function.arguments)
    assert c.function.name == "get_weather", f"{ad}: ad {c.function.name}"
    assert "paris" in str(args.get("city", "")).lower(), f"{ad}: args {args}"
    print(f"OK {ad}: {c.function.name}({c.function.arguments})")
    return c


def main():
    srv = subprocess.Popen(["zig-out/bin/echo-core", MODEL, "--server", "--port", str(PORT), *sys.argv[1:]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        cl = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        m = cl.models.list().data[0].id

        r = cl.chat.completions.create(model=m, messages=SORU, tools=TOOLS, temperature=0, max_tokens=128)
        assert r.choices[0].finish_reason == "tool_calls", r.choices[0].finish_reason
        c = kontrol("akışsız", r.choices[0].message.tool_calls)

        st = cl.chat.completions.create(model=m, messages=SORU, tools=TOOLS, temperature=0, max_tokens=128, stream=True)
        parcalar, son, metin = [], None, ""
        for ch in st:
            d = ch.choices[0].delta
            if d.tool_calls:
                parcalar += d.tool_calls
            if d.content:
                metin += d.content
            son = ch.choices[0].finish_reason or son
        assert son == "tool_calls", son
        assert "<tool_call>" not in metin, metin
        kontrol("akışlı", parcalar)

        gecmis = SORU + [{"role": "assistant", "content": r.choices[0].message.content or "",
                          "tool_calls": [{"id": c.id, "type": "function",
                                          "function": {"name": c.function.name, "arguments": c.function.arguments}}]},
                         {"role": "tool", "tool_call_id": c.id, "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
        r2 = cl.chat.completions.create(model=m, messages=gecmis, tools=TOOLS, temperature=0, max_tokens=96)
        cevap = r2.choices[0].message.content or ""
        assert r2.choices[0].finish_reason == "stop" and not r2.choices[0].message.tool_calls, r2.choices[0]
        assert "18" in cevap, cevap
        print("OK araç sonucu → cevap:", cevap.strip().replace("\n", " ")[:200])

        r3 = cl.chat.completions.create(model=m, messages=[{"role": "user", "content": "Say hello."}], tools=TOOLS,
                                        temperature=0, max_tokens=32)
        assert not r3.choices[0].message.tool_calls and r3.choices[0].message.content, r3.choices[0]
        print("OK araç gerekmeyen soru: düz cevap")
        print("HEPSİ GEÇTİ")
    finally:
        srv.terminate()


main()
