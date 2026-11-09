"""Çok turlu sunucu: önek yeniden kullanımı (taslakçı dahil) sıfırdan işlemeyle aynı cevabı veriyor mu?
Kullanım: python benchmarks/onek_test.py [sunucu bayrakları, ör. --fast -t 1]  (echo-core-zig dizininden)"""
import json, subprocess, sys, time, urllib.request

MODEL = "../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf"
PORT = 8096
TUR = [
    [{"role": "user", "content": "Name three primary colors."}],
    [{"role": "user", "content": "Now name one fruit for each of those colors."}],
    [{"role": "user", "content": "Which of those fruits is the sweetest? One sentence."}],
]


def ac():
    p = subprocess.Popen(["zig-out/bin/echo-core", MODEL, "--server", "--port", str(PORT), *sys.argv[1:]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.5)
    raise SystemExit("sunucu açılmadı")


def sor(msgs):
    body = {"messages": msgs, "temperature": 0, "max_tokens": 64}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]


def main():
    # 1) tek sunucu, turlar sırayla (önek yeniden kullanılır)
    srv = ac()
    gecmis, cevaplar = [], []
    try:
        for t in TUR:
            gecmis += t
            c = sor(gecmis)
            cevaplar.append(c)
            gecmis.append({"role": "assistant", "content": c})
    finally:
        srv.terminate(); srv.wait()
    # 2) her tur için taze sunucu (önek yok)
    bozuk = 0
    for i in range(1, len(TUR)):
        msgs = []
        for k in range(i + 1):
            msgs += TUR[k]
            if k < i:
                msgs.append({"role": "assistant", "content": cevaplar[k]})
        srv = ac()
        try:
            taze = sor(msgs)
        finally:
            srv.terminate(); srv.wait()
        ok = taze == cevaplar[i]
        bozuk += not ok
        print(f"{'AYNI  ' if ok else 'FARKLI'} tur {i + 1}: {cevaplar[i][:80]!r}")
        if not ok:
            a, b = cevaplar[i], taze
            j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            print(f"       {j}. karakterde ayrılıyor; önekli: {a[max(0, j - 30):j + 30]!r}")
            print(f"       {' ' * len(str(j))}               taze: {b[max(0, j - 30):j + 30]!r}")
    for c in cevaplar:
        assert c.strip(), "boş cevap"
    print("HEPSİ GEÇTİ" if not bozuk else f"{bozuk} tur farklı")
    sys.exit(1 if bozuk else 0)


main()
