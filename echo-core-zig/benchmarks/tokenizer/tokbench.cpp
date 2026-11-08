// Tokenizer hız + doğruluk ölçümü. Referans: Qwen resmi tokenizer.json (hazirla.py ref_*.txt üretir).
// Kullanım: tokbench <veri_dizini> <model.gguf>  -> stdout'a TSV
#include <llama.h>
#include "ucuncu/fastBPE.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
using namespace std;
using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using u64 = uint64_t;

static string oku(const string &p) { ifstream f(p, ios::binary); stringstream s; s << f.rdbuf(); return s.str(); }

// ---- (a,b) -> değer; açık adresli hash ----
struct HMap {
    vector<u64> k; vector<u32> v1, v2; u64 mask = 0; int sh = 0;
    void init(size_t n) { size_t c = 1; sh = 64; while (c < n * 2) { c <<= 1; sh--; } k.assign(c, 0); v1.resize(c); v2.resize(c); mask = c - 1; }
    inline size_t h(u64 x) const { return (x * 0x9E3779B97F4A7C15ull) >> sh; }
    void put(u64 key, u32 a, u32 b) { key++; size_t i = h(key); while (k[i] && k[i] != key) i = (i + 1) & mask; if (!k[i]) { k[i] = key; v1[i] = a; v2[i] = b; } }
    inline bool get(u64 key, u32 &a, u32 &b) const { key++; size_t i = h(key); while (k[i]) { if (k[i] == key) { a = v1[i]; b = v2[i]; return true; } i = (i + 1) & mask; } return false; }
};

// ---- vocab ----
static vector<string> id2b; static unordered_map<string_view, int> b2id; static int byte_id[256];
static HMap pairs;  // (sol,sağ) -> (rank, birleşik id)
static string b2u[256]; static unordered_map<u32, u8> u2b;
static vector<u16> s1; static vector<u8> s2;
static inline u8 bayrak(u32 cp) { return cp >= 0x110000 ? 0 : s2[(size_t)s1[cp >> 8] * 256 + (cp & 255)]; }
enum { FL = 1, FM = 2, FN = 4, FS = 8 };

static void utf8put(string &s, u32 c) {
    if (c < 0x80) s += char(c);
    else if (c < 0x800) { s += char(0xC0 | c >> 6); s += char(0x80 | (c & 63)); }
    else { s += char(0xE0 | c >> 12); s += char(0x80 | ((c >> 6) & 63)); s += char(0x80 | (c & 63)); }
}
static inline u32 utf8get(const u8 *t, size_t n, size_t i, int &len) {
    u8 c = t[i];
    if (c < 0x80) { len = 1; return c; }
    if ((c >> 5) == 6 && i + 1 < n) { len = 2; return (c & 31) << 6 | (t[i + 1] & 63); }
    if ((c >> 4) == 14 && i + 2 < n) { len = 3; return (c & 15) << 12 | (t[i + 1] & 63) << 6 | (t[i + 2] & 63); }
    if ((c >> 3) == 30 && i + 3 < n) { len = 4; return (c & 7) << 18 | (t[i + 1] & 63) << 12 | (t[i + 2] & 63) << 6 | (t[i + 3] & 63); }
    len = 1; return 0xFFFD;
}

static void yukle(const string &dir) {
    vector<int> bs; for (int b = '!'; b <= '~'; b++) bs.push_back(b); for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b); for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    vector<int> cs = bs; int n = 0; vector<bool> var(256); for (int b : bs) var[b] = true;
    for (int b = 0; b < 256; b++) if (!var[b]) { bs.push_back(b); cs.push_back(256 + n++); }
    for (size_t i = 0; i < bs.size(); i++) { b2u[bs[i]].clear(); utf8put(b2u[bs[i]], cs[i]); u2b[cs[i]] = bs[i]; }

    string v = oku(dir + "/vocab.bin"); const char *p = v.data(); u32 cnt; memcpy(&cnt, p, 4); p += 4;
    id2b.resize(250000);
    for (u32 i = 0; i < cnt; i++) { u32 id, len; memcpy(&id, p, 4); memcpy(&len, p + 4, 4); p += 8; id2b[id].assign(p, len); p += len; }
    for (size_t i = 0; i < id2b.size(); i++) if (!id2b[i].empty()) b2id[id2b[i]] = (int)i;
    for (int b = 0; b < 256; b++) byte_id[b] = b2id.at(string(1, char(b)));

    pairs.init(260000);
    istringstream m(oku(dir + "/merges.txt")); string line; u32 rank = 0;
    auto cevir = [](const string &s) { string o; const u8 *t = (const u8 *)s.data(); for (size_t i = 0; i < s.size();) { int l; u32 c = utf8get(t, s.size(), i, l); o += char(u2b.at(c)); i += l; } return o; };
    while (getline(m, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        string a = cevir(line.substr(0, sp)), b = cevir(line.substr(sp + 1));
        auto ia = b2id.find(a), ib = b2id.find(b), im = b2id.find(a + b);
        if (ia != b2id.end() && ib != b2id.end() && im != b2id.end()) pairs.put((u64)ia->second << 32 | (u32)ib->second, rank, im->second);
        rank++;
    }
    string u = oku(dir + "/ucat.bin"); u32 nb; memcpy(&nb, u.data(), 4);
    s1.resize(0x1100); memcpy(s1.data(), u.data() + 4, 0x1100 * 2); s2.assign(u.begin() + 4 + 0x2200, u.end());
}

// ---- qwen35 ön-bölme (regex'in elle yazılmış hali) ----
struct Pre { vector<u32> c, o; vector<u8> f; vector<pair<u32, u32>> sp; };
static void onbol(const char *txt, size_t n, Pre &P) {
    const u8 *t = (const u8 *)txt; P.c.clear(); P.o.clear(); P.f.clear(); P.sp.clear();
    for (size_t i = 0; i < n;) { int l; u32 c = utf8get(t, n, i, l); P.c.push_back(c); P.o.push_back(i); P.f.push_back(bayrak(c)); i += l; }
    size_t N = P.c.size(); P.o.push_back(n);
    auto &c = P.c; auto &f = P.f;
    auto low = [](u32 x) { return x >= 'A' && x <= 'Z' ? x + 32 : x; };
    auto isP = [&](size_t x) { return !(f[x] & (FS | FL | FM | FN)); };
    for (size_t k = 0; k < N;) {
        size_t e = 0;
        if (c[k] == '\'' && k + 1 < N) {
            u32 a = low(c[k + 1]);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') e = k + 2;
            else if (k + 2 < N) { u32 b = low(c[k + 2]); if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = k + 3; }
        }
        if (!e) {
            if (f[k] & (FL | FM)) { e = k + 1; while (e < N && (f[e] & (FL | FM))) e++; }
            else if (c[k] != '\r' && c[k] != '\n' && !(f[k] & (FL | FN)) && k + 1 < N && (f[k + 1] & (FL | FM))) { e = k + 2; while (e < N && (f[e] & (FL | FM))) e++; }
            else if (f[k] & FN) e = k + 1;
            else {
                size_t s = k; if (c[k] == ' ' && k + 1 < N && isP(k + 1)) s = k + 1;
                if (isP(s)) { e = s + 1; while (e < N && isP(e)) e++; while (e < N && (c[e] == '\r' || c[e] == '\n')) e++; }
                else {  // boşluk (f[k] & FS)
                    size_t j = k; long nl = -1; while (j < N && (f[j] & FS)) { if (c[j] == '\r' || c[j] == '\n') nl = j; j++; }
                    if (j == k) j = k + 1;  // tanımsız karakter: tek başına
                    if (nl >= 0) e = nl + 1; else if (j == N) e = j; else if (j - k >= 2) e = j - 1; else e = j;
                }
            }
        }
        P.sp.push_back({P.o[k], P.o[e]}); k = e;
    }
}

// ---- BPE çeşitleri (kelime başına) ----
static inline bool pr(u32 a, u32 b, u32 &r, u32 &m) { return pairs.get((u64)a << 32 | b, r, m); }

struct Sym { int id, prev, next; };
struct HE { u32 r; int pos; int a, b; };
static void bpe_heap(const u8 *w, size_t n, vector<int> &out) {
    static vector<Sym> s; static vector<HE> h; s.resize(n); h.clear();
    for (size_t i = 0; i < n; i++) s[i] = {byte_id[w[i]], (int)i - 1, i + 1 < n ? (int)i + 1 : -1};
    auto cmp = [](const HE &x, const HE &y) { return x.r != y.r ? x.r > y.r : x.pos > y.pos; };
    auto ekle = [&](int l, int r) { u32 rk, m; if (pr(s[l].id, s[r].id, rk, m)) { h.push_back({rk, l, s[l].id, s[r].id}); push_heap(h.begin(), h.end(), cmp); } };
    for (size_t i = 0; i + 1 < n; i++) ekle(i, i + 1);
    while (!h.empty()) {
        HE t = h.front(); pop_heap(h.begin(), h.end(), cmp); h.pop_back();
        int l = t.pos; if (s[l].id != t.a) continue; int r = s[l].next; if (r < 0 || s[r].id != t.b) continue;
        u32 rk, m; pr(t.a, t.b, rk, m);
        s[l].id = m; s[r].id = -1; s[l].next = s[r].next; if (s[r].next >= 0) s[s[r].next].prev = l;
        if (s[l].prev >= 0) ekle(s[l].prev, l); if (s[l].next >= 0) ekle(l, s[l].next);
    }
    for (int i = 0; i >= 0; i = s[i].next) out.push_back(s[i].id);
}
static void bpe_dizi(const u8 *w, size_t n, vector<int> &out) {
    static vector<int> v; v.resize(n); for (size_t i = 0; i < n; i++) v[i] = byte_id[w[i]];
    while (v.size() > 1) {
        u32 best = UINT32_MAX, bm = 0; int bi = -1;
        for (size_t i = 0; i + 1 < v.size(); i++) { u32 r, m; if (pr(v[i], v[i + 1], r, m) && r < best) { best = r; bm = m; bi = i; } }
        if (bi < 0) break;
        v[bi] = bm; v.erase(v.begin() + bi + 1);
    }
    out.insert(out.end(), v.begin(), v.end());
}
static void bpe_dizi_kisa(const u8 *w, size_t n, vector<int> &out) {
    auto it = b2id.find(string_view((const char *)w, n));
    if (it != b2id.end()) { out.push_back(it->second); return; }
    bpe_dizi(w, n, out);
}

// ---- trie (LinMaxMatch / Viterbi) ----
static HMap kenar; static vector<int> term; static u32 dugum = 1;
static void trie_kur() {
    kenar.init(2'000'000); term.assign(2'000'000, -1);
    for (size_t id = 0; id < id2b.size(); id++) {
        const string &s = id2b[id]; if (s.empty()) continue; u32 nd = 0;
        for (u8 ch : s) { u32 a, b; if (kenar.get((u64)nd << 8 | ch, a, b)) nd = a; else { kenar.put((u64)nd << 8 | ch, dugum, 0); nd = dugum++; } }
        if (term[nd] < 0) term[nd] = (int)id;
    }
}
static void linmax(const u8 *w, size_t n, vector<int> &out) {
    for (size_t p = 0; p < n;) {
        u32 nd = 0; int best = byte_id[w[p]]; size_t be = p + 1;
        for (size_t j = p; j < n; j++) { u32 a, b; if (!kenar.get((u64)nd << 8 | w[j], a, b)) break; nd = a; if (term[nd] >= 0) { best = term[nd]; be = j + 1; } }
        out.push_back(best); p = be;
    }
}
static void viterbi(const u8 *w, size_t n, vector<int> &out) {
    static vector<u32> d, from; static vector<int> tid; d.assign(n + 1, UINT32_MAX); from.resize(n + 1); tid.resize(n + 1); d[0] = 0;
    for (size_t i = 0; i < n; i++) {
        if (d[i] == UINT32_MAX) continue; u32 nd = 0;
        for (size_t j = i; j < n; j++) { u32 a, b; if (!kenar.get((u64)nd << 8 | w[j], a, b)) break; nd = a;
            if (term[nd] >= 0 && d[i] + 1 < d[j + 1]) { d[j + 1] = d[i] + 1; from[j + 1] = i; tid[j + 1] = term[nd]; } }
    }
    size_t k = out.size(); for (size_t j = n; j > 0; j = from[j]) out.push_back(tid[j]); reverse(out.begin() + k, out.end());
}

// ---- fastBPE (Facebook) çekirdeği: process_bpe ----
static unordered_map<fastBPE::tps, uint32_t, fastBPE::pair_hash> fb_codes; static unordered_map<string, fastBPE::tps> fb_rev;
static unordered_map<string, uint32_t> fb_vocab; static unordered_map<string, int> u2id;
static string gstr(const string &b) { string o; for (u8 c : b) o += b2u[c]; return o; }
static void fb_kur(const string &dir) {
    for (size_t i = 0; i < id2b.size(); i++) if (!id2b[i].empty()) u2id[gstr(id2b[i])] = (int)i;
    istringstream m(oku(dir + "/merges.txt")); string line; u32 r = 0;
    while (getline(m, line)) { if (line.empty()) continue; size_t sp = line.find(' '); string a = line.substr(0, sp), b = line.substr(sp + 1);
        fb_codes.emplace(make_pair(a, b), r); fb_codes.emplace(make_pair(a, b + "</w>"), r); r++; }  // kelime sonu işareti uyumu
}
static void fastbpe(const u8 *w, size_t n, vector<int> &out) {
    vector<string> sw; sw.reserve(n); for (size_t i = 0; i < n; i++) sw.push_back(b2u[w[i]]); sw.back() += "</w>";
    string r = fastBPE::process_bpe(sw, fb_codes, fb_rev, fb_vocab);
    size_t a = 0; while (true) { size_t sp = r.find(' ', a); string tok = r.substr(a, sp == string::npos ? string::npos : sp - a);
        if (sp != string::npos) tok.resize(tok.size() - 2); out.push_back(u2id.at(tok)); if (sp == string::npos) break; a = sp + 1; }
}

// ---- bugünkü echo SimpleTokenizer kopyası (tokenizer.zig:137): uzunluğa göre sıralı listede doğrusal açgözlü arama ----
static vector<pair<string, int>> echo_sirali;
static void echo_bugun(const char *t, size_t n, vector<int> &out) {
    for (size_t p = 0; p < n;) {
        bool ok = false;
        for (auto &[s, id] : echo_sirali) if (p + s.size() <= n && memcmp(t + p, s.data(), s.size()) == 0) { out.push_back(id); p += s.size(); ok = true; break; }
        if (!ok) { auto it = u2id.find(b2u[(u8)t[p]]); out.push_back(it == u2id.end() ? -1 : it->second); p++; }
    }
}

// ---- llama.cpp referansı ----
static const llama_vocab *lv;
static void llama_tok(const char *t, size_t n, vector<int> &out) {
    size_t k = out.size(); out.resize(k + n + 16);
    int r = llama_tokenize(lv, t, (int)n, out.data() + k, (int)(n + 16), false, true);
    if (r < 0) { out.resize(k - r); r = llama_tokenize(lv, t, (int)n, out.data() + k, -r, false, true); }
    out.resize(k + r);
}

using Fn = function<void(const char *, size_t, vector<int> &)>;
static vector<pair<string, int>> kullanici;  // <think> vb., uzundan kısaya
static Fn kelimeli(void (*f)(const u8 *, size_t, vector<int> &)) {
    return [f](const char *t, size_t n, vector<int> &out) {
        static Pre P;
        auto parca = [&](size_t a, size_t b) { if (b <= a) return; onbol(t + a, b - a, P); for (auto [x, y] : P.sp) f((const u8 *)t + a + x, y - x, out); };
        size_t bas = 0;
        for (const char *q = (const char *)memchr(t, '<', n); q; q = (const char *)memchr(q + 1, '<', n - (q + 1 - t))) {
            size_t p = q - t;
            for (auto &[s, id] : kullanici) if (p + s.size() <= n && memcmp(t + p, s.data(), s.size()) == 0) { parca(bas, p); out.push_back(id); bas = p + s.size(); q = t + bas - 1; break; }
            if (q + 1 >= t + n) break;
        }
        parca(bas, n);
    };
}

static double sure(const Fn &f, const string &s, int R, size_t &ntok) {
    vector<double> v; vector<int> out; out.reserve(s.size());
    for (int i = 0; i < R; i++) { out.clear(); auto t0 = chrono::steady_clock::now(); f(s.data(), s.size(), out); v.push_back(chrono::duration<double>(chrono::steady_clock::now() - t0).count()); }
    sort(v.begin(), v.end()); ntok = out.size(); return v[v.size() / 2];
}
static vector<string> paragraflar(const string &s) { vector<string> p; size_t a = 0; while (true) { size_t b = s.find("\n\n", a); p.push_back(s.substr(a, b == string::npos ? string::npos : b - a)); if (b == string::npos) break; a = b + 2; } return p; }

static void sessiz(ggml_log_level, const char *, void *) {}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "kullanim: tokbench <veri> <model.gguf>\n"); return 1; }
    string dir = argv[1];
    llama_log_set(sessiz, nullptr); llama_backend_init();
    auto mp = llama_model_default_params(); mp.vocab_only = true;
    llama_model *model = llama_model_load_from_file(argv[2], mp); if (!model) { fprintf(stderr, "model yuklenemedi\n"); return 1; }
    lv = llama_model_get_vocab(model);
    yukle(dir); trie_kur(); fb_kur(dir);
    { istringstream ku(oku(dir + "/kullanici.tsv")); string l; while (getline(ku, l)) { size_t tb = l.find('\t'); if (tb != string::npos) kullanici.push_back({l.substr(tb + 1), stoi(l.substr(0, tb))}); }
      sort(kullanici.begin(), kullanici.end(), [](auto &a, auto &b) { return a.first.size() > b.first.size(); }); }
    for (auto &[s, id] : u2id) echo_sirali.push_back({s, id});
    stable_sort(echo_sirali.begin(), echo_sirali.end(), [](auto &a, auto &b) { return a.first.size() > b.first.size(); });
    fprintf(stderr, "trie dugum: %u  (Aho-Corasick yogun tablo: %u x 256 x 4 B = %.0f MB)\n", dugum, dugum, dugum * 256.0 * 4 / 1e6);
    printf("#trie_dugum\t%u\n", dugum);

    struct Y { const char *ad; Fn f; int R; };
    vector<Y> ys = {
        {"llama.cpp (libllama)", llama_tok, 5},
        {"Qwen BPE: heap (oncelik kuyrugu)", kelimeli(bpe_heap), 5},
        {"Qwen BPE: dizi tarama", kelimeli(bpe_dizi), 5},
        {"Qwen BPE: dizi tarama + tam kelime kisayolu", kelimeli(bpe_dizi_kisa), 5},
        {"fastBPE cekirdegi (process_bpe)", kelimeli(fastbpe), 3},
        {"LinMaxMatch (trie, en uzun eslesme)", kelimeli(linmax), 5},
        {"Viterbi (en az token, trie)", kelimeli(viterbi), 5},
        {"Yalniz on-bolme (regex elle)", [](const char *t, size_t n, vector<int> &o) { static Pre P; onbol(t, n, P); o.resize(P.sp.size()); }, 5},
    };
    vector<string> korp = {"tr", "en", "kod", "uzunkelime"};
    for (auto &k : korp) {
        string s = oku(dir + "/korpus_" + k + ".txt");
        auto par = paragraflar(s);
        vector<vector<int>> ref(par.size());
        { istringstream rf(oku(dir + "/ref_" + k + ".txt")); string l; for (size_t i = 0; i < par.size() && getline(rf, l); i++) { istringstream ls(l); int x; while (ls >> x) ref[i].push_back(x); } }
        for (auto &y : ys) {
            size_t nt; double t = sure(y.f, s, y.R, nt);
            size_t tam = 0; bool on = strncmp(y.ad, "Yalniz", 6) != 0;
            if (on) for (size_t i = 0; i < par.size(); i++) { vector<int> o; y.f(par[i].data(), par[i].size(), o); tam += o == ref[i];
                if (o != ref[i] && getenv("TOKFARK") && strstr(y.ad, getenv("TOKFARK"))) {  // ilk farkı göster
                    size_t j = 0; while (j < o.size() && j < ref[i].size() && o[j] == ref[i][j]) j++;
                    auto g = [](const vector<int> &v, size_t j) { string s; for (size_t x = j; x < min(v.size(), j + 4); x++) s += "[" + (v[x] >= 0 && (size_t)v[x] < id2b.size() ? id2b[v[x]] : string("?")) + "]"; return s; };
                    fprintf(stderr, "FARK %s/%zu tok%zu\n  biz: %s\n  ref: %s\n", k.c_str(), i, j, g(o, j).c_str(), g(ref[i], j).c_str()); } }
            printf("%s\t%s\t%zu\t%.6f\t%zu\t%zu\t%zu\n", y.ad, k.c_str(), s.size(), t, nt, on ? tam : 0, on ? par.size() : 0);
            fflush(stdout);
        }
        if (k == "tr") {  // bugünkü echo: çok yavaş, yalnız ilk 4 KB
            string sm = s.substr(0, 4096); while (!sm.empty() && ((u8)sm.back() & 0xC0) == 0x80) sm.pop_back(); if (!sm.empty() && (u8)sm.back() >= 0xC0) sm.pop_back();
            size_t nt; double t = sure(echo_bugun, sm, 1, nt);
            auto pp = paragraflar(sm); size_t tam = 0;
            for (size_t i = 0; i < pp.size(); i++) { vector<int> o; echo_bugun(pp[i].data(), pp[i].size(), o); tam += i + 1 < pp.size() && o == ref[i]; }
            printf("echo bugunku (tokenizer.zig kopyasi)\t%s\t%zu\t%.6f\t%zu\t%zu\t%zu\n", k.c_str(), sm.size(), t, nt, tam, pp.size());
        }
    }
    llama_model_free(model);
}
