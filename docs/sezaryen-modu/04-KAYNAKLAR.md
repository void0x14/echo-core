# Sezaryen Modu — Kaynaklar

Her satır: link + neden önemli + sayı/etiket.

---

## Çekirdek denklem ve üretim ölçümü

- [Speculative Decoding: Performance or Illusion? (MLSys 2026)](https://proceedings.mlsys.org/paper_files/paper/2026/file/554e056fe2b6d9fd27ffcd3367ae1267-Paper-Conference.pdf) — Hızlanma denklemi, alfa = 1 simülasyonu, üretimde 4.9x adaptif tavan. **LİTERATÜR.**
- [PyTorch: A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) — Temel anlatım, draft/verify. **LİTERATÜR.**

## Eğitimsiz / embedding tarafı

- [ESP: Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](https://arxiv.org/abs/2603.17942) — Mask token injection, 1.22-1.48x. **LİTERATÜR.**
- [Prompt Lookup Decoding](https://ryanorban.com/notes/prompt-lookup-decoding/) — n-gram, draft model yok, 2.4x. **LİTERATÜR.**
- [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252) — Retrieval draft, 1.62-2.36x. **LİTERATÜR.**
- [ANPD: Adaptive N-gram Parallel Decoding](https://arxiv.org/html/2404.08698v2) — 3.67x. **LİTERATÜR.**
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Jacobi + n-gram, 1.8x; 4x çoklu GPU. **LİTERATÜR.**
- [Skeleton-of-Thought](https://sites.google.com/view/sot-llm) — Paralel segment, ~2x. **LİTERATÜR.**
- [FastCoT](https://arxiv.org/abs/2311.08263) — Model-agnostik paralel decoding, ~%20. **LİTERATÜR.**

## Eğitimli / hidden state tarafı

- [Hidden Transfer](https://arxiv.org/abs/2404.12022) — Orta katmana pseudo hidden state, 1.4-2.3x. **LİTERATÜR.**
- [FIRP](https://arxiv.org/abs/2410.20488) — Gelecek hidden state tahmini, 1.9-3x. **LİTERATÜR.**
- [EAGLE](https://arxiv.org/abs/2401.15077) — Hidden state + token embedding, 2-3x. **LİTERATÜR.**
- [EAGLE-3](https://arxiv.org/html/2503.01840) — Hidden state taklitini bırakıp token tahminine yönelir. **LİTERATÜR.**
- [Medusa](https://arxiv.org/html/2401.10774v2) — Çoklu head, tree attention. **LİTERATÜR.**
- [ParallelSpec](https://arxiv.org/abs/2410.05589) — Paralel drafter, 2.84x. **LİTERATÜR.**

## Hibrit C

- [Lyanna: Make Every Draft Count](https://arxiv.org/abs/2602.21224) — Hidden-state draft + token-info injection + reuse, 3.3x. **LİTERATÜR.**
- [DFlash](https://arxiv.org/html/2602.06036) — Hidden bağlam + blok difüzyon, 6.1x (Qwen3-8B, H200). **LİTERATÜR.**
- [Falcon](https://arxiv.org/abs/2412.12639) — Yarı-AR drafter + tree, 2.91-3.51x. **LİTERATÜR.**
- [Hybrid Spark](https://arxiv.org/abs/2611.05611) — Model-based + model-free drafter, 1.10-1.62x ek. **LİTERATÜR.**
- [MetaSD](https://arxiv.org/abs/2604.05417) — Çoklu drafter, bandit. **LİTERATÜR.**
- [PaSS](https://arxiv.org/abs/2311.13581) — Eğitimli embedding; "embedding = eğitimsiz" değil. **LİTERATÜR.**
- [CLaSp](https://aclanthology.org/2025.acl-long.1525/) — Eğitimsiz hidden state; "hidden state = eğitimli" değil. **LİTERATÜR.**
- [Token Recycling](https://aclanthology.org/2025.acl-long.338/) — Reddedilen adayları geri dönüştür. **LİTERATÜR.**

## Deterministik / dış kesinlik

- [llguidance fast-forward](https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md) — Grammar zorunlu metni blok ekle; token seçimini kaldır. **LİTERATÜR.**
- [CopyLoop](https://arxiv.org/html/2610.38822) — Looped-Transformer'da 2.98-10.70x; özel model ailesi. **LİTERATÜR.**

## Doğrulama ve kabul

- [SVIP: Draft Model Knows When to Stop](https://arxiv.org/abs/2411.18462) — Entropi tabanlı adaptif draft uzunluğu. **LİTERATÜR.**
- [AdaEDL](https://arxiv.org/abs/2410.18351) — Erken draft durdurma, %10-57. **LİTERATÜR.**
- [PEARL](https://arxiv.org/abs/2408.11850) — Paralel draft + adaptif uzunluk, 4.43x. **LİTERATÜR.**
- [LK Losses](https://arxiv.org/abs/2602.23881) — Kabul oranını doğrudan optimize eden loss. **LİTERATÜR.**

## Model-agnostik runtime / hook

- [TransformerLens hook sistemi](https://transformerlensorg.github.io/TransformerLens/content/hook_system.html) — Aktivasyon okuma/yazma. **LİTERATÜR.**
- [IBM vLLM Hook](https://github.com/IBM/vLLM-Hook) — vLLM içinde hidden state müdahalesi. **LİTERATÜR.**
- [PyTorch cache açıklaması](https://huggingface.co/docs/transformers/cache_explanation) — KV cache/attention mask uyumu. **LİTERATÜR.**

## Bu repodaki ölçümler

- Qwen3.5-0.8B Q8_0, CPU, t=6, batch B=1 -> 55 tok/s; B=256 -> ~300 tok/s. **ÖLÇÜLDÜ.**
- IVF LM head: 38 -> 52-55 tok/s; top-1 %96,2-97,75. **ÖLÇÜLDÜ.**
- Batch taraması: B=128-512 arası ~295-300 tok/s plato; 321 tek seferlik gürültü. **ÖLÇÜLDÜ.**

---

## Not

Bu listede "LİTERATÜR" etiketi, sayının başka bir donanım/model/görevde ölçüldüğünü söyler. Bu repoda aynı sayıyı garanti etmez. Her yeni deney kendi etiketini taşır.
