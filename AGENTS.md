# Zig Upstream MCP — ZORUNLU KULLANIM

Bu projede Zig ile ilgili herhangi bir soru, araştırma, dokümantasyon lookup, 
issue/PR takibi veya güncelleme kontrolü için **zig-upstream** MCP araçları 
KULLANILMALIDIR.

## Kullanım Şekli

Prompt'unuzda `use zig-upstream:` ön eki + doğal dil sorunuzu yazın:

```
use zig-upstream: zig_upstream__search query="comptime nasıl çalışır"
use zig-upstream: zig_upstream__issues label="bug" limit=5
use zig-upstream: zig_upstream__context
use zig-upstream: zig_upstream__activity days=7
```

## Mevcut Araçlar

| Araç | Ne işe yarar |
|------|-------------|
| `zig_upstream__version` | Yerel Zig sürümü |
| `zig_upstream__context` | Sürüm + milestone bağlamı |
| `zig_upstream__repo_summary` | Codeberg repo istatistiği |
| `zig_upstream__search` | **Ana araç** — doğal dilde tüm kaynaklarda ara |
| `zig_upstream__issues` | Codeberg issue ara |
| `zig_upstream__prs` | Codeberg PR ara |
| `zig_upstream__docs` | Zig dokümantasyonunda ara |
| `zig_upstream__commits` | Son commit'leri getir |
| `zig_upstream__source` | Repo'dan dosya oku |
| `zig_upstream__milestones` | Release milestone takibi |
| `zig_upstream__activity` | Son aktivite özeti |
| `zig_upstream__issue` | Tek bir issue/PR detayı |
| `zig_upstream__help` | Tüm araçları listele |

## Kural

- Zig ile ilgili HER ŞEY için önce `zig_upstream__search` dene
- Doğal dil sorgusu yeterli, keyword çıkarmaya gerek yok
- Dokümantasyon için `zig_upstream__docs` kullan
- Issue/PR için `zig_upstream__issues` / `zig_upstream__prs` kullan
