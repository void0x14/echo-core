# Qwen3.5 Linear Operator Contract

## Scope

Bu belge, mevcut `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf` icin yerelde dogrulanmis contract'i sabitler. Amaç, generic Mamba/SSM yolunu Qwen3.5 linear katmanlarina zorlamamak ve kod degisikliklerini dogrulanmis veri uzerinden yonlendirmektir.

## Verified Metadata

- `general.architecture = qwen35`
- `qwen35.embedding_length = 2560`
- `qwen35.attention.head_count = 16`
- `qwen35.attention.head_count_kv = 4`
- `qwen35.attention.key_length = 256`
- `qwen35.attention.value_length = 256`
- `qwen35.block_count = 32`
- `qwen35.full_attention_interval = 4`
- `qwen35.ssm.conv_kernel = 4`
- `qwen35.ssm.group_count = 16`
- `qwen35.ssm.inner_size = 4096`
- `qwen35.ssm.state_size = 128`
- `qwen35.ssm.time_step_rank = 32`

## Layer Pattern

- Toplam katman: 32
- Yerel analiz ve mevcut loader log'larina gore hibrit dagilim: 24 linear benzeri katman, 8 full-attention katmani
- Mevcut kod tabaninda bu linear katmanlar `ssm` diye etiketleniyor, ancak tensor contract'i generic Mamba ile ayni degil

## Full-Attention Tensors

- `attn_qkv.weight`
- `attn_gate.weight`
- `attn_norm.weight`
- `post_attention_norm.weight`
- `ffn_gate.weight`
- `ffn_up.weight`
- `ffn_down.weight`

## Qwen Linear Tensors

- `ssm_a`
- `ssm_alpha.weight`
- `ssm_beta.weight`
- `ssm_dt.bias`
- `ssm_conv1d.weight`
- `ssm_norm.weight`
- `ssm_out.weight`

## Known-Absent Generic Mamba Tensors

- `ssm_x.weight`
- `ssm_dt.weight`
- `ssm_D.weight`
- `ssm_A.weight`
- `ssm_B.weight`
- `ssm_C.weight`

## Dtype Contract

- `q8_0`: `attn_qkv.weight`, `attn_gate.weight`, `ffn_*`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_out.weight`
- `f32`: `attn_norm.weight`, `post_attention_norm.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight`, `ssm_norm.weight`

## Non-Negotiable Rules For Implementation

- `head_dim` bu model icin `attention.key_length` metadata'sindan alinmali; `hidden_dim / num_heads` turetimi burada canonical degil
- Qwen linear katmanlari generic `ssmForward` contract'ina zorlanmamali
- `f32` tensorler fp16 byte dizisi gibi reinterpret edilmemeli
- `ssm_x.weight`, `ssm_dt.weight`, `ssm_D.weight` yokmus gibi degil, gercekten absent kabul edilmeli
- `qwen_linear` ile generic `ssm` ayri layer/operator kind olarak ele alinmali

## Open Questions

- `attn_qkv.weight` ic ayrisiminin tam runtime contract'i
- `attn_gate.weight` ile `ssm_out.weight` arasindaki kesin operator baglantisi
- `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias` tensorlerinin resmi implementasyondaki matematiksel rolleri
- Qwen linear state modelinin tam sekli
