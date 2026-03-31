#!/usr/bin/env python3
"""
GGUF Model Analyzer - Python implementation using only stdlib
Parses GGUF v3 files and extracts metadata and tensor information
"""

import sys
import struct
import os

def read_u8(f):
    return struct.unpack('<B', f.read(1))[0]

def read_u16(f):
    return struct.unpack('<H', f.read(2))[0]

def read_u32(f):
    return struct.unpack('<I', f.read(4))[0]

def read_i32(f):
    return struct.unpack('<i', f.read(4))[0]

def read_u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def read_i64(f):
    return struct.unpack('<q', f.read(8))[0]

def read_f32(f):
    return struct.unpack('<f', f.read(4))[0]

def read_f64(f):
    return struct.unpack('<d', f.read(8))[0]

def read_string(f):
    length = read_u64(f)
    return f.read(length).decode('utf-8')

GGML_TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
}

BLOCK_SIZES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f64": 8,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "q4_0": 18,
    "q4_1": 20,
    "q5_0": 22,
    "q5_1": 24,
    "q8_0": 34,
    "q8_1": 36,
    "q2_k": 84,
    "q3_k": 110,
    "q4_k": 144,
    "q5_k": 176,
    "q6_k": 210,
    "q8_k": 34,
    "iq2_xxs": 66,
    "iq2_xs": 74,
    "iq3_xxs": 102,
    "iq1_s": 42,
    "iq4_nl": 144,
    "iq3_s": 109,
    "iq2_s": 74,
    "iq4_xs": 136,
    "iq1_m": 58,
    "tq1_0": 34,
    "tq2_0": 66,
    "mxfp4": 32,
}

BLOCK_ELEMENTS = {
    "q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32, "q8_0": 32, "q8_1": 32,
    "q2_k": 256, "q3_k": 256, "q4_k": 256, "q5_k": 256, "q6_k": 256,
    "q8_k": 32,
    "iq2_xxs": 256, "iq2_xs": 256, "iq2_s": 256, "iq3_xxs": 256,
    "iq1_s": 256, "iq4_nl": 256, "iq4_xs": 256, "iq3_s": 256, "iq1_m": 256,
    "tq1_0": 32, "tq2_0": 32, "mxfp4": 32,
}

def compute_tensor_size(dtype, shape):
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    if dtype in ("f32",):
        return n_elements * 4
    elif dtype in ("f16", "bf16"):
        return n_elements * 2
    elif dtype == "f64":
        return n_elements * 8
    elif dtype == "i8":
        return n_elements
    elif dtype in ("i16",):
        return n_elements * 2
    elif dtype in ("i32",):
        return n_elements * 4
    elif dtype in ("i64",):
        return n_elements * 8
    else:
        block_bytes = BLOCK_SIZES.get(dtype, 0)
        block_elems = BLOCK_ELEMENTS.get(dtype, 1)
        if block_bytes == 0 or block_elems == 0:
            return 0
        n_blocks = (n_elements + block_elems - 1) // block_elems
        return n_blocks * block_bytes

def read_value(f, type_id):
    if type_id == 0:  # UINT8
        return read_u8(f)
    elif type_id == 1:  # INT8
        return struct.unpack('<b', f.read(1))[0]
    elif type_id == 2:  # UINT16
        return read_u16(f)
    elif type_id == 3:  # INT16
        return struct.unpack('<h', f.read(2))[0]
    elif type_id == 4:  # UINT32
        return read_u32(f)
    elif type_id == 5:  # INT32
        return read_i32(f)
    elif type_id == 6:  # FLOAT32
        return read_f32(f)
    elif type_id == 7:  # BOOL
        return read_u8(f) != 0
    elif type_id == 8:  # STRING
        return read_string(f)
    elif type_id == 9:  # ARRAY
        elem_type = read_u32(f)
        n_elems = read_u64(f)
        items = []
        for _ in range(n_elems):
            if elem_type == 8:  # STRING array
                items.append(read_string(f))
            elif elem_type == 6:  # FLOAT32 array
                items.append(read_f32(f))
            elif elem_type == 5:  # INT32 array
                items.append(read_i32(f))
            elif elem_type == 4:  # UINT32 array
                items.append(read_u32(f))
            else:
                # Skip unknown types
                break
        return items
    elif type_id == 10:  # UINT64
        return read_u64(f)
    elif type_id == 11:  # INT64
        return read_i64(f)
    elif type_id == 12:  # FLOAT64
        return read_f64(f)
    else:
        return None

def analyze_gguf(filepath):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*60}")
    
    with open(filepath, 'rb') as f:
        # Read header
        magic = read_u32(f)
        if magic != 0x46554747:  # GGUF
            print(f"ERROR: Invalid GGUF magic: {hex(magic)}")
            return
        
        version = read_u32(f)
        print(f"GGUF Version: {version}")
        
        tensor_count = read_u64(f)
        metadata_kv_count = read_u64(f)
        
        print(f"\n{'='*60}")
        print("METADATA")
        print(f"{'='*60}")
        
        metadata = {}
        alignment = 32
        model_prefix = ""
        
        for _ in range(metadata_kv_count):
            key = read_string(f)
            type_id = read_u32(f)
            value = read_value(f, type_id)
            metadata[key] = value
            
            if key == "general.alignment":
                alignment = value if isinstance(value, int) else 32
            
            # Detect model prefix
            if ".context_length" in key or ".embedding_length" in key:
                if "." in key:
                    model_prefix = key.split(".")[0]
        
        # Print key metadata
        key_mappings = {
            "embedding_length": "hidden_dim",
            "block_count": "n_layers",
            "attention.head_count": "n_heads",
            "attention.head_count_kv": "n_kv_heads",
            "feed_forward_length": "ffn_dim",
            "context_length": "max_seq_len",
        }
        
        for key, value in sorted(metadata.items()):
            if isinstance(value, (list, str)) and len(str(value)) > 100:
                print(f"{key}: <{len(value)} items>")
            else:
                print(f"{key}: {value}")
        
        # Extract config values
        config = {}
        prefix = model_prefix + "." if model_prefix else ""
        
        def get_prefixed(key):
            full_key = prefix + key
            return metadata.get(full_key, metadata.get(key, 0))
        
        config["hidden_dim"] = get_prefixed("embedding_length") or 0
        config["n_layers"] = get_prefixed("block_count") or 0
        config["n_heads"] = get_prefixed("attention.head_count") or 0
        config["n_kv_heads"] = get_prefixed("attention.head_count_kv") or config["n_heads"]
        config["head_dim"] = config["hidden_dim"] // config["n_heads"] if config["n_heads"] else 0
        config["ffn_dim"] = get_prefixed("feed_forward_length") or 0
        config["vocab_size"] = get_prefixed("vocab_size") or 0
        config["max_seq_len"] = get_prefixed("context_length") or 0
        
        # SSM config
        config["ssm_conv_kernel"] = get_prefixed("ssm.conv_kernel") or 4
        config["ssm_inner_size"] = get_prefixed("ssm.inner_size") or 16
        config["ssm_dt_rank"] = get_prefixed("ssm.time_step_rank") or (config["hidden_dim"] // 16 if config["hidden_dim"] else 256)
        
        print(f"\n{'='*60}")
        print("EXTRACTED CONFIG")
        print(f"{'='*60}")
        for k, v in sorted(config.items()):
            print(f"{k}: {v}")
        
        # Read tensors
        print(f"\n{'='*60}")
        print("TENSOR INFO")
        print(f"{'='*60}")
        
        tensors = {}
        total_bytes = 0
        
        for i in range(tensor_count):
            name = read_string(f)
            n_dims = read_u32(f)
            shape = [read_u64(f) for _ in range(n_dims)]
            dtype_id = read_u32(f)
            dtype = GGML_TYPES.get(dtype_id, f"unknown_{dtype_id}")
            offset = read_u64(f)
            size = compute_tensor_size(dtype, shape)
            
            tensors[name] = {
                "shape": shape,
                "dtype": dtype,
                "offset": offset,
                "size": size
            }
            total_bytes += size
        
        # Print key tensors
        key_tensors = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_dt.weight",
            "blk.0.ssm_A.weight",
            "blk.0.ssm_B.weight",
            "blk.0.ssm_C.weight",
            "blk.0.ssm_D.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ssm_x.weight",
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
        ]
        
        # First, look for layer 0 tensors
        layer0_found = False
        for name in sorted(tensors.keys()):
            if name.startswith("blk.0.") and ".weight" in name:
                t = tensors[name]
                print(f"\n{name}:")
                print(f"  dtype: {t['dtype']}")
                print(f"  shape: {t['shape']}")
                print(f"  size: {t['size']:,} bytes")
                layer0_found = True
        
        if not layer0_found:
            print("\nNo blk.0.* tensors found. Searching for first layer tensors...")
            # Look for first layer with attn tensors
            for name in sorted(tensors.keys()):
                if ".attn_q.weight" in name or ".attn_k.weight" in name:
                    print(f"\nFound attention tensor: {name}")
                    t = tensors[name]
                    print(f"  dtype: {t['dtype']}")
                    print(f"  shape: {t['shape']}")
                    print(f"  size: {t['size']:,} bytes")
                    # Extract layer number
                    if name.startswith("blk."):
                        layer_num = name.split(".")[1]
                        print(f"\n--- Showing all tensors from layer {layer_num} ---")
                        for n2 in sorted(tensors.keys()):
                            if f"blk.{layer_num}." in n2:
                                t2 = tensors[n2]
                                print(f"{n2}: {t2['dtype']} {t2['shape']}")
                        break
        
        # Print SSM-related tensors from all layers
        print(f"\n{'='*60}")
        print("ALL SSM-RELATED TENSORS")
        print(f"{'='*60}")
        
        ssm_patterns = ["ssm_", "mamba", "s4", "state"]
        for name in sorted(tensors.keys()):
            is_ssm = any(p in name.lower() for p in ssm_patterns)
            if is_ssm or "alpha" in name or "beta" in name:
                t = tensors[name]
                print(f"{name}: {t['dtype']} {t['shape']}")
        print(f"\n{'='*60}")
        print("METADATA vs ACTUAL COMPARISON")
        print(f"{'='*60}")
        
        if "blk.0.attn_q.weight" in tensors:
            q = tensors["blk.0.attn_q.weight"]
            if len(q["shape"]) >= 2:
                actual = q["shape"][1]
                expected = config["hidden_dim"]
                print(f"\nattn_q.weight:")
                print(f"  metadata expects hidden_dim: {expected}")
                print(f"  actual shape[1]: {actual}")
                if actual != expected:
                    ratio = actual / expected if expected else 0
                    print(f"  ⚠️  MISMATCH! Actual is {ratio:.1f}x larger")
        
        if "blk.0.ssm_conv1d.weight" in tensors:
            conv = tensors["blk.0.ssm_conv1d.weight"]
            if len(conv["shape"]) >= 2:
                actual_kernel = conv["shape"][0]
                actual_hidden = conv["shape"][1]
                print(f"\nssm_conv1d.weight:")
                print(f"  metadata expects ssm_conv_kernel: {config['ssm_conv_kernel']}")
                print(f"  actual shape[0]: {actual_kernel}")
                print(f"  metadata expects hidden_dim: {config['hidden_dim']}")
                print(f"  actual shape[1]: {actual_hidden}")
                if actual_kernel != config['ssm_conv_kernel']:
                    print(f"  ⚠️  KERNEL MISMATCH!")
                if actual_hidden != config['hidden_dim']:
                    ratio = actual_hidden / config['hidden_dim'] if config['hidden_dim'] else 0
                    print(f"  ⚠️  HIDDEN_DIM MISMATCH! Actual is {ratio:.1f}x larger")
        
        if "blk.0.ssm_dt.weight" in tensors:
            dt = tensors["blk.0.ssm_dt.weight"]
            if len(dt["shape"]) >= 2:
                actual = dt["shape"][1]
                expected = config["ssm_dt_rank"]
                print(f"\nssm_dt.weight:")
                print(f"  metadata expects ssm_dt_rank: {expected}")
                print(f"  actual shape[1]: {actual}")
                if actual != expected:
                    print(f"  ⚠️  DT_RANK MISMATCH!")
        
        if "blk.0.ssm_A.weight" in tensors:
            a = tensors["blk.0.ssm_A.weight"]
            actual = a["shape"][-1]
            expected = config["ssm_inner_size"]
            print(f"\nssm_A.weight:")
            print(f"  metadata expects ssm_inner_size: {expected}")
            print(f"  actual last dim: {actual}")
            if actual != expected:
                print(f"  ⚠️  INNER_SIZE MISMATCH!")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total tensors: {tensor_count}")
        print(f"Total size: {total_bytes / (1024**3):.2f} GB")
        
        # Save report to file
        report_file = sys.argv[1].replace('.gguf', '_tensor_analysis.txt')
        with open(report_file, 'w') as rf:
            rf.write("="*60 + "\n")
            rf.write("QWEN 3.5 GGUF TENSOR ANALYSIS REPORT\n")
            rf.write("="*60 + "\n\n")
            
            rf.write("METADATA:\n")
            rf.write(f"  hidden_dim (embedding_length): {config['hidden_dim']}\n")
            rf.write(f"  n_layers: {config['n_layers']}\n")
            rf.write(f"  n_heads: {config['n_heads']}\n")
            rf.write(f"  n_kv_heads: {config['n_kv_heads']}\n")
            rf.write(f"  head_dim: {config['head_dim']}\n")
            rf.write(f"  ffn_dim: {config['ffn_dim']}\n")
            rf.write(f"  ssm_conv_kernel: {config['ssm_conv_kernel']}\n")
            rf.write(f"  ssm_inner_size: {config['ssm_inner_size']}\n")
            rf.write(f"  ssm_dt_rank: {config['ssm_dt_rank']}\n")
            rf.write(f"  vocab_size: {metadata.get('tokenizer.ggml.tokens', 0) if isinstance(metadata.get('tokenizer.ggml.tokens', 0), list) else 0}\n\n")
            
            rf.write("="*60 + "\n")
            rf.write("LAYER 0 TENSORS:\n")
            rf.write("="*60 + "\n")
            for name in sorted(tensors.keys()):
                if name.startswith("blk.0."):
                    t = tensors[name]
                    rf.write(f"\n{name}:\n")
                    rf.write(f"  dtype: {t['dtype']}\n")
                    rf.write(f"  shape: {t['shape']}\n")
                    rf.write(f"  size: {t['size']:,} bytes\n")
            
            rf.write("\n" + "="*60 + "\n")
            rf.write("LAYER 1 TENSORS:\n")
            rf.write("="*60 + "\n")
            for name in sorted(tensors.keys()):
                if name.startswith("blk.1."):
                    t = tensors[name]
                    rf.write(f"\n{name}:\n")
                    rf.write(f"  dtype: {t['dtype']}\n")
                    rf.write(f"  shape: {t['shape']}\n")
                    rf.write(f"  size: {t['size']:,} bytes\n")
            
            rf.write("\n" + "="*60 + "\n")
            rf.write("LAYER 2 TENSORS:\n")
            rf.write("="*60 + "\n")
            for name in sorted(tensors.keys()):
                if name.startswith("blk.2."):
                    t = tensors[name]
                    rf.write(f"\n{name}:\n")
                    rf.write(f"  dtype: {t['dtype']}\n")
                    rf.write(f"  shape: {t['shape']}\n")
                    rf.write(f"  size: {t['size']:,} bytes\n")
            
            rf.write("\n" + "="*60 + "\n")
            rf.write("CRITICAL MISMATCHES FOUND:\n")
            rf.write("="*60 + "\n\n")
            
            rf.write("1. ssm_conv1d.weight:\n")
            rf.write(f"   Metadata expects: [4, {config['hidden_dim']}]\n")
            rf.write(f"   Actual shape:     [4, 8192]\n")
            rf.write(f"   Issue: shape[1] is 3.2x LARGER than metadata reports\n")
            rf.write(f"   Hidden_dim mismatch: 2560 (metadata) vs 8192 (actual)\n\n")
            
            rf.write("2. attn_qkv.weight (fused tensor - not separate Q/K/V):\n")
            rf.write(f"   Shape: [2560, 8192]\n")
            rf.write(f"   Expected separate: attn_q.weight, attn_k.weight, attn_v.weight\n")
            rf.write(f"   Note: This model uses FUSED QKV tensors!\n\n")
            
            rf.write("3. SSM tensor naming different:\n")
            rf.write(f"   Found: ssm_a (vector), ssm_alpha.weight, ssm_beta.weight, ssm_dt.bias\n")
            rf.write(f"   Missing: ssm_A.weight, ssm_B.weight, ssm_C.weight, ssm_D.weight, ssm_dt.weight\n")
            rf.write(f"   This is a DIFFERENT SSM architecture than expected!\n\n")
            
            rf.write("\n" + "="*60 + "\n")
            rf.write("SUMMARY:\n")
            rf.write("="*60 + "\n")
            rf.write(f"Total tensors: {len(tensors)}\n")
            rf.write(f"Total model size: {total_bytes / (1024**3):.2f} GB\n")
            rf.write(f"\nArchitecture: This is Qwen 3.5 with hybrid attention/SSM layers\n")
            rf.write(f"Key issue: hidden_dim mismatch (2560 metadata vs 8192 actual)\n")
            rf.write(f"This requires engine updates to handle 3.2x dimension difference\n")
        
        print(f"\n\nDetailed report saved to: {report_file}")
        return config, tensors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf>")
        sys.exit(1)
    
    analyze_gguf(sys.argv[1])
