#!/usr/bin/env python3
import sys
import os
import ctypes
import numpy as np

def generate_vector_add_cpu(n):
    return f'''
extern "C" __attribute__((noinline))
int vector_add_launch(const float* h_a, const float* h_b, float* h_c, int n) {{
    for (int i = 0; i < n; i++) {{
        h_c[i] = h_a[i] + h_b[i];
    }}
    return 0;
}}
'''

def generate_matmul_cpu(m, n, k):
    return f'''
extern "C" __attribute__((noinline))
int matmul_launch(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {{
    for (int row = 0; row < m; row++) {{
        for (int col = 0; col < n; col++) {{
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {{
                sum += h_a[row * k + i] * h_b[i * n + col];
            }}
            h_c[row * n + col] = sum;
        }}
    }}
    return 0;
}}
'''

def generate_softmax_cpu(n):
    return f'''
#include <cmath>

extern "C" __attribute__((noinline))
int softmax_launch(const float* h_input, float* h_output, int n) {{
    float max_val = h_input[0];
    for (int i = 1; i < n; i++) {{
        if (h_input[i] > max_val) max_val = h_input[i];
    }}
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {{
        sum += std::exp(h_input[i] - max_val);
    }}
    
    for (int i = 0; i < n; i++) {{
        h_output[i] = std::exp(h_input[i] - max_val) / sum;
    }}
    return 0;
}}
'''

def generate_attention_cpu(seq_len, head_dim):
    return f'''
#include <cmath>
#include <algorithm>

extern "C" __attribute__((noinline))
int attention_launch(const float* h_q, const float* h_k, const float* h_v, float* h_o, int seq_len, int head_dim) {{
    float scale = 1.0f / std::sqrt((float)head_dim);
    
    for (int row = 0; row < seq_len; row++) {{
        float m_i = -1e20f;
        float s_i = 0.0f;
        
        for (int col = 0; col < head_dim; col++) {{
            h_o[row * head_dim + col] = 0.0f;
        }}
        
        for (int j = 0; j < seq_len; j++) {{
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {{
                dot += h_q[row * head_dim + d] * h_k[j * head_dim + d];
            }}
            float S_ij = dot * scale;
            
            float m_next = std::max(m_i, S_ij);
            float alpha = std::exp(m_i - m_next);
            float beta = std::exp(S_ij - m_next);
            
            float s_next = s_i * alpha + beta;
            
            for (int col = 0; col < head_dim; col++) {{
                h_o[row * head_dim + col] = h_o[row * head_dim + col] * alpha + beta * h_v[j * head_dim + col];
            }}
            
            m_i = m_next;
            s_i = s_next;
        }}
        
        for (int col = 0; col < head_dim; col++) {{
            h_o[row * head_dim + col] /= s_i;
        }}
    }}
    return 0;
}}
'''

def generate_vector_add_mlx(n):
    return f'''
#include <mlx/mlx.h>
#include <iostream>

extern "C" __attribute__((noinline))
int vector_add_launch(const float* h_a, const float* h_b, float* h_c, int n) {{
    auto arr_a = mlx::core::array(h_a, {{n}}, mlx::core::float32);
    auto arr_b = mlx::core::array(h_b, {{n}}, mlx::core::float32);
    auto arr_c = mlx::core::add(arr_a, arr_b);
    mlx::core::eval(arr_c);
    std::copy(arr_c.data<float>(), arr_c.data<float>() + n, h_c);
    return 0;
}}
'''

def generate_matmul_mlx(m, n, k):
    return f'''
#include <mlx/mlx.h>
#include <iostream>

extern "C" __attribute__((noinline))
int matmul_launch(const float* h_a, const float* h_b, float* h_c, int m, int n, int k) {{
    auto arr_a = mlx::core::array(h_a, {{m, k}}, mlx::core::float32);
    auto arr_b = mlx::core::array(h_b, {{k, n}}, mlx::core::float32);
    auto arr_c = mlx::core::matmul(arr_a, arr_b);
    mlx::core::eval(arr_c);
    std::copy(arr_c.data<float>(), arr_c.data<float>() + m * n, h_c);
    return 0;
}}
'''

def generate_softmax_mlx(n):
    return f'''
#include <mlx/mlx.h>
#include <iostream>

extern "C" __attribute__((noinline))
int softmax_launch(const float* h_input, float* h_output, int n) {{
    auto arr_input = mlx::core::array(h_input, {{n}}, mlx::core::float32);
    auto arr_output = mlx::core::softmax(arr_input, std::vector<int>{{-1}});
    mlx::core::eval(arr_output);
    std::copy(arr_output.data<float>(), arr_output.data<float>() + n, h_output);
    return 0;
}}
'''

def generate_attention_mlx(seq_len, head_dim):
    return f'''
#include <mlx/mlx.h>
#include <iostream>
#include <cmath>

extern "C" __attribute__((noinline))
int attention_launch(const float* h_q, const float* h_k, const float* h_v, float* h_o, int seq_len, int head_dim) {{
    auto q = mlx::core::array(h_q, {{seq_len, head_dim}}, mlx::core::float32);
    auto k = mlx::core::array(h_k, {{seq_len, head_dim}}, mlx::core::float32);
    auto v = mlx::core::array(h_v, {{seq_len, head_dim}}, mlx::core::float32);
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    auto scores = mlx::core::matmul(q, mlx::core::transpose(k, std::vector<int>{{1, 0}}));
    scores = mlx::core::multiply(scores, mlx::core::array(scale));
    
    auto probs = mlx::core::softmax(scores, std::vector<int>{{-1}});
    auto out = mlx::core::matmul(probs, v);
    
    mlx::core::eval(out);
    std::copy(out.data<float>(), out.data<float>() + seq_len * head_dim, h_o);
    return 0;
}}
'''

def compile_kernel(name, source, cache_dir, backend="cpu"):
    cpp_path = os.path.join(cache_dir, f"{name}.cpp")
    
    if os.name == "nt":
        so_path = os.path.join(cache_dir, f"{name}.dll")
        export_sym = ""
        if "vector_add" in name:
            export_sym = "vector_add_launch"
        elif "matmul" in name:
            export_sym = "matmul_launch"
        elif "softmax" in name:
            export_sym = "softmax_launch"
        elif "attention" in name:
            export_sym = "attention_launch"
            
        # Define away __attribute__ and add robust linker export pragma for MSVC
        source = f'#ifdef _MSC_VER\n#define __attribute__(x)\n#pragma comment(linker, "/EXPORT:{export_sym}")\n#endif\n' + source
    else:
        so_path = os.path.join(cache_dir, f"{name}.so")

    with open(cpp_path, "w") as f:
        f.write(source)

    if os.name == "nt":
        # Find Visual Studio on Windows
        vswhere_path = os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            "Microsoft Visual Studio", "Installer", "vswhere.exe"
        )
        vs_path = None
        if os.path.exists(vswhere_path):
            try:
                import subprocess
                res = subprocess.run(
                    [vswhere_path, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                    capture_output=True, text=True
                )
                vs_path = res.stdout.strip()
            except Exception:
                pass
        
        if not vs_path or not os.path.exists(vs_path):
            default_vs = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community"
            if os.path.exists(default_vs):
                vs_path = default_vs

        if vs_path:
            vcvars = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
            if os.path.exists(vcvars):
                obj_path = os.path.join(cache_dir, f"{name}.obj")
                cmd = f'"{vcvars}" amd64 && cl.exe /LD /O2 /EHsc /std:c++11 /Fe:"{so_path}" /Fo:"{obj_path}" "{cpp_path}"'
                import subprocess
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Compiled with MSVC cl.exe", file=sys.stderr)
                    return so_path
                else:
                    print(f"MSVC compilation failed: {result.stderr}", file=sys.stderr)
        
        print("ERROR: No C++ compiler / Visual Studio 2022 installation found on Windows", file=sys.stderr)
        raise RuntimeError("No C++ compiler found")
    else:
        compilers = [
            "/usr/bin/clang++", "/usr/bin/g++", "/usr/bin/gcc", "/usr/bin/clang",
        ]
        for compiler in compilers:
            try:
                if backend == "mlx":
                    cmd = [compiler, "-shared", "-fPIC", "-O3", "-std=c++20",
                           "-I/opt/homebrew/include", "-L/opt/homebrew/lib", "-lmlx",
                           "-o", so_path, cpp_path]
                else:
                    cmd = [compiler, "-shared", "-fPIC", "-O3", "-ffast-math", "-std=c++11",
                           "-o", so_path, cpp_path]
                
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Compiled with {compiler}", file=sys.stderr)
                    return so_path
                else:
                    print(f"Compilation failed with {compiler}: {result.stderr}", file=sys.stderr)
            except FileNotFoundError:
                continue
        else:
            print(f"ERROR: No C++ compiler found (tried {compilers})", file=sys.stderr)
            raise RuntimeError("No C++ compiler found")

def generate_mrope_cpu():
    return '''
extern "C" __attribute__((noinline))
int mrope_launch(
    const float* q,
    const float* k,
    const float* cos,
    const float* sin,
    float* q_out,
    float* k_out,
    int bs, int num_heads, int num_kv_heads, int seq_len, int head_dim
) {
    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int c = 0; c < head_dim; c++) {
                    int comp;
                    int cos_c;
                    if (c < 24) { comp = 0; cos_c = c; }
                    else if (c < 44) { comp = 1; cos_c = c; }
                    else if (c < 64) { comp = 2; cos_c = c; }
                    else if (c < 88) { comp = 0; cos_c = c; }
                    else if (c < 108) { comp = 1; cos_c = c; }
                    else { comp = 2; cos_c = c; }

                    int idx = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim + c;
                    int cos_idx = comp * (bs * seq_len * head_dim) + b * (seq_len * head_dim) + s * head_dim + cos_c;

                    float cos_val = cos[cos_idx];
                    float sin_val = sin[cos_idx];

                    if (c < 64) {
                        int partner_idx = idx + 64;
                        q_out[idx] = q[idx] * cos_val - q[partner_idx] * sin_val;
                    } else {
                        int partner_idx = idx - 64;
                        q_out[idx] = q[idx] * cos_val + q[partner_idx] * sin_val;
                    }
                }
            }
        }
    }
    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < num_kv_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int c = 0; c < head_dim; c++) {
                    int comp;
                    int cos_c;
                    if (c < 24) { comp = 0; cos_c = c; }
                    else if (c < 44) { comp = 1; cos_c = c; }
                    else if (c < 64) { comp = 2; cos_c = c; }
                    else if (c < 88) { comp = 0; cos_c = c; }
                    else if (c < 108) { comp = 1; cos_c = c; }
                    else { comp = 2; cos_c = c; }

                    int idx = b * (num_kv_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim + c;
                    int cos_idx = comp * (bs * seq_len * head_dim) + b * (seq_len * head_dim) + s * head_dim + cos_c;

                    float cos_val = cos[cos_idx];
                    float sin_val = sin[cos_idx];

                    if (c < 64) {
                        int partner_idx = idx + 64;
                        k_out[idx] = k[idx] * cos_val - k[partner_idx] * sin_val;
                    } else {
                        int partner_idx = idx - 64;
                        k_out[idx] = k[idx] * cos_val + k[partner_idx] * sin_val;
                    }
                }
            }
        }
    }
    return 0;
}
'''

def generate_mrope_mlx():
    return generate_mrope_cpu()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_kernel.py <op> <args...>", file=sys.stderr)
        sys.exit(1)

    op = sys.argv[1]
    cache_dir = os.environ.get("TILELANG_CACHE_DIR", "/tmp/tilelang")
    backend = os.environ.get("TILELANG_BACKEND", "cpu")
    os.makedirs(cache_dir, exist_ok=True)

    if op == "vector_add":
        n = int(sys.argv[2])
        source = generate_vector_add_mlx(n) if backend == "mlx" else generate_vector_add_cpu(n)
        so_path = compile_kernel(f"vector_add_{backend}_{n}", source, cache_dir, backend)
        print(so_path)
    elif op == "matmul":
        m, n, k = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        source = generate_matmul_mlx(m, n, k) if backend == "mlx" else generate_matmul_cpu(m, n, k)
        so_path = compile_kernel(f"matmul_{backend}_{m}x{n}x{k}", source, cache_dir, backend)
        print(so_path)
    elif op == "softmax":
        n = int(sys.argv[2])
        source = generate_softmax_mlx(n) if backend == "mlx" else generate_softmax_cpu(n)
        so_path = compile_kernel(f"softmax_{backend}_{n}", source, cache_dir, backend)
        print(so_path)
    elif op == "attention":
        seq_len, head_dim = int(sys.argv[2]), int(sys.argv[3])
        source = generate_attention_mlx(seq_len, head_dim) if backend == "mlx" else generate_attention_cpu(seq_len, head_dim)
        so_path = compile_kernel(f"attention_{backend}_{seq_len}x{head_dim}", source, cache_dir, backend)
        print(so_path)
    elif op == "mrope":
        source = generate_mrope_mlx() if backend == "mlx" else generate_mrope_cpu()
        so_path = compile_kernel(f"mrope_{backend}", source, cache_dir, backend)
        print(so_path)
    else:
        print(f"Unknown op: {op}", file=sys.stderr)
        sys.exit(1)
