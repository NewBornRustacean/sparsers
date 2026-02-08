# sparsers
sparse-rs; sparse matrix computation written in rust

### Why reinvent another wheel?
1. As we know, there're great frameworks to deal with matrix multiplication including sparse matrices.
As a Deep Learning Engineer who has lived and breathed PyTorch for years, I recognize it as a great framework. However, we’ve all been there: wrestling with torch-sparse or torch-scatter dependency hell during deployment, or hitting performance walls in CPU-bound sparse operations.
Perhaps one day, when the industry fully embraces Rust for ML(who knows? lol), this kind of works could be a "building block" for rustaceans.
2. So called "agentic engineering" is rising. We're truly living in an era of "Agentic Engineering." I co-work with AI to write codes in my day-to-day work. It's efficient, but something was missing-the feeling of "hand made". I missed the deep dives into documentation, the meditative struggle with the borrow checker, and the rewarding "click" when rust-analyzer finally clears those red lines. To me, this project, `sparsers`, isn't just about performance. it's kinda nostalgia for the vanshing craftmanship

### Features
- `spmm`: sparse X dense matrix multiplication. [see also](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)
- `sddmm`: multiplies two dense matrices X1 and X2, then elementwise-multiplies the result with sparse matrix A at the nonzero locations. [like this: dgl](https://www.dgl.ai/dgl_docs/generated/dgl.sparse.sddmm.html)

### naive benchmark
#### machine spec
| Component | Specification |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7600 (Zen 4 Architecture) |
| **Cores / Threads** | 6 Cores / 12 Threads |
| **L3 Cache** | 32 MiB (Unified) |

#### sparsers and torch (num_threads=12)
| Implementation | runtime |
| :--- | :--- |
| **torch** | 572.08 ms |
| **sparsers** | 322.22 ms |


<details>
  <summary> criterion screenshot </summary>
  <br>

![alt text](<스크린샷 2026-02-07 21-03-11.png>)

</details>

<details>
  <summary> torch script for this simple bench</summary>
  <br>

  ```python
    import torch
    import numpy as np
    import torch.utils.benchmark as benchmark
    from scipy.sparse import load_npz

    def setup_data(npz_path, feature_dim=128):
        sparse_matrix = load_npz(npz_path)
        
        crow_indices = torch.from_numpy(sparse_matrix.indptr).to(torch.int64)
        col_indices = torch.from_numpy(sparse_matrix.indices).to(torch.int64)
        values = torch.from_numpy(sparse_matrix.data).to(torch.float64)
        
        csr_tensor = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, 
            size=sparse_matrix.shape, dtype=torch.float64
        )
        
        dense_b = torch.ones((sparse_matrix.shape[1], feature_dim), dtype=torch.float64)
        
        return csr_tensor, dense_b

    def bench_pytorch_spmm(csr, dense_b):
        return torch.mm(csr, dense_b)

    if __name__ == "__main__":
        npz_path = "/home/seom/workspace/sparsers/data/roadNet-CA.csr.npz"
        csr, dense_b = setup_data(npz_path)

        t0 = benchmark.Timer(
            stmt='bench_pytorch_spmm(csr, dense_b)',
            setup='from __main__ import bench_pytorch_spmm',
            globals={'csr': csr, 'dense_b': dense_b},
            num_threads=1,
            label='SpMM Performance',
            sub_label='PyTorch Native CSR',
            description='roadNet-CA'
        )

        
        t1 = benchmark.Timer(
            stmt='bench_pytorch_spmm(csr, dense_b)',
            setup='from __main__ import bench_pytorch_spmm',
            globals={'csr': csr, 'dense_b': dense_b},
            num_threads=12, 
            label='SpMM Performance',
            sub_label='PyTorch Native CSR (Multi-thread)',
            description='roadNet-CA'
        )

        print(t0.blocked_autorange())
        print(t1.blocked_autorange())
  ```

  </details>


### simple example with ndarray
you can find more examples at `examples/`


```rust 
use ndarray::{Array2, array};
use sparsers::{sddmm_ndarray, spmm_ndarray};

fn main() {
    // Example: SpMM using CSR matrix converted from 10 * 10 ndarray
    let sparse_array = array![
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let dense_matrix = array![
        // Dense matrix (10x2)
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
        [13.0, 14.0],
        [15.0, 16.0],
        [17.0, 18.0],
        [19.0, 20.0],
    ];
    let result = spmm_ndarray::<u32, f32>(sparse_array.view(), dense_matrix.view());
    println!("Result of SpMM:\n{:#?}", result);

    // Example: SDDMM using CSR matrix converted from 5 * 5 ndarray
    let sparse_array = array![
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 8.0, 0.0],
    ];

    let dense_matrix_d1 = array![
        // Dense matrix d1 (5x1)
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ];
    let dense_matrix_d2 = array![
        // Dense matrix d2 (1x5)
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ];

    let result = sddmm_ndarray::<u32, f32>(
        sparse_array.view(),
        dense_matrix_d1.view(),
        dense_matrix_d2.view(),
    );
    println!("Result of SDDMM:\n{:#?}", result);
}

```
### things to do
- add more ops to kernel: transpose(csr to csc), broadcasting element-wise ops, etc.
- python binding(pyo3)
- comprehensive benchmarks
