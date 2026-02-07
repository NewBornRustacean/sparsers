# sparsers
sparse-rs; sparse matrix computation written in rust

### Why reinvent another wheel?
1. As we know, there're great frameworks to deal with matrix multiplication including sparse matrices.
As a Deep Learning Engineer who has lived and breathed PyTorch for years, I recognize it as a great framework. However, we’ve all been there: wrestling with torch-sparse or torch-scatter dependency hell during deployment, or hitting performance walls in CPU-bound sparse operations.
Perhaps one day, when the industry fully embraces Rust for ML(who knows? lol), this kind of works could be a "building block" for rustaceans.
2. So called "agentic engineering" is rising. We're truly living in an era of "Agentic Engineering." I co-work with AI to write codes in my day-to-day work. It's efficient, but something was missing-the feeling of "hand made". I missed the deep dives into documentation, the meditative struggle with the borrow checker, and the rewarding "click" when rust-analyzer finally clears those red lines. To me, this project, `sparsers`, isn't just about performance. it's kinda nostalgia for the vanshing craftmanship

### naive benchmark
| Component | Specification |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7600 (Zen 4 Architecture) |
| **Cores / Threads** | 6 Cores / 12 Threads |
| **L3 Cache** | 32 MiB (Unified) |

avx-512 is disabled. 

![alt text](<스크린샷 2026-02-07 21-03-11.png>)


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
