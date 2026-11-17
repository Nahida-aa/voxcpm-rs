//! wgpu 后端（Linux 上经 RADV 走 Vulkan）。
//!
//! `VOXLAB_FLOAT=f16|f32` 选精度（默认 f32）。f32 下 4.4 GB bf16 权重约 8.7 GB，
//! 核显共享显存容易 OOM；f16 与 bf16 同为 2 字节，容量不变。
//! `VOXLAB_NAIVE_MATMUL=1` 强制朴素 matmul（该后端不需要，保留一致性）。
use clap::Parser;

fn main() -> anyhow::Result<()> {
    voxcpm_burn::init_logging();
    let cli = voxcpm_burn::Cli::parse();
    match std::env::var("VOXLAB_FLOAT").as_deref() {
        Ok("f16") => voxcpm_burn::run::<burn::backend::wgpu::Wgpu<half::f16, i32>>(cli, Default::default()),
        Ok("bf16") => voxcpm_burn::run::<burn::backend::wgpu::Wgpu<half::bf16, i32>>(cli, Default::default()),
        _ => voxcpm_burn::run::<burn::backend::wgpu::Wgpu<f32, i32>>(cli, Default::default()),
    }
}
