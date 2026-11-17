//! vulkan 后端（burn/vulkan feature）。
//!
//! 精度由环境变量 `VOXLAB_FLOAT` 选择：`f16` / `f32`（默认 `bf16`）。
//! gfx1103 没有 bf16 matmul（cubek-matmul 会明确报 not supported），
//! 所以这类卡需要换一个浮点类型；f16 与 bf16 同为 2 字节权重，容量不变。
use burn_cubecl::kernel::matmul::set_force_naive_matmul;
use clap::Parser;

fn main() -> anyhow::Result<()> {
    voxcpm_burn::init_logging();
    // gfx1103 has no bf16 matmul; the Cube kernel panics rather than falling
    // back, so allow forcing the naive kernel.
    if std::env::var("VOXLAB_NAIVE_MATMUL").is_ok() {
        println!("matmul: naive (VOXLAB_NAIVE_MATMUL)");
        set_force_naive_matmul(true);
    }
    let cli = voxcpm_burn::Cli::parse();
    match std::env::var("VOXLAB_FLOAT").as_deref() {
        Ok("f16") => voxcpm_burn::run::<burn::backend::Vulkan<half::f16, i32>>(cli, Default::default()),
        Ok("f32") => voxcpm_burn::run::<burn::backend::Vulkan<f32, i32>>(cli, Default::default()),
        _ => voxcpm_burn::run::<burn::backend::Vulkan<half::bf16, i32>>(cli, Default::default()),
    }
}
