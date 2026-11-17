//! 跨后端数值对照工具（vox-lab 内部使用，不属于上游 crate 的公开 API）。
//!
//! 两条子命令：
//!   encode --feat <f32le> --shape B,T,P,D      音频 patch -> LM hidden (f32le)
//!   vae    --latent <f32le> --shape B,D,T       latent -> 波形 (f32le)
//!   generate --text ... --out <f32le>          完整生成（受 --seed 控制）
//!
//! 输出一律是裸 f32 little-endian，便于和 PyTorch 侧直接比对 SNR。
//!
//! 精度由环境变量选择（与 voxcpm-burn-* 一致）：
//!   VOXLAB_FLOAT=bf16 | f16 | f32         （Vulkan）
//!   VOXLAB_NAIVE_MATMUL=1                 （gfx1103 上跑 bf16 需要）

use std::env;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use clap::{Parser, Subcommand};
use voxcpm_rs::{GenerateOptions, VoxCPM};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "")]
    model_dir: String,
    /// Output path for raw f32 LE.
    #[arg(long)]
    out: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Raw audio [B,1,T] -> AudioVAE latent [B,D,T'].
    Audio {
        #[arg(long)]
        input: PathBuf,
        /// B,T samples (mono)
        #[arg(long)]
        shape: String,
    },
    /// Audio patches -> LM hidden.
    Encode {
        /// Raw f32 LE file.
        #[arg(long)]
        input: PathBuf,
        /// B,T,P,D
        #[arg(long)]
        shape: String,
    },
    /// Latent patches -> waveform.
    Vae {
        #[arg(long)]
        input: PathBuf,
        /// B,D,T
        #[arg(long)]
        shape: String,
    },
    /// Full text-to-waveform generation.
    Generate {
        #[arg(long)]
        text: String,
        #[arg(long, default_value_t = 4)]
        timesteps: usize,
        #[arg(long, default_value_t = 2.0)]
        cfg: f32,
        #[arg(long, default_value_t = 64)]
        max_len: usize,
        #[arg(long)]
        seed: Option<u64>,
    },
}

fn read_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} is not a multiple of 4 bytes", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn parse_shape(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().context("bad shape element"))
        .collect()
}

fn write_f32(path: &PathBuf, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn run<B: Backend>(cli: Cli, device: B::Device, tag: &str) -> Result<()> {
    println!("backend: {tag}");
    let model_dir = if cli.model_dir.is_empty() {
        PathBuf::from("data/models/voxcpm2")
    } else {
        PathBuf::from(&cli.model_dir)
    };
    let model: VoxCPM<B> = VoxCPM::from_local(&model_dir, &device).context("load model")?;

    match cli.cmd {
        Cmd::Audio { input, shape } => {
            let dims = parse_shape(&shape)?;
            if dims.len() != 3 {
                bail!("--shape for audio must be B,1,T");
            }
            let values = read_f32(&input)?;
            let audio = Tensor::<B, 3>::from_data(TensorData::new(values, dims), &device);
            let (out, out_dims) = model.encode_audio_f32(audio)?;
            println!("latent dims: {out_dims:?}");
            write_f32(&cli.out, &out)?;
        }
        Cmd::Encode { input, shape } => {
            let dims = parse_shape(&shape)?;
            if dims.len() != 4 {
                bail!("--shape for encode must be B,T,P,D");
            }
            let values = read_f32(&input)?;
            let expected = dims.iter().product::<usize>();
            if values.len() != expected {
                bail!("input has {} values, shape {dims:?} wants {expected}", values.len());
            }
            let feat = Tensor::<B, 4>::from_data(TensorData::new(values, dims), &device);
            let (out, out_dims) = model.encode_feat_f32(feat)?;
            println!("encode out dims: {out_dims:?}");
            write_f32(&cli.out, &out)?;
        }
        Cmd::Vae { input, shape } => {
            let dims = parse_shape(&shape)?;
            if dims.len() != 3 {
                bail!("--shape for vae must be B,D,T");
            }
            let values = read_f32(&input)?;
            let latent = Tensor::<B, 3>::from_data(TensorData::new(values, dims), &device);
            let wav = model.decode_feat_f32(latent, Some(7))?;
            println!("vae out samples: {}", wav.len());
            write_f32(&cli.out, &wav)?;
        }
        Cmd::Generate {
            text,
            timesteps,
            cfg,
            max_len,
            seed,
        } => {
            let mut b = GenerateOptions::builder()
                .timesteps(timesteps)
                .cfg(cfg)
                .max_len(max_len);
            if let Some(seed) = seed {
                b = b.seed(seed);
            }
            let wav = model.generate(&text, b.build()).context("generate")?;
            println!("generate samples: {}", wav.len());
            write_f32(&cli.out, &wav)?;
        }
    }
    Ok(())
}

fn main() {
    // Print the error explicitly: a Result returned from main is printed by
    // Termination, which some terminals swallow for context-wrapped errors.
    if let Err(e) = run_main() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run_main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();

    #[cfg(feature = "vulkan")]
    {
        use burn::backend::Vulkan;
        if env::var("VOXLAB_NAIVE_MATMUL").is_ok() {
            burn_cubecl::kernel::matmul::set_force_naive_matmul(true);
            println!("matmul: naive");
        }
        return match env::var("VOXLAB_FLOAT").as_deref() {
            Ok("bf16") => run::<Vulkan<half::bf16, i32>>(cli, Default::default(), "vulkan/bf16"),
            Ok("f32") => run::<Vulkan<f32, i32>>(cli, Default::default(), "vulkan/f32"),
            _ => run::<Vulkan<half::f16, i32>>(cli, Default::default(), "vulkan/f16"),
        };
    }

    #[cfg(all(feature = "cpu", not(feature = "vulkan")))]
    {
        use burn::backend::NdArray;
        return run::<NdArray<f32>>(cli, Default::default(), "cpu/f32");
    }

    #[allow(unreachable_code)]
    {
        bail!("build with --features vulkan or --features cpu")
    }
}
