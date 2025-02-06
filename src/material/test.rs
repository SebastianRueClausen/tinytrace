#![allow(dead_code)]

use glam::Vec4;
use std::mem;
use tinytrace_backend::{
    binding, Binding, BindingType, Buffer, BufferRequest, BufferType, BufferWrite, Context, Error,
    Extent, Handle, Lifetime, MemoryLocation, Shader, ShaderRequest,
};

use crate::asset::material::MaterialConstants;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable, bytemuck::Pod)]
struct BsdfEvalSample {
    wi: Vec4,
    wo: Vec4,
    normal: Vec4,
    tangent: Vec4,
    bitangent: Vec4,
    bary: Vec4,
    position: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Zeroable, bytemuck::Pod)]
struct BsdfEvalResult {
    bsdf: Vec4,
    pdf: Vec4,
}

pub struct BdsfSampler {
    shader: Handle<Shader>,
}

impl BdsfSampler {
    pub fn new(context: &mut Context) -> Result<Self, Error> {
        let shader = format!(
            r#"
                #include "material"
                #include "fuzz_brdf"
                #include "coat_brdf"
                #include "specular_brdf"
                #include "specular_btdf"
                #include "metal_brdf"
                #include "diffuse_brdf"
                #include "bsdf"

                struct BsdfEvalSample {{
                    vec4 wi;
                    vec4 wo;
                    vec4 normal;
                    vec4 tangent;
                    vec4 bitangent;
                    vec4 bary;
                    vec4 position;
                }};

                struct BsdfEvalResult {{
                    vec4 bsdf;
                    vec4 pdf;
                }};

                #include "<bindings>"
                void main() {{
                    uint index = gl_GlobalInvocationID.x;
                    BsdfEvalSample s = samples[index];

                    uint rnd = 0;
                    MaterialConstants mat = material;
                    Generator generator = Generator(rnd);
                    Lobes lobes = bsdf_prepare(mat, s.wi.xyz, generator);
                    rnd = generator.state;

                    float pdf;
                    BsdfEvalResult r;
                    r.bsdf = bsdf_evaluate(
                        mat, lobes, s.wi.xyz, s.wo.xyz, pdf
                    ).xyzx;
                    r.bsdf.w = 1.0;
                    r.pdf = vec4(pdf);
                    results[index] = r;
                }}
            "#
        );
        let shader = context.create_shader(
            Lifetime::Static,
            &ShaderRequest {
                source: &shader,
                block_size: Extent::new(1, 1),
                bindings: &[
                    binding!(storage_buffer, BsdfEvalSample, samples, true, false),
                    binding!(storage_buffer, BsdfEvalResult, results, true, true),
                    binding!(storage_buffer, MaterialConstants, material, false, false),
                ],
                push_constant_size: None,
            },
        )?;
        Ok(Self { shader })
    }

    fn create_buffer(context: &mut Context, size: u64) -> Result<Handle<Buffer>, Error> {
        context.create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
                size,
            },
        )
    }

    fn sample(
        &self,
        context: &mut Context,
        samples: &[BsdfEvalSample],
        material: MaterialConstants,
    ) -> Result<Vec<BsdfEvalResult>, Error> {
        let sample_buffer = Self::create_buffer(
            context,
            (samples.len() * mem::size_of::<BsdfEvalSample>()) as u64,
        )?;
        let result_buffer = Self::create_buffer(
            context,
            (samples.len() * mem::size_of::<BsdfEvalResult>()) as u64,
        )?;
        let material_buffer =
            Self::create_buffer(context, mem::size_of::<MaterialConstants>() as u64)?;
        context
            .write_buffers(&[
                BufferWrite {
                    buffer: sample_buffer.clone(),
                    data: bytemuck::cast_slice(&samples).into(),
                },
                BufferWrite {
                    buffer: material_buffer.clone(),
                    data: bytemuck::bytes_of(&material).into(),
                },
            ])
            .unwrap();
        context
            .bind_shader(&self.shader)
            .bind_buffer("samples", &sample_buffer)
            .bind_buffer("results", &result_buffer)
            .bind_buffer("material", &material_buffer)
            .dispatch(samples.len() as u32, 1)
            .unwrap();
        let download = context.download(&[result_buffer.clone()], &[]).unwrap();
        let results: &[BsdfEvalResult] = bytemuck::cast_slice(&download.buffers[&result_buffer]);

        Ok(results.to_vec())
    }
}
