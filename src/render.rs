//! Batched trail rendering: all trails are packed into shared GPU storage
//! buffers and drawn with a single instanced draw call per blend mode, rather
//! than one draw call + bind group per trail. Geometry is expanded in the
//! vertex shader (camera-facing billboards), so there are no vertex buffers.

use crate::types::{TrailData, TrailHeader, TrailPoint, TrailRenderMode, TrailStyle};
use bevy::{
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    prelude::*,
    render::{
        extract_component::ExtractComponent,
        extract_resource::ExtractResource,
        render_phase::{
            PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{
            encase::StorageBuffer as EncaseStorageBuffer, BindGroup, BindGroupEntries,
            BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferDescriptor,
            BufferUsages, Canonical, RenderPipeline, RenderPipelineDescriptor, Specializer,
            SpecializerKey, TextureFormat,
        },
        renderer::{RenderDevice, RenderQueue},
        view::{ViewTarget, ViewUniformOffset, ViewUniforms},
    },
};

mod plugin_2d;
pub use plugin_2d::{Trail2dPipeline, TrailRender2dPlugin};
mod plugin_3d;
pub use plugin_3d::{Trail3dPipeline, TrailRender3dPlugin};

/// The three blend modes, in [`TrailRenderMode`] discriminant order, so a mode
/// round-trips through `mode as usize` and back.
const MODES: [TrailRenderMode; 3] = [
    TrailRenderMode::Opaque,
    TrailRenderMode::Additive,
    TrailRenderMode::Transparent,
];

/// Per-blend-mode draw anchor; the renderer owns one per [`TrailRenderMode`] as
/// the representative entity for that batch's phase item.
///
/// Mesh-less on purpose: the binned phase caches per `MainEntity` across item
/// kinds, so reusing a real trail entity (which may draw its own mesh) would
/// make the two items fight over one slot and drop both. The mode lives here
/// (not as a [`TrailRenderMode`] component) so anchors stay out of user queries.
#[derive(Component, ExtractComponent, Clone, Copy)]
struct TrailBatchAnchor {
    mode: TrailRenderMode,
}

/// Binds one blend mode's batched buffers and issues a single instanced draw
/// covering every trail in that batch.
struct DrawTrailBatch;

impl<P> RenderCommand<P> for DrawTrailBatch
where
    P: PhaseItem,
{
    type Param = SRes<TrailBatches>;
    type ViewQuery = ();
    // The phase item's entity is the batch anchor; its mode tells us which batch
    // this draw is for.
    type ItemQuery = bevy::ecs::system::lifetimeless::Read<TrailBatchAnchor>;

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        anchor: Option<&'w TrailBatchAnchor>,
        param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(anchor) = anchor else {
            return RenderCommandResult::Skip;
        };
        let batch = &param.into_inner().modes[anchor.mode as usize];

        let Some(bind_group) = batch.bind_group.as_ref() else {
            return RenderCommandResult::Skip;
        };
        if batch.instance_count == 0 || batch.max_verts == 0 {
            return RenderCommandResult::Skip;
        }

        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..batch.max_verts, 0..batch.instance_count);

        RenderCommandResult::Success
    }
}

/// A GPU storage buffer that grows on demand and remembers its capacity, so a
/// steady or shrinking trail count doesn't reallocate every frame.
#[derive(Default)]
struct GrowBuffer {
    buffer: Option<Buffer>,
    capacity: u64,
}

/// Per-blend-mode batch of trails: shared GPU buffers, one bind group, and the
/// instanced draw parameters. Buffers are kept across frames and only grown
/// (which forces a bind-group rebuild) when the trail count rises.
#[derive(Default)]
struct GpuBatch {
    headers: GrowBuffer,
    points: GrowBuffer,
    styles: GrowBuffer,
    bind_group: Option<BindGroup>,
    /// Number of trails (= instances) in this batch.
    instance_count: u32,
    /// Vertices per instance: 2 per live ring point, sized to the longest trail.
    max_verts: u32,
}

#[derive(Resource, Default)]
struct TrailBatches {
    /// One batch per blend mode, indexed by `TrailRenderMode as usize`.
    modes: [GpuBatch; 3],
}

/// Whether any trail changed this frame. Extracted to the render world so it can
/// skip the whole repack when nothing changed (e.g. a static scene).
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
struct TrailsChanged(bool);

/// Sets [`TrailsChanged`] if any trail's data, style, or mode changed, or one
/// was spawned (reads as `Changed`) or despawned. O(changes), not O(trails).
fn detect_trail_changes(
    data_changed: Query<(), Changed<TrailData>>,
    style_changed: Query<(), Changed<TrailStyle>>,
    mode_changed: Query<(), Changed<TrailRenderMode>>,
    removed: RemovedComponents<TrailData>,
    mut changed: ResMut<TrailsChanged>,
) {
    changed.0 = !data_changed.is_empty()
        || !style_changed.is_empty()
        || !mode_changed.is_empty()
        || !removed.is_empty();
}

/// Bind group for the view uniform (camera matrices), bound at group 0 so the
/// shader can project world-space points into clip space. Rebuilt each frame.
#[derive(Resource, Default)]
struct TrailViewBindGroup(Option<BindGroup>);

/// Binds the view uniform at group `I` with the per-view dynamic offset.
struct SetTrailViewBindGroup<const I: usize>;

impl<P, const I: usize> RenderCommand<P> for SetTrailViewBindGroup<I>
where
    P: PhaseItem,
{
    type Param = SRes<TrailViewBindGroup>;

    type ViewQuery = bevy::ecs::system::lifetimeless::Read<ViewUniformOffset>;

    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<()>,
        param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bind_group) = param.into_inner().0.as_ref() else {
            return RenderCommandResult::Failure("View bind group missing");
        };

        pass.set_bind_group(I, bind_group, &[view_offset.offset]);

        RenderCommandResult::Success
    }
}

/// Scratch accumulators for building one batch, reused across frames.
#[derive(Default)]
struct BatchScratch {
    headers: Vec<TrailHeader>,
    points: Vec<TrailPoint>,
    styles: Vec<TrailStyle>,
    bytes: Vec<u8>,
}

/// Encodes `data` (a `ShaderType` array) into `scratch`, ensures `buffer` is at
/// least that large (reallocating with growth headroom if not), and queues the
/// upload. Returns `true` if the buffer was reallocated (so the caller knows to
/// rebuild the bind group).
fn ensure_and_write<T>(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    target: &mut GrowBuffer,
    data: &[T],
    scratch: &mut Vec<u8>,
    label: &'static str,
) -> bool
where
    T: bevy::render::render_resource::ShaderType
        + bevy::render::render_resource::encase::ShaderSize
        + bevy::render::render_resource::encase::internal::WriteInto,
{
    scratch.clear();
    {
        let mut wrapper = EncaseStorageBuffer::new(&mut *scratch);
        wrapper.write(data).unwrap();
    }
    let size = scratch.len() as u64;

    let mut reallocated = false;
    if target.buffer.is_none() || target.capacity < size {
        // Grow with 50% headroom so a steadily increasing trail count doesn't
        // reallocate every frame.
        let new_cap = (size + size / 2).max(size);
        target.buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: new_cap,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        target.capacity = new_cap;
        reallocated = true;
    }

    render_queue.write_buffer(target.buffer.as_ref().unwrap(), 0, scratch);
    reallocated
}

/// The custom draw commands Bevy executes for each batch phase item.
type DrawTrailCommands = (SetItemPipeline, SetTrailViewBindGroup<0>, DrawTrailBatch);

#[derive(Copy, Clone, PartialEq, Eq, Hash, SpecializerKey)]
pub struct TrailPipelineKey {
    pub msaa: Msaa,
    pub mode: TrailRenderMode,
    /// Whether the target view is HDR, so the color target format matches it.
    pub hdr: bool,
}

pub struct TrailPipelineSpecializer;

impl Specializer<RenderPipeline> for TrailPipelineSpecializer {
    type Key = TrailPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        descriptor.multisample.count = key.msaa.samples();

        // Pick the blend state for this trail's render mode. The fragment shader
        // outputs straight (non-premultiplied) color, so additive scales the
        // contribution by alpha and transparent uses standard alpha blending.
        let blend = match key.mode {
            TrailRenderMode::Opaque => None,
            TrailRenderMode::Additive => Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            }),
            TrailRenderMode::Transparent => Some(BlendState::ALPHA_BLENDING),
        };

        // Match the view's render target format so HDR cameras (e.g. for bloom)
        // don't hit a pipeline/target format mismatch.
        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        if let Some(fragment) = descriptor.fragment.as_mut() {
            for target in fragment.targets.iter_mut().flatten() {
                target.format = format;
                target.blend = blend;
            }
        }

        Ok(key)
    }
}
