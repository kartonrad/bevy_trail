use crate::{
    render::*,
    types::{TrailData, TrailRenderMode, TrailStyle},
};
use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::core_2d::{BatchSetKey2d, Opaque2d, Opaque2dBinKey, CORE_2D_DEPTH_FORMAT},
    ecs::change_detection::Tick,
    mesh::PrimitiveTopology,
    render::{
        extract_component::ExtractComponentPlugin,
        extract_resource::ExtractResourcePlugin,
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            binding_types::{storage_buffer_read_only_sized, uniform_buffer},
            BindGroupLayoutDescriptor, BindGroupLayoutEntries, ColorTargetState, ColorWrites,
            CompareFunction, DepthBiasState, DepthStencilState, FragmentState, PipelineCache,
            PrimitiveState, RenderPipeline, RenderPipelineDescriptor, ShaderStages,
            StencilFaceState, StencilState, TextureFormat, Variants, VertexState,
        },
        sync_world::MainEntity,
        view::{ExtractedView, ViewUniform},
        Render, RenderApp, RenderSystems,
    },
};

pub struct TrailRender2dPlugin;

impl Plugin for TrailRender2dPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "src/", "shaders/trail_drawing2d.wgsl");

        // Main World
        app.add_plugins((
            ExtractComponentPlugin::<TrailData>::default(),
            ExtractComponentPlugin::<TrailStyle>::default(),
            ExtractComponentPlugin::<TrailRenderMode>::default(),
            ExtractComponentPlugin::<TrailBatchAnchor>::default(),
            ExtractResourcePlugin::<TrailsChanged>::default(),
        ))
        // Runs after the Update systems that mutate trails (emit/sync) and
        // before extract, so the flag reflects the same data that gets extracted.
        .init_resource::<TrailsChanged>()
        .add_systems(Last, detect_trail_changes);

        // One anchor entity per blend mode (see `TrailBatchAnchor`).
        for mode in MODES {
            app.world_mut().spawn(TrailBatchAnchor { mode });
        }

        // Render World
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_render_command::<Opaque2d, DrawTrailCommands>()
            .init_resource::<TrailViewBindGroup>()
            .init_resource::<TrailBatches>()
            .add_systems(
                Render,
                (
                    prepare_trail_batches.in_set(RenderSystems::PrepareResources),
                    prepare_trail_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    queue_custom_phase_item2d.in_set(RenderSystems::Queue),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        // TrailPipeline needs RenderDevice, which isn't created until App::run.
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<Trail2dPipeline>();
    }
}

/// Rebuilds the view bind group each frame from the current [`ViewUniforms`].
fn prepare_trail_view_bind_group(
    mut bind_group: ResMut<TrailViewBindGroup>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<Trail2dPipeline>,
    view_uniforms: Res<ViewUniforms>,
) {
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    let view_layout = pipeline_cache.get_bind_group_layout(&pipeline.view_layout);
    bind_group.0 = Some(render_device.create_bind_group(
        "trail_view_bind_group",
        &view_layout,
        &BindGroupEntries::single(view_binding),
    ));
}

/// Packs every trail into per-mode shared storage buffers (headers, ring points,
/// styles) and uploads them, ready for one instanced draw per mode.
#[allow(clippy::too_many_arguments)]
fn prepare_trail_batches(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<Trail2dPipeline>,
    pipeline_cache: Res<PipelineCache>,
    changed: Res<TrailsChanged>,
    mut batches: ResMut<TrailBatches>,
    trails: Query<(&TrailData, &TrailStyle, &TrailRenderMode)>,
    mut scratch: Local<[BatchScratch; 3]>,
) {
    // Nothing changed → last frame's buffers and bind groups are still valid.
    // (The first frame reports changed, so they build then.)
    if !changed.0 {
        return;
    }

    for s in scratch.iter_mut() {
        s.headers.clear();
        s.points.clear();
        s.styles.clear();
    }

    let mut max_verts = [0u32; 3];

    for (trail, style, mode) in &trails {
        let m = *mode as usize;
        let s = &mut scratch[m];

        // The shader walks `capacity` points from `offset`, so exactly that many
        // must be uploaded. Catch a hand-built instance that broke the invariant.
        debug_assert_eq!(
            trail.cpu_data.len(),
            trail.header.capacity as usize,
            "TrailData invariant violated: cpu_data.len() must equal header.capacity"
        );

        let mut header = trail.header.clone();
        header.offset = s.points.len() as u32;
        let length = header.length;
        s.headers.push(header);
        s.styles.push(style.clone());
        s.points.extend_from_slice(&trail.cpu_data);

        // Size the draw to live points (2 verts each), not full capacity.
        max_verts[m] = max_verts[m].max(length * 2);
    }

    let layout = pipeline_cache.get_bind_group_layout(&pipeline.batch_layout);

    for m in 0..3 {
        let s = &mut scratch[m];
        let batch = &mut batches.modes[m];

        batch.instance_count = s.headers.len() as u32;
        batch.max_verts = max_verts[m];

        if s.headers.is_empty() {
            batch.bind_group = None;
            continue;
        }

        let h = ensure_and_write(
            &render_device,
            &render_queue,
            &mut batch.headers,
            &s.headers,
            &mut s.bytes,
            "trail_batch_headers",
        );
        let p = ensure_and_write(
            &render_device,
            &render_queue,
            &mut batch.points,
            &s.points,
            &mut s.bytes,
            "trail_batch_points",
        );
        let st = ensure_and_write(
            &render_device,
            &render_queue,
            &mut batch.styles,
            &s.styles,
            &mut s.bytes,
            "trail_batch_styles",
        );

        // Only rebuild the bind group when a buffer was (re)allocated.
        if h || p || st || batch.bind_group.is_none() {
            batch.bind_group = Some(render_device.create_bind_group(
                "trail_batch_bind_group",
                &layout,
                &BindGroupEntries::sequential((
                    batch.headers.buffer.as_ref().unwrap().as_entire_binding(),
                    batch.points.buffer.as_ref().unwrap().as_entire_binding(),
                    batch.styles.buffer.as_ref().unwrap().as_entire_binding(),
                )),
            ));
        }
    }
}

/// Enqueues one batched phase item per (view, non-empty blend mode). Each item
/// expands into a single instanced draw covering all trails of that mode.
#[allow(clippy::too_many_arguments)]
fn queue_custom_phase_item2d(
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<Trail2dPipeline>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque2d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque2d>>,
    batches: Res<TrailBatches>,
    anchors: Query<(Entity, &MainEntity, &TrailBatchAnchor)>,
    views: Query<(&ExtractedView, &Msaa)>,
    mut next_tick: Local<Tick>,
) {
    let draw_custom_phase_item = opaque_draw_functions.read().id::<DrawTrailCommands>();

    // The renderer-owned anchor entity that represents each blend mode's batch.
    let mut anchor_by_mode: [Option<(Entity, MainEntity)>; 3] = [None; 3];
    for (entity, main, anchor) in anchors.iter() {
        anchor_by_mode[anchor.mode as usize] = Some((entity, *main));
    }

    for (view, msaa) in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for m in 0..3 {
            let batch = &batches.modes[m];
            if batch.instance_count == 0 {
                continue;
            }
            let Some(representative) = anchor_by_mode[m] else {
                continue;
            };

            let mode = MODES[m];
            let Ok(pipeline_id) = pipeline.variants.specialize(
                &pipeline_cache,
                TrailPipelineKey {
                    msaa: *msaa,
                    mode,
                    hdr: view.hdr,
                },
            ) else {
                continue;
            };

            // Bump the tick so Bevy rebuilds this item's bin. Only three items
            // exist, so doing it every frame is free; the `Tick` wraps safely.
            let this_tick = next_tick.get() + 1;
            next_tick.set(this_tick);

            opaque_phase.add(
                BatchSetKey2d { indexed: false },
                Opaque2dBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                    pipeline: pipeline_id,
                    draw_function: draw_custom_phase_item,
                    material_bind_group_id: None,
                },
                representative,
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
                *next_tick,
            );
        }
    }
}

#[derive(Resource)]
pub struct Trail2dPipeline {
    /// Specialized pipeline variants; also keeps the shader handle alive via the
    /// base descriptor.
    pub variants: Variants<RenderPipeline, TrailPipelineSpecializer>,
    /// Layout for the view uniform bind group (group 0).
    pub view_layout: BindGroupLayoutDescriptor,
    /// Layout for the batched trail buffers bind group (group 1).
    pub batch_layout: BindGroupLayoutDescriptor,
}

impl FromWorld for Trail2dPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "shaders/trail_drawing2d.wgsl");

        let view_layout = BindGroupLayoutDescriptor::new(
            "trail_view_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX,
                uniform_buffer::<ViewUniform>(true),
            ),
        );

        // group 1: headers, points, styles — all read-only storage arrays.
        let batch_layout = BindGroupLayoutDescriptor::new(
            "trail_batch_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );

        let base_descriptor = RenderPipelineDescriptor {
            label: Some("trail render pipeline".into()),
            layout: vec![view_layout.clone(), batch_layout.clone()],
            vertex: VertexState {
                shader: shader.clone(),
                // No vertex buffers; geometry is generated in the shader.
                ..default()
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                targets: vec![Some(ColorTargetState {
                    // Placeholder format; the specializer overwrites this with
                    // the view's actual format (HDR vs. default) based on
                    // `TrailPipelineKey::hdr`.
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            // Note that if your view has no depth buffer this will need to be
            // changed.
            depth_stencil: Some(DepthStencilState {
                /*
                                format: CORE_3D_DEPTH_FORMAT,
                                depth_write_enabled: false,
                                depth_compare: CompareFunction::Always,
                                stencil: default(),
                                bias: default(),
                */
                format: CORE_2D_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            ..default()
        };

        let variants = Variants::new(TrailPipelineSpecializer, base_descriptor);

        Self {
            variants,
            view_layout,
            batch_layout,
        }
    }
}
