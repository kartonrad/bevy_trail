use bevy::prelude::*;
use bevy_trail::prelude::*;

#[derive(Component)]
struct CircleMover {
    radius: f32,
    speed: f32,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, Trail3dPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, move_in_circle_system)
        .run();
}

fn setup(
    mut commands: Commands,
    mut mesh: ResMut<Assets<Mesh>>,
    mut material: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Mesh2d(mesh.add(Circle::new(20.0))),
        MeshMaterial2d(material.add(ColorMaterial::from_color(Color::WHITE))),
        Transform::from_xyz(100.0, 0.0, 0.0),
        Trail::new(64).with_max_length(16.0).with_max_time(8.0),
        TrailStyle {
            start_color: LinearRgba::WHITE,
            end_color: LinearRgba::RED,
            start_width: 40.0,
            ..default()
        },
        TrailEmitter::default(),
        CircleMover {
            radius: 150.0,
            speed: 5.0,
        },
    ));

    commands.spawn((Camera2d::default(),));
}

fn move_in_circle_system(time: Res<Time>, mut movers: Query<(&mut Transform, &CircleMover)>) {
    for (mut transform, mover) in &mut movers {
        let angle = time.elapsed_secs() * mover.speed;
        transform.translation =
            Vec3::new(mover.radius * angle.cos(), mover.radius * angle.sin(), 0.0);
    }
}
