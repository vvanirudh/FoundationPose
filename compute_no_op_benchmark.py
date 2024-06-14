from estimater import *
from datareader import *
import argparse
from scipy.spatial.transform import Rotation as R
from Utils import add_err, adds_err

def visualize(pose, to_origin, K, rgb, bbox, only_axes=False, wait_for_key=False):
    """Visualize"""
    center_pose = pose.copy()
    if not only_axes:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(
        rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True
    )
    cv2.imshow('1', vis[..., ::-1])
    if wait_for_key:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def initialize_pose_estimator(mesh):
    """Initialize the pose estimation model"""
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    return est

def compute_translations_rotations(poses):
    """Computes translation vector and rotations from 4x4 pose matrices"""
    translations = [pose[:3, -1] for pose in poses]
    rotations = [R.from_matrix(pose[:3, :3]) for pose in poses]
    return translations, rotations

def compute_translation_error_m(translations_1, translations_2):
    """Computes translation error (in m) between two sets of translation vectors"""
    assert len(translations_1) == len(translations_2), "Should have same number of elements"
    relative_translations = [
        translations_1[idx] - translations_2[idx]
        for idx in range(len(translations_1))
    ]
    relative_translations_x_m = [translation[0] for translation in relative_translations]
    relative_translations_y_m = [translation[1] for translation in relative_translations]
    relative_translations_z_m = [translation[2] for translation in relative_translations]
    return [
        max(relative_translations_x_m), max(relative_translations_y_m), max(relative_translations_z_m)
    ]

def compute_rotation_error_deg(rotations_1, rotations_2):
    """Computes rotation error (in deg) between two sets of rotations"""
    assert len(rotations_1) == len(rotations_2), "Should have the same number of elements"
    relative_rotations = [
        rotations_1[idx] * rotations_2[idx].inv()
        for idx in range(len(rotations_1))
    ]
    relative_rotations_euler = [
        relative_rotation.as_euler('xyz')
        for relative_rotation in relative_rotations
    ]
    relative_rotations_x_deg = np.rad2deg([euler[0] for euler in relative_rotations_euler])
    relative_rotations_y_deg = np.rad2deg([euler[1] for euler in relative_rotations_euler])
    relative_rotations_z_deg = np.rad2deg([euler[2] for euler in relative_rotations_euler])
    return [
        max(relative_rotations_x_deg), max(relative_rotations_y_deg), max(relative_rotations_z_deg)
    ]


def compute_no_op_benchmark(args):
    """
    Computes predicted pose for successive frames of a stationary
    object, and computes error between the predicted poses
    """
    mesh = trimesh.load(args.mesh_file)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    est = initialize_pose_estimator(mesh)

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    poses = []
    pose_detected = False
    for i in range(len(reader.color_files)):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)
        # NOTE: Only the first image has the mask
        mask = reader.get_mask(0).astype(bool)

        if (not args.use_tracking) or (not pose_detected):
            pose = est.register(
                K=reader.K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=args.est_refine_iter,
            ).reshape(4, 4)
            pose_detected = True
        else:
            pose = est.track_one(
                rgb=rgb,
                depth=depth,
                K=reader.K,
                iteration=args.track_refine_iter,
            ).reshape(4, 4)
        poses.append(pose.copy())

        if args.debug:
            visualize(pose, to_origin, reader.K, rgb, bbox, only_axes=(not args.draw_bbox), wait_for_key=args.wait_for_key)

    translations, rotations = compute_translations_rotations(poses)
    add_errs = [add_err(poses[idx], poses[idx+1], mesh.vertices) for idx in range(len(poses)-1)]
    adds_errs = [adds_err(poses[idx], poses[idx+1], mesh.vertices) for idx in range(len(poses)-1)]

    # Compute maximum translation error and rotation error
    translation_error_m = compute_translation_error_m(translations[:-1], translations[1:])
    rotation_error_deg = compute_rotation_error_deg(rotations[:-1], rotations[1:])
    add_error_m = max(add_errs)
    adds_error_m = max(adds_errs)


    print("Translation error is", translation_error_m, "meters")
    print("Rotation error is", rotation_error_deg, "degrees")
    print("ADD error is", add_error_m, "meters")
    print("ADD-S error is", adds_error_m, "meters")

def create_parser():
    """Create CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj'
    )
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--use_tracking', action='store_true')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--wait_for_key', action='store_true')
    parser.add_argument('--draw_bbox', action='store_true')

    return parser

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)

    compute_no_op_benchmark(args)


if __name__ == '__main__':
    main()
