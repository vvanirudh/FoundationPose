from estimater import *
from datareader import *
from scipy.spatial.transform import Rotation as R
from compute_no_op_benchmark import initialize_pose_estimator, visualize, create_parser, compute_translations_rotations, compute_translation_error_m, compute_rotation_error_deg
from Utils import add_err, adds_err


def compute_benchmark(args):
    """Computes the max error between predicted pose and GT pose from AR tags for all frames"""
    mesh = trimesh.load(args.mesh_file)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    est = initialize_pose_estimator(mesh)

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    poses = []
    ar_tag_poses = []
    pose_detected = False
    for i in range(len(reader.color_files)):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)
        # NOTE: Only the first image has the mask
        mask = reader.get_mask(0).astype(bool)
        ar_tag_pose = reader.get_ar_tag_pose(i)

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

        if ar_tag_pose.size == 0:
            print("No GT found")
            continue
        assert ar_tag_pose.shape[0] == 1, "Currently only supports 1 AR tag"
        ar_tag_pose = ar_tag_pose[0]
        poses.append(pose.copy())
        ar_tag_poses.append(ar_tag_pose.copy())

        if args.debug:
            visualize(
                pose, to_origin, reader.K, rgb, bbox,
                only_axes=(not args.draw_bbox), wait_for_key=args.wait_for_key
            )
            visualize(
                ar_tag_pose, to_origin, reader.K, rgb, bbox,
                  only_axes=(not args.draw_bbox), wait_for_key=args.wait_for_key
            )


    est_translations, est_rotations = compute_translations_rotations(poses)
    gt_translations, gt_rotations = compute_translations_rotations(ar_tag_poses)
    add_errs = [add_err(poses[idx], ar_tag_poses[idx], mesh.vertices) for idx in range(len(poses))]
    adds_errs = [adds_err(poses[idx], ar_tag_poses[idx], mesh.vertices) for idx in range(len(poses))]

    translation_error_m = compute_translation_error_m(est_translations, gt_translations)
    rotation_error_deg = compute_rotation_error_deg(est_rotations, gt_rotations)
    add_error_m = max(add_errs)
    adds_error_m = max(adds_errs)

    print("Translation error is", translation_error_m, "meters")
    print("Rotation error is", rotation_error_deg, "degrees")
    print("ADD error is", add_error_m, "meters")
    print("ADD-S error is", adds_error_m, "meters")


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)

    compute_benchmark(args)

if __name__ == '__main__':
    main()