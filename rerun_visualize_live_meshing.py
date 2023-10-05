import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

from experiment_modules.depth_model import DepthModel
import options
from tools import fusers_helper
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu
from utils.geometry_utils import NormalGenerator


import modules.cost_volume as cost_volume
import rerun as rr
from rerun.components import MeshProperties
from utils.visualization_utils import reverse_imagenet_normalize, colormap_image


from typing import Dict, Any

# depth prediction normals computer
PRED_FORMAT_SIZE = [192, 256]
compute_normals = NormalGenerator(PRED_FORMAT_SIZE[0], PRED_FORMAT_SIZE[1]).cuda()


def log_source_data(src_entity_path: str, src_data: Dict[str, Any]) -> None:
    src_images_k3hw = reverse_imagenet_normalize(
        torch.tensor(src_data["image_b3hw"][0].cuda())
    )
    num_src_cameras = src_data["world_T_cam_b44"][0].shape[0]
    for src_idx in range(num_src_cameras):
        src_cam_path = f"{src_entity_path}/{src_idx}"
        world_T_cam_44 = src_data["world_T_cam_b44"][0][src_idx].squeeze().cpu().numpy()
        K_44 = src_data["K_s0_b44"][0][src_idx].squeeze().cpu().numpy()
        log_camera(src_cam_path, world_T_cam_44, K_44)
        log_image(src_cam_path, src_images_k3hw[src_idx], denormalize=False)


def log_camera(
    entity_path: str, world_T_cam_44: torch.Tensor, K_44: torch.Tensor
) -> None:
    assert world_T_cam_44.shape == (4, 4)
    assert K_44.shape == (4, 4)
    # Convert and log camera parameters
    Rot, trans = world_T_cam_44[:3, :3], world_T_cam_44[:3, 3]
    K_33 = K_44[:3, :3]

    rr.log(
        f"{entity_path}/image",
        rr.Pinhole(
            image_from_camera=K_33,
            width=PRED_FORMAT_SIZE[1],
            height=PRED_FORMAT_SIZE[0],
        )
    )

    rr.log(entity_path, rr.Transform3D(translation=trans, mat3x3=Rot))


def log_image(
    entity_path: str, color_frame_b3hw: torch.Tensor, denormalize=True
) -> None:
    # Image logging
    color_frame_3hw = color_frame_b3hw.squeeze(0)
    if denormalize:
        main_color_3hw = reverse_imagenet_normalize(color_frame_3hw)
    else:
        main_color_3hw = color_frame_3hw
    pil_image = Image.fromarray(
        np.uint8(main_color_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_image = pil_image.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log(f"{entity_path}/image/rgb", rr.Image(pil_image))


def log_rerun(
    entity_path: str,
    cur_data: Dict[str, Any],
    src_data: Dict[str, Any],
    outputs: Dict[str, Any],
    scene_trimesh_mesh: trimesh.Trimesh,
    should_log_source_cams: bool = True,
) -> None:
    """
    Logs camera intri/extri, depth, rgb, and mesh to rerun.
    """
    curr_entity_path = f"{entity_path}/current_cam"
    src_entity_path = f"{entity_path}/source_cam"
    if should_log_source_cams:
        log_source_data(src_entity_path, src_data)

    world_T_cam_44 = cur_data["world_T_cam_b44"].squeeze().cpu().numpy()
    K_44 = cur_data["K_s0_b44"].squeeze().cpu().numpy()
    log_camera(curr_entity_path, world_T_cam_44, K_44)

    # Depth logging
    depth_pred = outputs["depth_pred_s0_b1hw"]
    our_depth_3hw = depth_pred.squeeze(0)
    our_depth_hw3 = our_depth_3hw.permute(1, 2, 0)
    rr.log(
        f"{curr_entity_path}/image/depth",
        rr.DepthImage(our_depth_hw3.numpy(force=True))
    )

    # Normal logging
    invK_s0_b44 = cur_data["invK_s0_b44"].cuda()
    normals_b3hw = compute_normals(depth_pred, invK_s0_b44)
    our_normals_3hw = 0.5 * (1 + normals_b3hw).squeeze(0)
    pil_normal = Image.fromarray(
        np.uint8(our_normals_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    rr.log(f"{curr_entity_path}/image/normal", rr.Image(pil_normal))

    # Image logging
    color_frame_b3hw = (
        cur_data["high_res_color_b3hw"]
        if "high_res_color_b3hw" in cur_data
        else cur_data["image_b3hw"]
    )
    color_frame_3hw = color_frame_b3hw.squeeze(0)
    main_color_3hw = reverse_imagenet_normalize(color_frame_3hw)
    pil_image = Image.fromarray(
        np.uint8(main_color_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_image = pil_image.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log(f"{curr_entity_path}/image/rgb", rr.Image(pil_image))

    # lowest cost guess from the cost volume
    lowest_cost_bhw = outputs["lowest_cost_bhw"]
    lowest_cost_3hw = colormap_image(
        lowest_cost_bhw,
        vmin=0,
        vmax=5,
    )
    pil_cost = Image.fromarray(
        np.uint8(lowest_cost_3hw.permute(1, 2, 0).cpu().detach().numpy() * 255)
    )
    pil_cost = pil_cost.resize((PRED_FORMAT_SIZE[1], PRED_FORMAT_SIZE[0]))
    rr.log("lowest_cost_volume", rr.Image(pil_cost))

    # Fused mesh logging
    rr.log(
        f"{entity_path}/mesh",
        rr.Mesh3D(
            vertex_positions=scene_trimesh_mesh.vertices,
            indices=scene_trimesh_mesh.faces,
            vertex_colors=scene_trimesh_mesh.visual.vertex_colors,
        )
    )


def main(opts):
    print("Setting batch size to 1.")
    opts.batch_size = 1

    # get dataset
    dataset_class, scans = get_dataset(
        opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id
    )

    model = DepthModel.load_from_checkpoint(
        opts.load_weights_from_checkpoint, args=None
    )
    if opts.fast_cost_volume and isinstance(
        model.cost_volume, cost_volume.FeatureVolumeManager
    ):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(
        opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type
    )

    mesh_output_folder_name = (
        f"{opts.fusion_resolution}_{opts.fusion_max_depth}_{opts.depth_fuser}"
    )

    if opts.mask_pred_depth:
        mesh_output_folder_name += "_masked"
    if opts.fuse_color:
        mesh_output_folder_name += "_color"

    incremental_mesh_output_dir = os.path.join(
        results_path, "incremental_meshes", mesh_output_folder_name
    )

    Path(incremental_mesh_output_dir).mkdir(parents=True, exist_ok=True)
    print("".center(80, "#"))
    print(f" Running Fusion! Using {opts.depth_fuser} ".center(80, "#"))
    print(
        f"Incremental Mesh Output directory:"
        f"\n{incremental_mesh_output_dir} ".center(80, "#")
    )
    if opts.use_precomputed_partial_meshes:
        print(" Loading precomputed incremental meshes. ".center(80, "#"))
    print("".center(80, "#"))
    print("")

    # path where cached depth maps are
    depth_output_dir = os.path.join(results_path, "depths")
    Path(os.path.join(depth_output_dir)).mkdir(parents=True, exist_ok=True)
    print("".center(80, "#"))
    print(" Reading cached depths if they exist. ".center(80, "#"))
    print(f"Directory:\n{depth_output_dir} ".center(80, "#"))
    if opts.cache_depths:
        print(" Caching depths if we need to compute them. ".center(80, "#"))
    print("".center(80, "#"))
    print("")

    video_output_dir = os.path.join(
        results_path, "viz", "reconstruction_videos", mesh_output_folder_name
    )
    Path(os.path.join(video_output_dir)).mkdir(parents=True, exist_ok=True)
    print("".center(80, "#"))
    print(" Outputting videos. ".center(80, "#"))
    print(f"Video Output directory:\n{video_output_dir} ".center(80, "#"))
    print("".center(80, "#"))
    print("")

    with torch.inference_mode():
        for scan in tqdm(scans):
            entity_path = f"{scan}/world"
            rr.log(entity_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
            Path(os.path.join(incremental_mesh_output_dir, scan)).mkdir(
                parents=True, exist_ok=True
            )

            # initialize fuser if we need to fuse
            if opts.run_fusion:
                fuser = fusers_helper.get_fuser(opts, scan)

            # set up dataset with current scan
            dataset = dataset_class(
                opts.dataset_path,
                split=opts.split,
                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=opts.fuse_color and opts.run_fusion,
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                skip_to_frame=opts.skip_to_frame,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            viz_depth_panel = True
            all_meshes_precomputed = True
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                cur_data, src_data = batch
                if "frame_id_string" in cur_data:
                    frame_id = cur_data["frame_id_string"][0]
                else:
                    frame_id = f"{str(batch_ind):6d}"

                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                # To save time and compute , we should load meshes if they've
                # all been computed and stored. We don't currently have a
                # mechanism for picking up fusion from a partial mesh. We should
                # only load and continue vizzing if we have a continious stream
                # of saved meshes. If this panics, run this script without
                # loading partial meshes
                trimesh_path = os.path.join(
                    incremental_mesh_output_dir, scan, f"{frame_id}.ply"
                )
                if not Path(trimesh_path).is_file():
                    all_meshes_precomputed = False

                if all_meshes_precomputed and opts.use_precomputed_partial_meshes:
                    scene_trimesh_mesh = trimesh.load(trimesh_path, force="mesh")

                    if viz_depth_panel:
                        pickled_depths_path = os.path.join(
                            depth_output_dir, scan, f"{frame_id}.pickle"
                        )

                        if Path(pickled_depths_path).is_file():
                            with open(pickled_depths_path, "rb") as handle:
                                outputs = pickle.load(handle)
                        else:
                            outputs = model(
                                "test",
                                cur_data,
                                src_data,
                                unbatched_matching_encoder_forward=True,
                                return_mask=True,
                            )

                        depth_pred = outputs["depth_pred_s0_b1hw"]

                else:
                    if not opts.run_fusion:
                        raise Exception(
                            "No precomputed partial mesh found and "
                            "run_fusion is disabled."
                        )

                    # check if depths are precomputed.
                    pickled_depths_path = os.path.join(
                        depth_output_dir, scan, f"{frame_id}.pickle"
                    )

                    if Path(pickled_depths_path).is_file():
                        with open(pickled_depths_path, "rb") as handle:
                            outputs = pickle.load(handle)
                    else:
                        outputs = model(
                            "test",
                            cur_data,
                            src_data,
                            unbatched_matching_encoder_forward=True,
                            return_mask=True,
                        )

                        if opts.cache_depths:
                            Path(os.path.join(depth_output_dir, scan)).mkdir(
                                parents=True, exist_ok=True
                            )

                            output_path = os.path.join(
                                depth_output_dir, scan, f"{frame_id}.pickle"
                            )

                            outputs["K_full_depth_b44"] = cur_data["K_full_depth_b44"]
                            outputs["K_s0_b44"] = cur_data["K_s0_b44"]
                            outputs["frame_id"] = frame_id
                            if "frame_id" in src_data:
                                outputs["src_ids"] = src_data["frame_id_string"]

                            with open(output_path, "wb") as handle:
                                pickle.dump(outputs, handle)

                    depth_pred = outputs["depth_pred_s0_b1hw"]

                    if opts.mask_pred_depth:
                        overall_mask_b1hw = (
                            outputs["overall_mask_bhw"].cuda().unsqueeze(1).float()
                        )
                        overall_mask_b1hw = F.interpolate(
                            overall_mask_b1hw, size=(192, 256), mode="nearest"
                        ).bool()
                        depth_pred[~overall_mask_b1hw] = 0

                    color_frame = (
                        cur_data["high_res_color_b3hw"]
                        if "high_res_color_b3hw" in cur_data
                        else cur_data["image_b3hw"]
                    )
                    fuser.fuse_frames(
                        depth_pred,
                        cur_data["K_s0_b44"],
                        cur_data["cam_T_world_b44"],
                        color_frame,
                    )

                    Path(os.path.join(incremental_mesh_output_dir, scan)).mkdir(
                        parents=True, exist_ok=True
                    )
                    mesh_path = os.path.join(
                        incremental_mesh_output_dir, scan, f"{frame_id}.ply"
                    )
                    fuser.export_mesh(path=mesh_path)

                    if opts.fuse_color:
                        scene_trimesh_mesh = trimesh.load(trimesh_path, force="mesh")
                    else:
                        scene_trimesh_mesh = fuser.get_mesh(convert_to_trimesh=True)

                rr.set_time_sequence("frame", int(batch_ind))
                log_rerun(entity_path, cur_data, src_data, outputs, scene_trimesh_mesh)

            del dataloader
            del dataset
            break


if __name__ == "__main__":
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    rr.script_add_args(option_handler.parser)
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    rr.script_setup(option_handler.cl_args, "SimpleRecon")
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
    rr.script_teardown(option_handler.cl_args)
