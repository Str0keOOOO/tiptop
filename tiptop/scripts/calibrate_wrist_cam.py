"""Modified from DROID repo: https://github.com/droid-dataset/droid"""

import logging
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R

from tiptop.config import tiptop_cfg, update_calibration_info
from tiptop.perception.cameras import get_hand_camera
from tiptop.perception.cameras.rs_camera import RealsenseIntrinsics
from tiptop.perception.cameras.zed_camera import ZedIntrinsics
from tiptop.utils import get_robot_client, setup_logging

# DFVision Q12-200-15: 12 x 9 squares with 15 mm square spacing.
CHECKERBOARD_SQUARES_X = 12
CHECKERBOARD_SQUARES_Y = 9
CHECKERBOARD_INNER_CORNERS = (CHECKERBOARD_SQUARES_X - 1, CHECKERBOARD_SQUARES_Y - 1)
CHECKERBOARD_SQUARE_SIZE_M = 0.015
CHECKERBOARD_OBJECT_POINTS = np.zeros(
    (CHECKERBOARD_INNER_CORNERS[0] * CHECKERBOARD_INNER_CORNERS[1], 3), dtype=np.float32
)
CHECKERBOARD_OBJECT_POINTS[:, :2] = (
    np.mgrid[0 : CHECKERBOARD_INNER_CORNERS[0], 0 : CHECKERBOARD_INNER_CORNERS[1]]
    .T.reshape(-1, 2)
    .astype(np.float32)
    * CHECKERBOARD_SQUARE_SIZE_M
)
CHECKERBOARD_DETECTION_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


_log = logging.getLogger(__name__)


def _annotate_checkerboard_requirements(image: np.ndarray) -> None:
    """Draw the physical checkerboard specification on a BGR calibration image."""
    lines = (
        f"Board: DFVision Q12-200-15 checkerboard ({CHECKERBOARD_SQUARES_X}x{CHECKERBOARD_SQUARES_Y}, "
        f"{CHECKERBOARD_SQUARE_SIZE_M * 1_000:.1f} mm squares)",
        "Using a different board? Update the calibration code before running.",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 8
    line_height = 20
    text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    x0, y1 = padding, image.shape[0] - padding
    x1 = min(image.shape[1] - padding, x0 + text_width + 2 * padding)
    y0 = max(padding, y1 - len(lines) * line_height - 2 * padding)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, dst=image)
    for index, line in enumerate(lines):
        y = y0 + padding + (index + 1) * line_height - 5
        cv2.putText(image, line, (x0 + padding, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[:3]) - np.array(source[:3])
    rot_diff = angle_diff(target[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff])
    return result


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result


def calibration_traj(t, pos_scale=0.1, angle_scale=0.2, hand_camera=False):
    x = -np.abs(np.sin(3 * t)) * pos_scale
    y = -0.8 * np.sin(2 * t) * pos_scale
    z = 0.5 * np.sin(4 * t) * pos_scale
    a = -np.sin(4 * t) * angle_scale
    b = np.sin(3 * t) * angle_scale
    c = np.sin(2 * t) * angle_scale
    if hand_camera:
        value = np.array([z, y, -x, c / 1.5, b / 1.5, -a / 1.5])
    else:
        value = np.array([x, y, z, a, b, c])
    return value


class CheckerboardDetector:
    def __init__(
        self,
        intrinsics_dict,
        inlier_error_threshold=3.0,
        reprojection_error_threshold=3.0,
        num_img_threshold=10,
        num_corner_threshold=len(CHECKERBOARD_OBJECT_POINTS),
    ):
        # Set Parameters
        self.inlier_error_threshold = inlier_error_threshold
        self.reprojection_error_threshold = reprojection_error_threshold
        self.num_img_threshold = num_img_threshold
        self.num_corner_threshold = num_corner_threshold
        self.intrinsic_params = {}
        self._intrinsics_dict = intrinsics_dict
        self._readings_dict = defaultdict(list)
        self._pose_dict = defaultdict(list)
        self._curr_cam_id = None

    def process_image(self, image):
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError
        img_size = image.shape[:2]

        found, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD_INNER_CORNERS, flags=CHECKERBOARD_DETECTION_FLAGS
        )
        if not found or corners is None or len(corners) < self.num_corner_threshold:
            return None

        return np.ascontiguousarray(corners, dtype=np.float32), img_size

    def add_sample(self, cam_id, image, pose):
        readings = self.process_image(image)
        if readings is None:
            return
        self._readings_dict[cam_id].append(readings)
        self._pose_dict[cam_id].append(pose)

    def calculate_target_to_cam(self, readings, train=True):
        threshold = self.num_img_threshold if train else 5
        if len(readings) < threshold:
            return None
        if isinstance(self._intrinsics_dict, RealsenseIntrinsics):
            cameraMatrix = self._intrinsics_dict.K_color
            distCoeffs = self._intrinsics_dict.distortion_color
        elif isinstance(self._intrinsics_dict, ZedIntrinsics):
            cameraMatrix = self._intrinsics_dict.K_left
            distCoeffs = self._intrinsics_dict.distortion_left
        else:
            raise NotImplementedError
        rmats, tvecs, successes = [], [], []
        for i, (corners, _) in enumerate(readings):
            solved, rvec, tvec = cv2.solvePnP(
                CHECKERBOARD_OBJECT_POINTS, corners, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not solved:
                continue
            projected, _ = cv2.projectPoints(CHECKERBOARD_OBJECT_POINTS, rvec, tvec, cameraMatrix, distCoeffs)
            reprojection_error = float(np.sqrt(np.mean((corners - projected) ** 2)))
            if reprojection_error > self.inlier_error_threshold:
                continue
            rmats.append(R.from_rotvec(rvec.flatten()).as_matrix())
            tvecs.append(tvec.flatten())
            successes.append(i)

        if len(successes) < threshold:
            return None
        return rmats, tvecs, successes

    def augment_image(self, cam_id, image, visualize=False, visual_type=["corners", "axes"]):
        if type(visual_type) != list:
            visual_type = [visual_type]
        assert all([t in ["corners", "axes"] for t in visual_type])
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self._curr_cam_id = cam_id

        image = np.copy(image)
        _annotate_checkerboard_requirements(image)
        readings = self.process_image(image)

        if readings is None:
            if visualize:
                cv2.imshow("Checkerboard: {0}".format(cam_id), image)
                cv2.waitKey(20)
            return image

        corners, _ = readings
        if "corners" in visual_type:
            cv2.drawChessboardCorners(image, CHECKERBOARD_INNER_CORNERS, corners, True)

        if "axes" in visual_type:
            if isinstance(self._intrinsics_dict, RealsenseIntrinsics):
                cameraMatrix = self._intrinsics_dict.K_color
                distCoeffs = self._intrinsics_dict.distortion_color
            elif isinstance(self._intrinsics_dict, ZedIntrinsics):
                cameraMatrix = self._intrinsics_dict.K_left
                distCoeffs = self._intrinsics_dict.distortion_left
            else:
                raise NotImplementedError
            solved, rvec, tvec = cv2.solvePnP(CHECKERBOARD_OBJECT_POINTS, corners, cameraMatrix, distCoeffs)
            if solved:
                cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

        # Visualize
        if visualize:
            cv2.imshow("Checkerboard: {0}".format(cam_id), image)
            cv2.waitKey(20)

        return image


class HandCameraCalibrator(CheckerboardDetector):
    def __init__(self, camera, lin_error_threshold=1e-3, rot_error_threshold=1e-2, train_percentage=0.7, **kwargs):
        self.lin_error_threshold = lin_error_threshold
        self.rot_error_threshold = rot_error_threshold
        self.train_percentage = train_percentage
        super().__init__(camera, **kwargs)

    def calibrate(self, cam_id):
        return self._calibrate_cam_to_gripper(cam_id=cam_id)

    def _calibrate_cam_to_gripper(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calibrate_base_to_target(self, cam_id=None, readings=None, gripper_poses=None, target2cam_results=None):
        # Get Calibration Data #
        if cam_id is not None:
            readings, gripper_poses = self._readings_dict[cam_id], self._pose_dict[cam_id]
            self._curr_cam_id = cam_id

        # Get Target2Cam Transformation #
        if target2cam_results is None:
            target2cam_results = self.calculate_target_to_cam(readings)
        if target2cam_results is None:
            return None

        R_target2cam, t_target2cam, successes = target2cam_results
        gripper_poses = np.array(gripper_poses)[successes]

        # Calculate Appropriate Transformations #
        t_gripper2base = [np.array(pose[:3]) for pose in gripper_poses]
        R_gripper2base = [R.from_euler("xyz", pose[3:6]).as_matrix() for pose in gripper_poses]

        # Perform Calibration #
        rmat, pos = cv2.calibrateHandEye(
            R_gripper2base=R_target2cam,
            t_gripper2base=t_target2cam,
            R_target2cam=R_gripper2base,
            t_target2cam=t_gripper2base,
            method=4,
        )

        # Return Pose #
        pos = pos.flatten()
        angle = R.from_matrix(rmat).as_euler("xyz")
        pose = np.concatenate([pos, angle])

        return pose

    def _calculate_gripper_to_base(self, train_readings, train_gripper_poses, eval_readings=None):
        if eval_readings is None:
            eval_readings = train_readings

        # Get Eval Target2Cam Transformations #
        eval_results = self.calculate_target_to_cam(eval_readings, train=False)
        if eval_results is None:
            return None
        eval_R_target2cam, eval_t_target2cam, eval_successes = eval_results
        rmats, tvecs = [], []

        # Get Train Target2Cam Transformations #
        train_results = self.calculate_target_to_cam(train_readings)
        if train_results is None:
            return None

        # Use Training Data For Calibrations #
        base2target = self._calibrate_base_to_target(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_base2target = R.from_euler("xyz", base2target[3:]).as_matrix()
        t_base2target = np.array(base2target[:3])

        cam2gripper = self._calibrate_cam_to_gripper(
            gripper_poses=train_gripper_poses, target2cam_results=train_results
        )
        R_cam2gripper = R.from_euler("xyz", cam2gripper[3:]).as_matrix()
        t_cam2gripper = np.array(cam2gripper[:3])

        # Calculate Gripper2Base #
        for i in range(len(eval_R_target2cam)):
            R_base2cam = eval_R_target2cam[i] @ R_base2target
            t_base2cam = eval_R_target2cam[i] @ t_base2target + eval_t_target2cam[i]

            R_base2gripper = R_cam2gripper @ R_base2cam
            t_base2gripper = R_cam2gripper @ t_base2cam + t_cam2gripper

            R_gripper2base = R.from_matrix(R_base2gripper).inv().as_matrix()
            t_gripper2base = -R_gripper2base @ t_base2gripper

            rmats.append(R_gripper2base)
            tvecs.append(t_gripper2base)

        # Return Poses #
        eulers = np.array([R.from_matrix(rmat).as_euler("xyz") for rmat in rmats])
        eval_poses = np.concatenate([np.array(tvecs), eulers], axis=1)

        return eval_poses, eval_successes

    def is_calibration_accurate(self, cam_id):
        # Set Camera #
        self._curr_cam_id = cam_id

        # Split Into Train / Test #
        readings = self._readings_dict[cam_id]
        min_train = int(np.ceil(self.num_img_threshold / self.train_percentage))
        min_test = int(np.ceil(5 / (1 - self.train_percentage)))
        min_samples = max(min_train, min_test)
        if len(readings) < min_samples:
            _log.warning(
                "Calibration validation needs at least %d valid checkerboard samples; collected %d",
                min_samples,
                len(readings),
            )
            return False
        poses = np.array(self._pose_dict[cam_id])
        ind = np.random.choice(len(readings), size=len(readings), replace=False)
        num_train = int(len(readings) * self.train_percentage)

        train_ind, test_ind = ind[:num_train], ind[num_train:]
        train_poses, test_poses = poses[train_ind], poses[test_ind]
        train_readings = [readings[i] for i in train_ind]
        test_readings = [readings[i] for i in test_ind]

        # Calculate Approximate Gripper2Base Transformations #
        results = self._calculate_gripper_to_base(train_readings, train_poses, eval_readings=test_readings)
        if results is None:
            _log.warning(
                "Calibration validation could not fit the checkerboard observations after reprojection-error filtering"
            )
            return False
        approx_poses, successes = results
        test_poses = np.array(test_poses)[successes]

        # Calculate Per Dimension Error #
        pose_error = np.array([pose_diff(pose, approx_pose) for pose, approx_pose in zip(test_poses, approx_poses)])
        lin_error = np.linalg.norm(pose_error[:, :3], axis=0) ** 2 / pose_error.shape[0]
        rot_error = np.linalg.norm(pose_error[:, 3:6], axis=0) ** 2 / pose_error.shape[0]

        # Check Calibration Error #
        lin_success = np.all(lin_error < self.lin_error_threshold)
        rot_success = np.all(rot_error < self.rot_error_threshold)

        _log.info(
            "Calibration validation: valid_samples=%d, linear_mse=%s (limit=%g), rotation_mse=%s (limit=%g)",
            len(readings),
            np.array2string(lin_error, precision=6),
            self.lin_error_threshold,
            np.array2string(rot_error, precision=6),
            self.rot_error_threshold,
        )

        return lin_success and rot_success


def _calibrate_wrist_camera(cam, client) -> None:
    """Run the interactive calibration after camera and robot clients are connected."""
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    from tiptop.motion_planning import get_motion_gen
    from tiptop.workspace import workspace_cuboids

    cam_id = cam.serial
    intrinsics_dict = cam.get_intrinsics()
    calibrator = HandCameraCalibrator(intrinsics_dict)

    # Setup motion planner. Hard-code the time_dilation_factor as the movements are small
    _log.info("Setting up motion planner...")
    world_cfg = WorldConfig(cuboid=list(workspace_cuboids()))
    motion_gen = get_motion_gen(world_cfg, collision_activation_distance=0.01, warmup_iters=4)
    plan_config = MotionGenPlanConfig(time_dilation_factor=0.4)

    # Visualize the camera feed
    while True:
        frame = cam.read_camera()
        viz_img = frame.bgr
        viz_img = calibrator.augment_image(cam_id=cam_id, image=viz_img)
        viz_img = cv2.putText(
            viz_img,
            "Move robot s.t. calibration board is visible.",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        viz_img = cv2.putText(
            viz_img, "Press 'y' to continue, 'n' to exit", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Calibration View", viz_img)
        key = cv2.waitKey(1)
        if key == ord("y"):
            break
        elif key == ord("n"):
            return

    def get_q_curr() -> Float[torch.Tensor, "d"]:
        _q_curr = client.get_joint_positions()
        _q_curr_pt = torch.tensor(_q_curr, dtype=torch.float32, device="cuda")
        return _q_curr_pt

    def get_mat4x4() -> Float[np.ndarray, "4 4"]:
        _mat4x4 = motion_gen.kinematics.get_state(get_q_curr()).ee_pose.get_numpy_matrix()[0]
        return _mat4x4

    # Bad hack for now, flush out the communication channel
    _log.debug("Attempting to flush out the buffer")
    for _ in range(100):
        get_q_curr()
    _log.debug("Flushed out the buffer (I hope)")

    pose_origin_mat4x4 = get_mat4x4()
    pose_origin = np.zeros(6)
    pose_origin[:3] = pose_origin_mat4x4[:3, 3]
    pose_origin[3:] = rmat_to_euler(pose_origin_mat4x4[:3, :3])
    i = 0

    step_size = 0.15
    while True:
        calib_pose = calibration_traj(i * step_size, hand_camera=True)
        desired_pose = change_pose_frame(calib_pose, pose_origin)
        desired_pose_mat4x4 = np.eye(4)
        desired_pose_mat4x4[:3, 3] = desired_pose[:3]
        desired_pose_mat4x4[:3, :3] = euler_to_rmat(desired_pose[3:])

        desired_pose_pt = torch.tensor(desired_pose_mat4x4, dtype=torch.float32, device="cuda")
        desired_pose_curobo = Pose.from_matrix(desired_pose_pt)

        if i == 0:
            # calibration_traj(0) is exactly the taught pose, so capture it
            # directly instead of letting pose IK choose a different joint branch.
            _log.info("Capturing the initial calibration sample without moving the robot")
        else:
            q_curr = get_q_curr()
            js_curr = JointState.from_position(q_curr[None])
            result = motion_gen.plan_single(js_curr, desired_pose_curobo, plan_config)
            if not bool(result.success):
                raise RuntimeError(
                    "Could not plan the calibration trajectory. "
                    f"waypoint={i}, status={result.status}, q_current={q_curr.cpu().tolist()}, "
                    f"target_pose={desired_pose.tolist()}"
                )

            plan = result.interpolated_plan
            dt = result.interpolation_dt
            timings = [dt] * plan.position.shape[0]
            full_trajectory = plan.position.cpu().numpy()
            velocities = plan.velocity.cpu().numpy()
            result = client.execute_joint_impedance_path(
                joint_confs=full_trajectory, joint_vels=velocities, durations=timings
            )
            if not result["success"]:
                raise RuntimeError(f"Could not move robot at waypoint={i}! Error: {result['error']}")

        # env.update_robot(action, action_space="cartesian_position", blocking=False)
        time.sleep(0.4)  # wait for robot to stabilize
        pose_origin_mat4x4 = get_mat4x4()
        # pose_origin_mat4x4 = np.array(state["ee_pose"])
        pose = np.zeros(6)
        pose[:3] = pose_origin_mat4x4[:3, 3]
        pose[3:] = rmat_to_euler(pose_origin_mat4x4[:3, :3])

        # Add Sample + Augment Images #
        frame = cam.read_camera()
        image = frame.bgr

        cycle_complete = (i * step_size) >= (2 * np.pi)
        cycle_prop_complete = 100 * (i * step_size) / (2 * np.pi)
        _log.debug(f"{cycle_prop_complete:.2f}% calibration complete")

        calibrator.add_sample(cam_id=cam_id, image=frame.bgr, pose=pose)
        augmented_image = calibrator.augment_image(cam_id=cam_id, image=image)
        augmented_image = cv2.putText(
            augmented_image,
            f"Calibration {cycle_prop_complete:.2f}% complete...",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration View", augmented_image)
        cv2.waitKey(1)

        # Check if cycle is complete
        if cycle_complete:
            break
        i += 1

    success = calibrator.is_calibration_accurate(cam_id)
    if not success:
        raise RuntimeError(
            "Calibration failed validation; no calibration result was saved. "
            f"valid_checkerboard_samples={len(calibrator._readings_dict[cam_id])}"
        )

    # Save the calibration
    transformation = calibrator.calibrate(cam_id)
    calibration_metadata = {}
    if str(tiptop_cfg().robot.type) == "cobot_magic":
        # MotionGen's ee_link is tool_center_point for Cobot.  Persist both
        # frame names so future readers never infer the old gripper-base frame.
        from tiptop.cobot_magic.frames import cobot_magic_calibration_metadata

        calibration_metadata = cobot_magic_calibration_metadata()
    update_calibration_info(cam_id, transformation, **calibration_metadata)
    _log.info(f"Updated calibration info for {cam_id}. Transformation: {transformation}")


def calibrate_wrist_camera() -> None:
    """Calibrate the wrist camera."""
    setup_logging()
    cam = get_hand_camera()
    client = get_robot_client()
    _calibrate_wrist_camera(cam, client)


def calibrate_wrist_camera_entrypoint():
    calibrate_wrist_camera()


if __name__ == "__main__":
    calibrate_wrist_camera_entrypoint()
