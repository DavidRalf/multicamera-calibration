import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

import micasense.image as image
import src.utils as utils


def detect_charuco_marker(image_paths):
    legacy = cv2.__version__ == "4.2.0"
    debug = False

    # pattern size
    # checkerSize, markerSize,pw,ph,arucoDict = 0.05825,0.05825/2, 16, 9,cv2.aruco.DICT_4X4_100 # nils
    checkerSize, markerSize, pw, ph, arucoDict = (
        0.06,
        0.045,
        12,
        9,
        cv2.aruco.DICT_5X5_100,
    )  # coarse
    # checkerSize, markerSize, pw,ph,arucoDict = 0.03,0.022, 24, 17,cv2.aruco.DICT_5X5_1000 # fine
    np.set_printoptions(suppress=True, linewidth=1000)

    dictionary = cv2.aruco.getPredefinedDictionary(arucoDict)
    if legacy:
        board = cv2.aruco.CharucoBoard_create(pw, ph, checkerSize, markerSize, dictionary)
    else:
        board = cv2.aruco.CharucoBoard([pw, ph], checkerSize, markerSize, dictionary)
        board.setLegacyPattern(True)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(board)
    if debug:
        print(board.chessboardCorners)
        # write calibration board image
        # set resolution of board image
        pixelPerBlock = 120
        img = board.generateImage((pixelPerBlock * pw, pixelPerBlock * ph))
        cv2.imwrite("charuco.bmp", img)

    ind = np.indices(((pw - 1), (ph - 1)))
    ind[1, :] = np.fliplr(ind[1, :])

    objp = np.zeros((1, (pw - 1) * (ph - 1), 3), np.float32)
    objp[0, :, :2] = ind.T.reshape(-1, 2)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    allCorners = []
    allIds = []
    objectPoints = []

    for f in sorted(image_paths):
        print("filename:", f)
        if f.endswith(".tif"):
            frame = image.Image(f).raw()
            frame = (frame / 256).astype(np.uint8)
            gray = frame
        else:
            ret, frame = (True, cv2.imread(f))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.any(frame is None):
            continue

        debugFrame = np.copy(frame)
        if legacy:
            charuco_corners, charuco_ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, dictionary
            )
            if len(charuco_corners) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    charuco_corners, charuco_ids, gray, board, minMarkers=2
                )
            else:
                continue
            # print(charuco_corners)
        else:
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
                gray
            )

        if charuco_ids is None:
            print("  no charuco ids")
        elif len(charuco_ids) < 10:
            print("  too few markers detected", len(charuco_ids))
        else:
            cv2.aruco.drawDetectedCornersCharuco(
                debugFrame, charuco_corners, charuco_ids, (255, 0, 0)
            )
            print(
                f"  charuco ids {np.min(charuco_ids)}..{np.max(charuco_ids)} ({len(charuco_ids)})"
            )

            allCorners.append(charuco_corners)
            allIds.append(charuco_ids)
            objpp = np.array([objp[0, a[0]] for a in charuco_ids])
            objectPoints.append(np.array([objpp]))
            if debug:
                print(charuco_corners)
                print(charuco_ids)
            for p, id, obp in zip(charuco_corners, charuco_ids, objpp):
                pass
                # cv2.drawMarker(debugFrame, tuple(p[0].astype(np.int32)), (255, 0, 0), 1, 30, 1)
                # cv2.putText(debugFrame, f"{id[0]}{obp}", tuple(p[0]), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255))

        cv2.imshow("frame", debugFrame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print()

    return objectPoints, allCorners, frame


def normalCalib(objectPoints, allCorners, example_frame, calib_flags=0):
    print("\n---------- Normal Calib ----------")
    inputsize = example_frame.shape[:2]
    inputsize = (inputsize[1], inputsize[0])

    nretval, ncameraMatrix, ndistCoeffs, nrvecs, ntvecs = \
        cv2.calibrateCamera(objectPoints, allCorners, inputsize, None, None, flags=calib_flags)
    print(f"normal calib RMS: {nretval}")
    print("cameraMatrix", ncameraMatrix, "distCoeffs", ndistCoeffs, sep="\n")
    # print(newCam)
    rect = cv2.undistort(
        example_frame, ncameraMatrix, ndistCoeffs, newCameraMatrix=ncameraMatrix
    )
    cv2.namedWindow("frameRect", cv2.WINDOW_NORMAL)
    cv2.imshow("frameRect", rect)
    return ncameraMatrix, ndistCoeffs


def calibrate_extrinsic(img_L, img_R, intrinsic_L, intrinsic_R, calib_flags):
    checkerSize, markerSize, pw, ph, arucoDict = (
        0.06,
        0.045,
        12,
        9,
        cv2.aruco.DICT_5X5_100,
    )  # coarse
    # checkerSize, markerSize, pw,ph,arucoDict = 0.03,0.022, 24, 17,cv2.aruco.DICT_5X5_1000 # fine

    # set resolution of board image
    pixelPerBlock = 120

    np.set_printoptions(suppress=True, linewidth=1000)

    dictionary = cv2.aruco.getPredefinedDictionary(arucoDict)

    board = cv2.aruco.CharucoBoard([pw, ph], checkerSize, markerSize, dictionary)
    board.setLegacyPattern(True)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board)

    ind = np.indices(((pw - 1), (ph - 1)))
    ind[1, :] = np.fliplr(ind[1, :])

    objp = np.zeros((1, (pw - 1) * (ph - 1), 3), np.float32)
    objp[0, :, :2] = ind.T.reshape(-1, 2)

    K_L, D_L = intrinsic_L
    K_R, D_R = intrinsic_R
    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    if img_R.dtype == np.uint16:
        img_R = (img_R / 256).astype(np.uint8)
        gray_R = img_R
    else:
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    # Detect aruco markers
    dict_L = {}
    dict_R = {}
    (
        charuco_corners_L,
        charuco_ids_L,
        marker_corners_L,
        marker_ids_L,
    ) = detector.detectBoard(gray_L)
    (
        charuco_corners_R,
        charuco_ids_R,
        marker_corners_R,
        marker_ids_R,
    ) = detector.detectBoard(gray_R)
    charuco_ids_L = charuco_ids_L.flatten()
    charuco_ids_R = charuco_ids_R.flatten()
    for id, corner in zip(charuco_ids_L, charuco_corners_L):
        dict_L[id] = corner
    for id, corner in zip(charuco_ids_R, charuco_corners_R):
        dict_R[id] = corner

    inter = np.intersect1d(charuco_ids_L, charuco_ids_R)
    # print("corners found in both:",inter)
    inter_corners_L = np.array([dict_L[id] for id in inter])
    inter_corners_R = np.array([dict_R[id] for id in inter])
    objectPoints = np.array([[objp[0, a] for a in inter]])

    # draw detected corners
    cv2.aruco.drawDetectedCornersCharuco(
        img_L, charuco_corners_L, charuco_ids_L, (255, 0, 0)
    )
    cv2.aruco.drawDetectedCornersCharuco(
        img_R, charuco_corners_R, charuco_ids_R, (255, 0, 0)
    )
    # draw corners found in both
    cv2.aruco.drawDetectedCornersCharuco(img_L, inter_corners_L, inter, (0, 255, 255))
    cv2.aruco.drawDetectedCornersCharuco(img_R, inter_corners_R, inter, (0, 255, 255))

    cv2.namedWindow("L_aruco", cv2.WINDOW_NORMAL)
    cv2.namedWindow("R_aruco", cv2.WINDOW_NORMAL)
    cv2.imshow("L_aruco", img_L)
    cv2.imshow("R_aruco", img_R)
    cv2.moveWindow("R_aruco", 680, 0)
    cv2.waitKey(1)
    image_size = (img_L.shape[1], img_L.shape[0])
    print("running stereo calibration")
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints * checkerSize,
        [inter_corners_L],
        [inter_corners_R],
        K_L,
        D_L,
        K_R,
        D_R,
        image_size,
        flags=calib_flags,
    )
    print(f"{ret = }\n{R = }\n{T = }\n{E = }\n{F = }")
    # print(f"{ret = }\n{M1 = }\n{M2 = }")
    print(f"{K_L = }\n{D_L = }\n{K_R = }\n{D_R = }")

    return R, T


def calibrate_intrinsic(image_paths):
    _objectPoints, _allCorners, _example_frame = detect_charuco_marker(image_paths)
    K, D = normalCalib(_objectPoints, _allCorners, _example_frame, calib_flags=0)
    return K, D


def calibrate_basler(basler1_path, basler2_path, image_number):
    K_1, D_1 = calibrate_intrinsic([str(img) for img in basler1_path.glob('*.png')])
    K_2, D_2 = calibrate_intrinsic([str(img) for img in basler2_path.glob('*.png')])

    img_L = cv2.imread(f"{basler1_path}/{image_number}.png")
    img_R = cv2.imread(f"{basler2_path}/{image_number}.png")

    R, T = calibrate_extrinsic(img_L, img_R, (K_1, D_1), (K_2, D_2), calib_flags=cv2.CALIB_FIX_INTRINSIC)
    image_size = (img_L.shape[1], img_L.shape[0])

    R_L, R_R, P_L, P_R, Q, roi_L, roi_R = cv2.stereoRectify(K_1, D_1, K_2, D_2, image_size, R, T,
                                                            flags=cv2.CALIB_ZERO_DISPARITY)

    utils.write_calib("../calib/SAMSON1_SAMSON2_stereo.yaml", K_1, D_1, R_L, P_L)
    utils.write_calib("../calib/SAMSON2_SAMSON1_stereo.yaml", K_2, D_2, R_R, P_R)


def calibrate_micasense(micasense_path, basler1_path, image_number):
    data = {}
    cal_samson_1 = utils.read_basler_calib("../calib/SAMSON1_SAMSON2_stereo.yaml")

    for i in range(1, 7):
        band_images = list(micasense_path.glob(f'IMG_*_{i}.tif'))
        band_images = [str(img) for img in band_images]
        micasense_image_number = utils.get_micasense_number_from_basler_number(image_number)

        K_M, D_M = calibrate_intrinsic(band_images)
        img_M = image.Image(f"{micasense_path}/IMG_{micasense_image_number}_{i}.tif").raw()
        img_B = cv2.imread(f"{basler1_path}/{image_number}.png")

        K_B, D_B, _, _ = cal_samson_1

        R_B1_M, T_B1_M = calibrate_extrinsic(img_B, img_M, (K_B, D_B), (K_M, D_M), calib_flags=cv2.CALIB_FIX_INTRINSIC)

        data[f"band_{i}"] = {
            'cameraMatrix': K_M.tolist(),
            'distCoeffs': D_M.tolist(),
            'rotation': R_B1_M.tolist(),
            'translation': T_B1_M.tolist(),
        }

    with open("../calib/micasense_calib.yaml", "w") as file:
        yaml.safe_dump(data, file, default_flow_style=None)


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate intrinsics and extrinsics.')
    parser.add_argument('micasense_path', type=str, help='Path to the Micasense calibration images')
    parser.add_argument('basler1_path', type=str, help='Path to the Basler (SAMSON1) calibration images')
    parser.add_argument('basler2_path', type=str, help='Path to the Basler (SAMSON2) calibration images')
    parser.add_argument('image_number', type=str,
                        help='Image number for extrinsics calibration bases on basler numbers (e.g., 000002)')
    parser.add_argument('calculate_basler_new', type=utils.str_to_bool, nargs='?', default='false',
                        help='Recalculate Basler calibration (true/false, default: false)')
    parser.add_argument('calculate_micasense_new', type=utils.str_to_bool, nargs='?', default='true',
                        help='Recalculate Micasense calibration (true/false, default: true)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    micasense_path = Path(args.micasense_path)
    basler1_path = Path(args.basler1_path)
    basler2_path = Path(args.basler2_path)

    utils.validate_directory(micasense_path, "Micasense")
    utils.validate_directory(basler1_path, "Basler1")
    utils.validate_directory(basler2_path, "Basler2")

    files_exist = utils.check_stereo_yaml_files("../calib")

    if args.calculate_basler_new or not files_exist:
        calibrate_basler(basler1_path, basler2_path, args.image_number)
    if args.calculate_micasense_new:
        calibrate_micasense(micasense_path, basler1_path, args.image_number)
