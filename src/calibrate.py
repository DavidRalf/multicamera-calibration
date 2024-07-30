from pathlib import Path
import argparse
import cv2
import numpy as np
import yaml
import sys
import glob
import utils as utils

legacy = cv2.__version__ == "4.2.0"
debug = False
SHOW_EPI = False
CALC_DISPARITY = False
REFINE_CALIBRATION = False
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


def detect_charuco_marker(band):
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

    #if len(sys.argv) <= 1:
    #    print(f"usage: {sys.argv[0]} IMAGE_PREFIX_FOR_GLOB")
    #    exit(0)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    allCorners = []
    allIds = []
    objectPoints = []
    #filepattern = sys.argv[1] + "*.png"
    print(f"{band = }")
    #gg = glob.glob(filepattern)
    # print(gg)

    for f in sorted(band):
        print("filename:", f)
        ret, frame = (True, cv2.imread(f))
        if np.any(frame is None):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debugFrame = frame.copy()
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


# print(allCorners)
def normalCalib(objectPoints, allCorners, example_frame, calib_flags=0):
    print("\n---------- Normal Calib ----------")
    inputsize = example_frame.shape[:2]
    inputsize = (inputsize[1], inputsize[0])

    nretval, ncameraMatrix, ndistCoeffs, nrvecs, ntvecs = \
        cv2.calibrateCamera(objectPoints, allCorners, inputsize, None, None, flags=calib_flags)
    print(f"normal calib RMS: {nretval}")

    if REFINE_CALIBRATION:
        # calculate reprojection errors https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        error_list = []
        for i in range(len(objectPoints)):
            imgpoints2, _ = cv2.projectPoints(objectPoints[i], nrvecs[i], ntvecs[i], ncameraMatrix, ndistCoeffs)
            error = cv2.norm(allCorners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            error_list.append(error)
        # compute variance of errors
        error_std = np.sqrt(np.mean(np.square(error_list)))

        # remove outlier from data
        objectPoints = [op for op, err in zip(objectPoints, error_list) if err < error_std]
        allCorners = [op for op, err in zip(allCorners, error_list) if err < error_std]

        # repeat calibration without outlier data
        nretval, ncameraMatrix, ndistCoeffs, nrvecs, ntvecs = \
            cv2.calibrateCamera(objectPoints, allCorners, inputsize, None, None, flags=calib_flags)
        print(f"normal calib (without outlier) RMS: {nretval}")

    print("cameraMatrix", ncameraMatrix, "distCoeffs", ndistCoeffs, sep="\n")
    # print(newCam)
    rect = cv2.undistort(
        example_frame, ncameraMatrix, ndistCoeffs, newCameraMatrix=ncameraMatrix
    )
    cv2.namedWindow("frameRect", cv2.WINDOW_NORMAL)
    cv2.imshow("frameRect", rect)
    return ncameraMatrix, ndistCoeffs


def calc_stereo(img_L, img_R, cal_L, cal_R, calib_flags=0):
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

    # set resolution of board image
    pixelPerBlock = 120

    np.set_printoptions(suppress=True, linewidth=1000)

    dictionary = cv2.aruco.getPredefinedDictionary(arucoDict)

    board = cv2.aruco.CharucoBoard([pw, ph], checkerSize, markerSize, dictionary)
    board.setLegacyPattern(True)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board)
    if debug:
        print(board.chessboardCorners)

    ind = np.indices(((pw - 1), (ph - 1)))
    ind[1, :] = np.fliplr(ind[1, :])

    objp = np.zeros((1, (pw - 1) * (ph - 1), 3), np.float32)
    objp[0, :, :2] = ind.T.reshape(-1, 2)


    K_L, D_L, R_L, P_L = cal_L
    K_R, D_R, R_R, P_R = cal_R
    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
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

    # show epilines
    if SHOW_EPI:
        lines1 = cv2.computeCorrespondEpilines(charuco_corners_R, 2, F)
        lines1 = lines1.reshape(-1, 3)
        img_L_line, img_R_line = utils.draw_lines(
            img_L, img_R, lines1, charuco_corners_L, charuco_corners_R
        )

        lines2 = cv2.computeCorrespondEpilines(charuco_corners_L, 1, F)
        lines2 = lines2.reshape(-1, 3)
        img_R_line2, img_L_line2 = utils.draw_lines(
            img_R, img_L, lines2, charuco_corners_R, charuco_corners_L
        )

        cv2.namedWindow("epi", cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("epi", 0, 0)
        cv2.imshow("epi", img_L_line)
        cv2.namedWindow("epi2", cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("epi2", 680, 0)
        cv2.imshow("epi2", img_R_line)
        cv2.namedWindow("epi3", cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("epi3", 0, 480)
        cv2.imshow("epi3", img_L_line2)
        cv2.namedWindow("epi4", cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("epi4", 680, 480)
        cv2.imshow("epi4", img_R_line2)

    return R, T



if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description='Calibrate intrinsics and extrinsics between Basler and Micasense bands.')

    # Add positional arguments
    parser.add_argument('micasense_path', type=str,
                        help='Path to the directory containing the micasense calibration images')
    parser.add_argument('basler_path', type=str, help='Path to the directory containing basler calibration images')
    parser.add_argument('image_number', type=str,
                        help='Image number for extrinsics calibration based on the basler numbers with leading zeros (e.g., 0059)')
    parser.add_argument('output_path', type=str, nargs='?', default=".", help='Path to save the calibration YAML file')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    micasense_path = Path(args.micasense_path)
    basler_path = Path(args.basler_path)
    output_path = Path(args.output_path)
    image_number = args.image_number

    # Collect image names that match the pattern
    bands = []
    for i in range(1, 7):  # Assuming there are 6 bands
        band = list(micasense_path.glob(f'IMG_*_{i}.tif'))
        band = [x.as_posix() for x in band]
        bands.append(band)

    print("Micasense Path:", micasense_path)
    print("Basler Path:", basler_path)

    data = {}
    for i, band in enumerate(bands):
        print(f"Processing band {i + 1}: {band}")
        _objectPoints, _allCorners, _example_frame = detect_charuco_marker(band)
        K, D = normalCalib(_objectPoints, _allCorners, _example_frame, calib_flags=0)

        cal_samson_1 = utils.read_basler_calib("/media/david/T71/multispektral/20240416_calib/SAMSON1/SAMSON1.yaml")
        cal_R = K, D,None,None
        K_L, D_L, _, _ = cal_samson_1
        K_R, D_R = K, D

        # Load the Basler image using the original image_number
        img_L = cv2.imread(basler_path.as_posix() + f"/{image_number}.png")

        # Create the Micasense file name by slicing off the first character
        micasense_image_number = image_number[2:]  # This will result in '0002'
        img_R = cv2.imread(micasense_path.as_posix() + f"/IMG_{int(micasense_image_number)-1}_{i + 1}.tif")
        calib_flags = cv2.CALIB_FIX_INTRINSIC
        R, T = calc_stereo(img_L, img_R, cal_samson_1, cal_R, calib_flags=calib_flags)
        print(f"R {R}")

        # Add or update the band data
        band_name = f"band_{i + 1}"
        data[band_name] = {
            'cameraMatrix': K.tolist(),
            'distCoeffs': D.tolist(),
            'rotation': R.tolist(),
            'translation': T.tolist(),
        }

    with open(output_path / "micasense_calib.yaml", "w") as file:
        yaml.safe_dump(data, file, default_flow_style=None)

    print(f"Calibration data saved to {output_path / 'micasense_calib.yaml'}")


