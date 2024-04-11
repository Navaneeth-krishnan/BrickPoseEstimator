from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import torch
import numpy as np
import cv2
import json


def readIntrinsic(json_file_path):

    # Read and extract information from the JSON file
    with open(json_file_path, "r") as file:
        cam_data = json.load(file)

    width = cam_data["width"]
    height = cam_data["height"]
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["px"]
    cy = cam_data["py"]
    dist_coeffs = cam_data["dist_coeffs"]
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return intrinsic_matrix


def find_closest_mask_centroid(masks, optical_center):
    """
    Find the mask whose centroid is closest to the optical center of the image.

    Parameters:
    - masks: List of binary mask images (numpy arrays).
    - optical_center: Tuple (x, y) representing the optical center of the image.

    Returns:
    - index of the closest mask in the list,
    - coordinates of the closest centroid.
    """
    min_distance = float("inf")
    closest_centroid = None
    closest_index = None

    for i, mask in enumerate(masks):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Assume largest contour is the object mask
            c = max(contours, key=cv2.contourArea)
            # Compute the centroid of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                # Compute distance to the optical center
                distance = np.linalg.norm(np.array(centroid) - np.array(optical_center))
                # Update if this is the closest centroid so far
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid
                    closest_index = i

    return masks[closest_index]


def getSAMMask(image):
    """
    Uses Segment Anything Model to extract object masks from an image and returns the
    mask closest to the image centre

    Parameters:
    - image: 8-bit RGB image.

    Returns:
    - A binary openCV mask image
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_h"

    CHECKPOINT_PATH = os.path.join("./", "sam_vit_h_4b8939.pth")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(multimask_output=True)
    optical_center = [image.shape[0] / 2, image.shape[1] / 2]

    mask = find_closest_mask_centroid(masks, optical_center)

    return mask


def convert_2d_to_3d(contour_points, depth_image, intrinsic_matrix, depth_scale):
    """
    Convert a set of 2D image points to 3D points

    Parameters:
    - contour_points: List of 2D contour points(OpenCV Contour).
    - depth_image: 16-bit depth image OpenCV
    - intrinsic: 3x3 K matrix of the camera
    - depth_scale : Factor by which the depth values in the depth image are scaled

    Returns:
    - A numpy array of x,y,z for each point
    """
    # Initialize an empty array for 3D points
    points_3d = []

    # Iterate over contour points
    for point in contour_points:
        # Get x, y coordinates of the contour point
        x, y = point[0]

        # Get the depth value at the contour point from the depth image
        depth_value = depth_image[y, x]

        # Convert depth value to 3D depth (in meters)
        depth_3d = depth_value * depth_scale / 1000.0  # Convert to meters

        # Convert 2D point to normalized image coordinates
        point_2d_normalized = np.array([[x], [y], [1]])

        # Apply inverse of intrinsic matrix to get 3D point in camera coordinate system
        point_3d_camera = np.linalg.inv(intrinsic_matrix) @ point_2d_normalized
        point_3d_camera *= depth_3d / point_3d_camera[2]  # Scale by depth

        # Append the 3D point to the list
        points_3d.append(point_3d_camera.flatten())

    return np.array(points_3d)


def getBrickPose(image, depth, intrinsic):
    """
    Get the Pose of the brick in the image that is closest to the image center

    Parameters:
    - image: 8-bit RGB image.
    - depth: 16-bit single channel image that stores depth value.
    - intrinsic: 3x3 K matrix of the camera

    Returns:
    - A 4x4 transformation matrix with the rotation and translation of the brick with respect to the camera coordinates

    """
    mask = getSAMMask(image)

    # Apply the mask to the depth image
    masked_depth = cv2.bitwise_and(depth, depth, mask=mask)

    # Normalize the masked depth image
    depth_image_normalized = cv2.normalize(
        masked_depth, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    # Flatten the masked depth image to a 1D array
    depth_values = depth_image_normalized.flatten()

    # Remove zero depth values (assuming zero represents invalid depth)
    depth_values = depth_values[depth_values > 0]

    # Plot histogram of depth values
    hist, bins = np.histogram(depth_values, bins=50)

    # Find the bin with the maximum frequency
    max_freq_index = np.argmax(hist)
    max_freq_value = bins[max_freq_index]

    # Find the range of the bin with maximum frequency
    max_freq_bin_range = (bins[max_freq_index], bins[max_freq_index + 1])

    # Create a new mask for depth values within the range of the bin with maximum frequency
    mask = np.logical_and(
        depth_image_normalized >= max_freq_bin_range[0],
        depth_image_normalized < max_freq_bin_range[1],
    )

    # Reshape the new mask to match the original depth image shape
    mask = mask.astype(np.uint8)

    # Define a kernel for dilation
    kernel = np.ones((1, 1), np.uint8)

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=5)

    # Find a high poly contour of the mask
    contours, hierarchy = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert the contour into a convex hull
    hull = cv2.convexHull(contours[0])

    # Approximate a Low poly contour from the hull
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Estimate the 3D location of the points in camera frame
    depth_scale = 0.1
    points_3d = convert_2d_to_3d(approx, masked_depth, intrinsic, depth_scale)

    # Solve PnP to get the pose of the brick with respect to the camera
    # Convert contour points to numpy array
    image_points = np.float32(approx.reshape(-1, 2))

    # Brick Dimensions
    brick_points = np.array(
        [[0, 0, 0], [0, 0.05, 0], [-0.21, 0.05, 0], [-0.21, 0, 0]], dtype=np.float32
    )
    # Estimate camera pose using solvePnP
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        brick_points, image_points, intrinsic, None
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # tvec meters --> millimeters
    tvec = tvec * 1000

    # Convert rotation matrix into a transformation matrix
    transformation_matrix = np.vstack(
        (np.hstack((rotation_matrix, tvec)), np.array([0, 0, 0, 1]))
    )

    return transformation_matrix
