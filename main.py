from flask import Flask, request, jsonify
import numpy as np
import cv2
import logging
from werkzeug.exceptions import BadRequest

from libraries import readIntrinsic, getBrickPose

# Initialize Flask app
app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)


@app.route("/get_pose", methods=["POST"])
def get_pose():
    try:
        # Validate and parse the incoming request
        if "image" not in request.files or "depth" not in request.files:
            raise BadRequest("Missing image or depth data")

        image_file = request.files["image"]
        depth_file = request.files["depth"]

        # Convert the image and depth data from the request files
        image = cv2.imdecode(
            np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR
        )
        depth = cv2.imdecode(
            np.frombuffer(depth_file.read(), np.uint16), cv2.IMREAD_UNCHANGED
        )

        # Dummy intrinsic matrix, replace it with the actual one
        intrinsic = readIntrinsic("cam.json")

        # Get the brick pose
        transformation_matrix = getBrickPose(image, depth, intrinsic)

        # Return the result
        return jsonify({"transformation_matrix": transformation_matrix.tolist()})

    except BadRequest as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
