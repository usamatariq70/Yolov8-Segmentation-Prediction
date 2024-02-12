from ultralytics import YOLO
import cv2
import time
import argparse
import numpy as np
import os


def get_prediction(model_path, img_path, output_dir):
    model = YOLO(model_path)

    img = cv2.imread(img_path)

    results = model.predict(img, conf=0.5)

    for r in results:
        count = len(r.boxes)
        seg_img = postprocess_prediction(img, r)
        cv2.imwrite(f"{output_dir}", seg_img)

    return count


def postprocess_prediction(img, prediction):
    mask = np.zeros_like(img)
    all_mask_points = []

    for i in range(len(prediction.boxes.cls)):
        mask_points = prediction.masks[i].xy[0].astype(np.int32)

        all_mask_points.append(
            {"conf": float(prediction.boxes[i].conf), "points": mask_points}
        )
        cv2.fillPoly(mask, [mask_points], (0, 255, 0))

    alpha = 0.5  # Adjust the alpha value for transparency
    result = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for mask_point in all_mask_points:
        text = "{:.2f}".format(mask_point["conf"])
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (
            (mask_point["points"][0][0] + mask_point["points"][2][0] - text_size[0])
            // 2,
            (mask_point["points"][0][1] + mask_point["points"][2][1] + text_size[1])
            // 2,
        )
        cv2.putText(
            result, text, text_position, font, font_scale, (0, 0, 0), font_thickness
        )

    return result


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Get prediction on image")
    parser.add_argument("--img", type=str, help="Provide the path of image")
    parser.add_argument("--model", type=str, help="Provide the path of model")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Provide the path of directory where you want to save output",
    )

    args = parser.parse_args()

    directory, img_name = os.path.split(args.img)
    output_dir = f"{args.output_dir}/{img_name}"

    count = get_prediction(args.model, args.img, output_dir)

    end_time = time.time()

    print("Count: ", count)
    print("Time taken: ", end_time - start_time)
