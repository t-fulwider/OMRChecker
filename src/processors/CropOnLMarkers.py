
import cv2
import numpy as np

from src.constants.image_processing import (
    APPROX_POLY_EPSILON_FACTOR,
    CANNY_PARAMS,
    DEFAULT_GAUSSIAN_BLUR_KERNEL,
    MIN_PAGE_AREA_THRESHOLD,
    PAGE_THRESHOLD_PARAMS,
)
from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class CropOnLMarkers(ImagePreprocessor):
    """
    Detects L-shaped corner markers and perspective-corrects an OMR sheet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        options = self.options

        # Normalized search regions:
        # [x_min, y_min, x_max, y_max]
        self.anchor_points = options.get(
            "anchorPoints",
            [
                [0.0, 0.0, 0.2, 0.2],  # Top-left
                [0.8, 0.0, 1.0, 0.2],  # Top-right
                [0.0, 0.8, 0.2, 1.0],  # Bottom-left
                [0.8, 0.8, 1.0, 1.0],  # Bottom-right
            ],
        )

        self.morph_kernel = tuple(
            int(value) for value in options.get("morphKernel", [3, 3])
        )

        self.min_contour_area = float(options.get("minContourArea", 30))
        self.max_contour_area_ratio = float(
            options.get("maxContourAreaRatio", 0.25)
        )

        self.canny_threshold1 = int(options.get("cannyThreshold1", 50))
        self.canny_threshold2 = int(options.get("cannyThreshold2", 150))

    def __str__(self):
        return "CropOnLMarkers"

    @staticmethod
    def exclude_files():
        return []

    def get_anchor_regions(self, image):
        # Convert normalized anchor regions into pixel coordinates.
        height, width = image.shape[:2]
        regions = []

        for region in self.anchor_points:
            if len(region) != 4:
                raise ValueError(
                    "Each anchor point must contain four coordinates."
                )

            x_min, y_min, x_max, y_max = region

            if not (
                0 <= x_min < x_max <= 1
                and 0 <= y_min < y_max <= 1
            ):
                raise ValueError(
                    "Anchor coordinates must be normalized between 0 and 1."
                )

            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            regions.append((x1, y1, x2, y2))

        return regions

    @staticmethod
    def to_grayscale(image):
        # Convert an image to grayscale
        if len(image.shape) == 2:
            return image

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def preprocess(self, image):

        # Reduce noise and extract edges.

        gray = self.to_grayscale(image)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.morph_kernel,
        )

        # Remove small isolated noise.
        opened = cv2.morphologyEx(
            normalized,
            cv2.MORPH_OPEN,
            kernel,
        )

        # Connect small gaps in printed marker edges.
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
        )

        edges = cv2.Canny(
            closed,
            self.canny_threshold1,
            self.canny_threshold2,
        )

        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def find_marker_contours(self, edges, region):
        # Find possible L-shaped marker contours inside one anchor region.
        x1, y1, x2, y2 = region
        region_edges = edges[y1:y2, x1:x2]

        contours = ImageUtils.grab_contours(
            cv2.findContours(
                region_edges.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )

        candidates = []

        region_area = region_edges.shape[0] * region_edges.shape[1]
        max_area = region_area * self.max_contour_area_ratio

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_contour_area or area > max_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)

            if bw < 5 or bh < 5:
                continue

            # An L marker should occupy only part of its bounding rectangle.
            extent = area / float(bw * bh)

            if extent > 0.85:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )

            # A filled L-shaped contour normally has several turning points.
            if len(approximation) < 5 or len(approximation) > 12:
                continue

            # Convert local contour coordinates into full-image coordinates.
            approximation = approximation.reshape(-1, 2)
            approximation[:, 0] += x1
            approximation[:, 1] += y1

            score = area * (1.0 - extent)

            candidates.append((score, approximation))

        candidates.sort(key=lambda item: item[0], reverse=True)

        return candidates

    def find_marker_points(self, image, edges):

        #Find marker contours in all configured anchor regions.

        #Returns all turning points from the selected marker contours.
        #The number of points is not restricted to four.
    
        marker_points = []

        regions = self.get_anchor_regions(image)

        for index, region in enumerate(regions):
            candidates = self.find_marker_contours(edges, region)

            if not candidates:
                logger.warning(
                    f"CropOnLMarkers: No marker found in anchor region {index + 1}."
                )
                continue

            # Use the strongest candidate in this region.
            _, points = candidates[0]
            marker_points.extend(points.tolist())

        if len(marker_points) < 4:
            return None

        return np.array(marker_points, dtype=np.float32)

    @staticmethod
    def get_outer_corners(points):

        #Calculate four outer page corners from a variable number of points.

        #The points are first combined into a convex hull. The extreme
        #coordinates are then used to estimate the four page corners.
        
        if points is None or len(points) < 4:
            return None

        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull_points = hull.reshape(-1, 2)

        if len(hull_points) < 4:
            return None

        sums = hull_points.sum(axis=1)
        differences = np.diff(hull_points, axis=1).reshape(-1)

        top_left = hull_points[np.argmin(sums)]
        bottom_right = hull_points[np.argmax(sums)]
        top_right = hull_points[np.argmin(differences)]
        bottom_left = hull_points[np.argmax(differences)]

        corners = np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype=np.float32,
        )

        # Make sure the four selected points are distinct.
        if len(np.unique(corners, axis=0)) != 4:
            return None

        return corners

    @classmethod
    def find_page_corners(cls, image):
        """Find a visible page outline when the corner markers are incomplete."""
        gray = cls.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, DEFAULT_GAUSSIAN_BLUR_KERNEL, 0)
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        _, thresholded = cv2.threshold(
            normalized,
            PAGE_THRESHOLD_PARAMS["threshold_value"],
            PAGE_THRESHOLD_PARAMS["max_pixel_value"],
            cv2.THRESH_TRUNC,
        )
        normalized = cv2.normalize(
            thresholded,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed = cv2.morphologyEx(normalized, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(
            closed,
            CANNY_PARAMS["lower_threshold"],
            CANNY_PARAMS["upper_threshold"],
        )

        contours = ImageUtils.grab_contours(
            cv2.findContours(
                edges,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )
        contours = sorted(
            (cv2.convexHull(contour) for contour in contours),
            key=cv2.contourArea,
            reverse=True,
        )[:5]

        for contour in contours:
            if cv2.contourArea(contour) < MIN_PAGE_AREA_THRESHOLD:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(
                contour,
                APPROX_POLY_EPSILON_FACTOR * perimeter,
                True,
            )

            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                return approximation.reshape(4, 2).astype(np.float32)

        return None

    def apply_filter(self, image, filename):
        # Detect L-markers and warp the OMR sheet.
        config = self.tuning_config

        edges = self.preprocess(image)

        marker_points = self.find_marker_points(image, edges)
        corners = self.get_outer_corners(marker_points)

        if corners is None:
            corners = self.find_page_corners(image)
            if corners is None:
                logger.error(
                    f"CropOnLMarkers: Could not find marker points or page "
                    f"corners for '{filename}'."
                )
                return None

            logger.warning(
                f"CropOnLMarkers: L-markers were incomplete for '{filename}'; "
                "using the visible page outline instead."
            )

        if config.outputs.show_image_level >= 4:
            debug_image = (
                image.copy()
                if len(image.shape) == 3
                else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            )

            # Draw all detected marker points.
            if marker_points is not None:
                for point in marker_points:
                    cv2.circle(
                        debug_image,
                        tuple(point.astype(int)),
                        4,
                        (255, 0, 0),
                        -1,
                    )

            # Draw the estimated outer page quadrilateral.
            cv2.polylines(
                debug_image,
                [corners.astype(int)],
                True,
                (0, 255, 0),
                3,
            )

            InteractionUtils.show(
                "CropOnLMarkers: detected corners",
                debug_image,
                config=config,
            )

        if config.outputs.show_image_level >= 5:
            InteractionUtils.show(
                "CropOnLMarkers: detected edges",
                edges,
                config=config,
            )

        return ImageUtils.four_point_transform(image, corners)