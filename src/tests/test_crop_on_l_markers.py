from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.processors.manager import PROCESSOR_MANAGER
from src.processors.CropOnLMarkers import InteractionUtils

CropOnLMarkers = PROCESSOR_MANAGER.processors["CropOnLMarkers"]

SAMPLE_IMAGES_DIR = (
    Path(__file__).resolve().parents[2]
    / "samples"
    / "community"
    / "DomBrzezinski"
    / "photos"
)


def test_falls_back_to_page_outline_when_markers_are_incomplete(monkeypatch):
    monkeypatch.setattr(
        InteractionUtils,
        "show",
        staticmethod(lambda *args, **kwargs: None),
    )
    config = SimpleNamespace(
        outputs=SimpleNamespace(show_image_level=5),
    )
    image_ops = SimpleNamespace(tuning_config=config)
    processor = CropOnLMarkers(
        options={},
        relative_dir=str(SAMPLE_IMAGES_DIR),
        image_instance_ops=image_ops,
    )

    for filename in ("omr_survey_1.jpg", "omr_survey_2.jpg"):
        image = cv2.imread(
            str(SAMPLE_IMAGES_DIR / filename),
            cv2.IMREAD_GRAYSCALE,
        )
        assert image is not None
        image = cv2.resize(image, (666, 820))

        if filename == "omr_survey_2.jpg":
            processor.find_marker_points = lambda *_args: None

        marker_points = processor.find_marker_points(
            image,
            processor.preprocess(image),
        )
        assert processor.get_outer_corners(marker_points) is None

        page_corners = processor.find_page_corners(image)
        assert page_corners is not None
        assert np.ptp(page_corners[:, 0]) > 400
        assert np.ptp(page_corners[:, 1]) > 500

        warped = processor.apply_filter(image, filename)
        assert warped is not None
        assert warped.size > 0
