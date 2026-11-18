import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from services.portrait.registration import PreparationError, manual_eye_features, register, validate_calibration

EXAMPLES = Path(__file__).resolve().parents[3] / 'docs/examples'


class Source:
    """The pinned Source2Info call order, with model boundaries observable."""
    def _crop(self, rgb, last_lmk=None, **kwargs):
        boxes, _ = self.insightface_det(rgb)
        lmk = self.landmark106(rgb, boxes[0])
        self.landmark203(lmk, self.landmark203.dsize)
        return rgb, np.eye(3), lmk

    def __call__(self, rgb, **kwargs):
        crop, _, _ = self._crop(rgb, **kwargs)
        eyes = self._get_eye_info(crop)
        return eyes, self._get_kp_info(crop), self._get_f3d(crop)


class Registrar(SimpleNamespace):
    def __call__(self, path, max_dim, **kwargs):
        return self.source2info(np.zeros((512, 512, 3), dtype=np.uint8), **kwargs)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.source = Source()
        self.source.insightface_det = Mock(return_value=(np.empty((0, 5)), None))
        self.source.landmark106 = Mock(return_value=np.ones((106, 2)))
        self.source.landmark203 = Mock(return_value=np.ones((203, 2)), dsize=224)
        self.source._get_eye_info = Mock(return_value=[np.ones((1, 2)), np.ones((1, 6))])
        self.source._get_kp_info = Mock(return_value='motion')
        self.source._get_f3d = Mock(return_value='appearance')
        self.registrar = Registrar(source2info=self.source)

    def test_two_original_profiles_only_replace_the_requested_stages(self):
        for profile in ('ditto-original', 'ditto-q'):
            calibration = json.loads((EXAMPLES / (profile + '.json')).read_text())
            validate_calibration(calibration)
            eyes, motion, appearance = register(self.registrar, 'asset', calibration)
            self.assertEqual((motion, appearance), ('motion', 'appearance'))
            box = self.source.landmark106.call_args.args[1]
            np.testing.assert_array_equal(box, [*calibration['face_box'], 1])
            self.assertEqual([v.shape for v in eyes], [(1, 2), (1, 6)])
        self.source.insightface_det.assert_not_called()
        self.source._get_eye_info.assert_called_once()
        for called in (self.source.landmark106, self.source.landmark203,
                       self.source._get_kp_info, self.source._get_f3d):
            self.assertEqual(called.call_count, 2)
        with self.assertRaises(PreparationError) as error:
            register(self.registrar, 'next-asset', {})
        self.assertEqual((error.exception.stage, error.exception.reason), ('face_detection', 'no_face'))

    def test_eye_offsets_use_crop_axes_and_interleaved_lr_order(self):
        left = dict(outer=[0, 4], inner=[8, 4], upper=[4, 2], lower=[4, 6], pupil=[4, 5])
        right = left | dict(outer=[18, 4], inner=[10, 4], pupil=[14, 4])
        matrix = [[0, -1, 20], [1, 0, 0], [0, 0, 1]]
        opened, moved = manual_eye_features(dict(left=left, right=right), matrix, 512)
        np.testing.assert_allclose(opened, [[1.5, 1.5]])
        np.testing.assert_allclose(moved * 512, [[1, 0, -1, 0, 0, 0]])

    def test_native_stage_errors_remain_distinct(self):
        self.source.insightface_det.return_value = (np.ones((1, 5)), None)
        for attr, stage in [('landmark106', 'landmark106'), ('landmark203', 'landmark203'),
                            ('_get_eye_info', 'eye_mesh'), ('_get_kp_info', 'motion_features'),
                            ('_get_f3d', 'appearance_features')]:
            with self.subTest(stage):
                fn = getattr(self.source, attr)
                fn.side_effect = RuntimeError('local-only details')
                with self.assertRaises(PreparationError) as error:
                    register(self.registrar, 'asset', {})
                self.assertEqual(error.exception.stage, stage)
                fn.side_effect = None
        self.assertEqual(register(self.registrar, 'asset', {})[1:], ('motion', 'appearance'))

    def test_invalid_manual_marks(self):
        for bad in ({}, {'face_box': [0, 1, True, 3]}, {'face_box': [2, 4, 2, 8]},
                    {'face_box': [0, 0, 641, 720]}, {'face_box': [0, 0, float('nan'), 60]},
                    {'eyes': {}}, {'eyes': {'left': {}, 'right': {}}}, {'mesh': []}):
            with self.subTest(bad), self.assertRaises(ValueError):
                validate_calibration(bad)
