"""Per-reference calibration for the pinned Ditto still-image registrar."""
from contextlib import contextmanager
from copy import copy
from math import dist


class PreparationError(Exception):
    def __init__(self, stage, reason='stage_failed'):
        super().__init__(stage, reason)
        self.stage = stage
        self.reason = reason


@contextmanager
def stage(name):
    try:
        yield
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(name) from error


def validate_calibration(value):
    if not isinstance(value, dict) or not value or value.keys() - {'face_box', 'eyes'}:
        raise ValueError('Invalid calibration')

    def vector(coords, limits):
        if (not isinstance(coords, list) or len(coords) != len(limits)
                or any(type(n) not in (int, float) or not 0 <= n <= limit
                       for n, limit in zip(coords, limits))):
            raise ValueError('Expected pixel coordinates on the 640x720 asset')

    if 'face_box' in value:
        box = value['face_box']
        vector(box, (640, 720, 640, 720))
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            raise ValueError('Empty face box')
    if 'eyes' in value:
        eyes = value['eyes']
        if not isinstance(eyes, dict) or eyes.keys() != {'left', 'right'}:
            raise ValueError('Expected both eyes in image order')
        for eye in eyes.values():
            if not isinstance(eye, dict) or eye.keys() != {'outer', 'inner', 'upper', 'lower', 'pupil'}:
                raise ValueError('Incomplete eye marks')
            for point in eye.values():
                vector(point, (639, 719))
            if dist(eye['outer'], eye['inner']) < 1:
                raise ValueError('Eye corners coincide')


def manual_eye_features(eyes, crop_to_original, size):
    """Use the pinned EyeAttrUtilsByMP convention with hand-marked 2D lids."""
    import numpy as np

    matrix = np.asarray(crop_to_original)[:2]
    opens, moves = [], []
    for side in ('left', 'right'):
        marks = eyes[side]
        points = np.array([marks[k] for k in ('outer', 'inner', 'upper', 'lower', 'pupil')])
        # MP uses crop-normalized coordinates. Preserve the rotation of Ditto's actual crop.
        outer, inner, upper, lower, pupil = np.linalg.solve(
            matrix[:, :2], (points - matrix[:, 2]).T).T / size
        width = np.linalg.norm(outer - inner)
        opens.append(3 * np.linalg.norm(upper - lower) / width)
        offset = pupil - (outer + inner) / 2
        move = offset - (outer - inner) / width * np.linalg.norm(offset)
        moves.append([*move, 0])
    return [np.array([opens], dtype=np.float32),
            np.array(moves, dtype=np.float32).T.reshape(1, 6)]


def register(registrar, image, calibration, **crop):
    import numpy as np

    # Copy only the orchestrators. Each copy calls the slot's existing model objects.
    local = copy(registrar)
    source = local.source2info = copy(registrar.source2info)
    detector = source.insightface_det

    def detect(rgb):
        with stage('face_detection'):
            if 'face_box' in calibration:
                return np.array([[*calibration['face_box'], 1]], dtype=np.float32), None
            boxes, keypoints = detector(rgb)
            if boxes is None or not len(boxes):
                raise PreparationError('face_detection', 'no_face')
            return boxes, keypoints
    source.insightface_det = detect

    def checked(fn, name):
        def call(*args, **kwargs):
            with stage(name):
                result = fn(*args, **kwargs)
                if name in ('landmark106', 'landmark203', 'eye_mesh'):
                    values = result if name == 'eye_mesh' and result is not None else (result,)
                    if any(v is None or not np.size(v) or not np.isfinite(v).all() for v in values):
                        raise PreparationError(name, 'invalid_features')
                return result
        return call

    for name in ('landmark106', 'landmark203'):
        # The registrar reads dsize on the 203 model as well as calling it.
        model = getattr(source, name)
        wrapped = checked(model, name)
        if name == 'landmark203':
            wrapped.dsize = model.dsize
        setattr(source, name, wrapped)
    original_crop = source._crop
    crop_matrix = None

    def run_crop(*args, **kwargs):
        nonlocal crop_matrix
        with stage('crop'):
            result = original_crop(*args, **kwargs)
            crop_matrix = result[1]
            return result
    source._crop = run_crop
    if 'eyes' in calibration:
        def manual_eyes(rgb):
            return manual_eye_features(calibration['eyes'], crop_matrix, rgb.shape[0])
        source._get_eye_info = checked(manual_eyes, 'eye_mesh')
    else:
        source._get_eye_info = checked(source._get_eye_info, 'eye_mesh')
    source._get_kp_info = checked(source._get_kp_info, 'motion_features')
    source._get_f3d = checked(source._get_f3d, 'appearance_features')
    with stage('registration'):
        return local(str(image), max_dim=720, **crop)
