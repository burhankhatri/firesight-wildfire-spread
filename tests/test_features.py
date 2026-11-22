import numpy as np

from src.features import engineer, FEATURES


def _batch(th=0.0, vs=5.0):
    n, s = 1, 64
    b = {v: np.zeros((n, s, s), np.float32) for v in
         ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc"]}
    b["th"][:] = th
    b["vs"][:] = vs
    b["tmmn"][:] = 280
    b["tmmx"][:] = 300
    prev = np.zeros((n, s, s), np.float32)
    prev[0, 32, 32] = 1
    b["PrevFireMask"] = prev
    return b


def test_all_features_present_with_image_shape():
    f = engineer(_batch())
    assert set(FEATURES) <= set(f)
    for k in FEATURES:
        assert f[k].shape == (1, 64, 64), k


def test_distance_to_fire_is_zero_on_fire_and_grows_away():
    f = engineer(_batch())
    assert f["dist_to_fire"][0, 32, 32] == 0
    assert f["dist_to_fire"][0, 32, 40] == 8


def test_upwind_fire_sees_fire_upwind_only():
    # Wind blowing FROM the north (th=0): the fire is upwind of pixels south of it.
    f = engineer(_batch(th=0.0))
    south, north = f["upwind_fire_5"][0, 37, 32], f["upwind_fire_5"][0, 27, 32]
    assert south > 0
    assert north == 0


def test_wind_components_point_downwind():
    f = engineer(_batch(th=0.0, vs=4.0))  # from north -> blows toward south
    assert np.isclose(f["wind_v"][0, 0, 0], -4.0)
    assert np.isclose(f["wind_u"][0, 0, 0], 0.0, atol=1e-5)


def test_temp_range():
    f = engineer(_batch())
    assert np.allclose(f["temp_range"], 20)


def test_distinct_fires_drops_shifted_copies():
    from src.data import distinct_fires
    a = np.zeros((3, 64, 64), np.float32)
    a[0, 20:30, 20:35] = 1          # fire 1
    a[1, 25:35, 10:25] = 1          # same fire, shifted crop
    a[2, 40:44, 40:60] = 1          # a different fire
    t = {"FireMask": a}
    assert distinct_fires(t, [0, 1, 2], n=3) == [0, 2]
