import numpy as np

from qfeval_data import util


class TestUtil:
    def test_are_broadcastable_shapes(self) -> None:
        assert util.are_broadcastable_shapes((3, 2, 1), (3, 1, 2))
        assert not util.are_broadcastable_shapes((3, 2, 1), (3, 3, 1))
        assert util.are_broadcastable_shapes((3, 2), (3, 3, 2))
        assert not util.are_broadcastable_shapes((3, 2), (3, 2, 1))

    def test_floor_time(self) -> None:
        assert util.floor_time(
            np.datetime64("2020-01-02 03:04:59"), np.timedelta64(1, "m")
        ) == np.datetime64("2020-01-02 03:04")
        np.testing.assert_array_equal(
            util.floor_time(
                np.array(
                    [
                        "2020-01-02 03:04:59",
                        "2020-01-02 03:05:00",
                        "2020-01-02 03:05:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(1, "m"),
            ),
            np.array(
                ["2020-01-02 03:04", "2020-01-02 03:05", "2020-01-02 03:05"],
                "datetime64[s]",
            ),
        )
        np.testing.assert_array_equal(
            util.floor_time(
                np.array(
                    [
                        "2020-01-31 23:59:59",
                        "2020-02-01 00:00:00",
                        "2020-02-01 00:00:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(1, "M"),
            ),
            np.array(
                ["2020-01", "2020-02", "2020-02"],
                "datetime64[M]",
            ),
        )
        # NOTE: 2021-04-11 is Sunday.
        np.testing.assert_array_equal(
            util.floor_time(
                np.array(
                    [
                        "2021-04-10 23:59:59",
                        "2021-04-11 00:00:00",
                        "2021-04-11 00:00:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(7, "D"),
            ),
            np.array(
                ["2021-04-04", "2021-04-11", "2021-04-11"],
                "datetime64[D]",
            ),
        )
        np.testing.assert_array_equal(
            util.floor_time(
                np.array(
                    [
                        "1999-12-31 23:59:59",
                        "2000-01-01 00:00:00",
                        "2000-01-01 00:00:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(100, "Y"),
            ),
            np.array(
                ["1900", "2000", "2000"],
                "datetime64[Y]",
            ),
        )

    def test_ceil_time(self) -> None:
        np.testing.assert_array_equal(
            util.ceil_time(
                np.array(
                    [
                        "2020-01-02 03:04:59",
                        "2020-01-02 03:05:00",
                        "2020-01-02 03:05:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(1, "m"),
            ),
            np.array(
                ["2020-01-02 03:05", "2020-01-02 03:05", "2020-01-02 03:06"],
                "datetime64[s]",
            ),
        )
        np.testing.assert_array_equal(
            util.ceil_time(
                np.array(
                    [
                        "2020-01-31 23:59:59",
                        "2020-02-01 00:00:00",
                        "2020-02-01 00:00:01",
                    ],
                    "datetime64[s]",
                ),
                np.timedelta64(1, "M"),
            ),
            np.array(
                ["2020-02", "2020-02", "2020-03"],
                "datetime64[M]",
            ),
        )

    def test_sha1(self) -> None:
        assert util.sha1("") == "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        assert util.sha1("foo") == "0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33"
        # TODO(masano): Fix this test.
        # Test if it is possible to calculate np.ndarray's hash.
        # assert (
        #     util.sha1(np.zeros((2, 2), dtype=np.float32))
        #     == "fd2a162ee08d093837c9f5e393aafc0ca281ba61"
        # )
        # Hash result should be different if its shape is different.
        # assert (
        #     util.sha1(np.zeros((1, 4), dtype=np.float32))
        #     == "39fbd17641dd0304077a72c30108d264f5d7f533"
        # )
