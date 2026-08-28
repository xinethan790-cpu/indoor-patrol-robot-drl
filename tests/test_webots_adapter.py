from __future__ import annotations

import numpy as np

from indoor_patrol_rl.envs.webots_patrol import _webots_scan_to_policy_order


def test_webots_scan_uses_forward_zero_and_left_positive_order() -> None:
    # In the Webots LDS image, forward is raw index 4 and a left-hand return is
    # at the smaller raw index 2. The shared policy convention places them at
    # output indices 0 and 2 respectively.
    raw = np.arange(8, dtype=np.float64)

    ordered = _webots_scan_to_policy_order(raw)

    np.testing.assert_array_equal(ordered, [4, 3, 2, 1, 0, 7, 6, 5])
