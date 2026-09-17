# -*- coding: utf-8 -*-
"""Minimal regression tests for the issues found during the scientific audit."""
import os
import json
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))

from discriminant import _lam_of, _select_K, marginal_separation  # noqa: E402
from metrics import conditional_reference_samples  # noqa: E402
from verify_impossibility import knn_variance_ratio  # noqa: E402
from provenance import (RESULT_SCHEMA_VERSION, protocol_fingerprint,
                        require_merge_compatible)  # noqa: E402


class DiscriminantAuditTests(unittest.TestCase):
    def test_marginal_separation_is_dimension_invariant_under_feature_duplication(self):
        """Repeating the same coordinate should not inflate the separation via an extra dimension."""
        rng = np.random.default_rng(4)
        x = np.r_[rng.normal(-3, 0.2, 120), rng.normal(3, 0.2, 120)][:, None]
        sep_1, _ = marginal_separation(x, kmax=3, seed=4, pca_dim=8)
        sep_4, _ = marginal_separation(np.repeat(x, 4, axis=1), kmax=3,
                                       seed=4, pca_dim=8)
        self.assertAlmostEqual(sep_1, sep_4, places=6)
        self.assertGreater(sep_1, 0.95)

    def test_lambda_uses_sample_weighted_pooled_variance(self):
        """Small clusters should not be weighted equally with large ones; the formula
        should equal the per-sample pooled residual."""
        rng = np.random.default_rng(7)
        x = np.r_[rng.normal(-4, 0.1, (180, 2)),
                  rng.normal(4, 1.0, (20, 2))]
        lam, details = _lam_of(x, 2, seed=7)
        centers, labels = details
        dmin = np.linalg.norm(centers[0] - centers[1])
        pooled = ((x - centers[labels]) ** 2).sum(axis=1).mean()
        expected = dmin / (2 * np.sqrt(pooled / x.shape[1]))
        self.assertAlmostEqual(lam, expected, places=10)

    def test_unimodal_gaussian_can_select_one_cluster(self):
        """K=1 must be a reachable result; the method must not force K>=2 for arbitrary data."""
        x = np.random.default_rng(12).normal(size=(240, 2))
        self.assertEqual(_select_K(x, kmax=5, seed=12, n_null=49), 1)


class ConditionalMetricAuditTests(unittest.TestCase):
    def test_constant_condition_uses_marginal_reference_without_label_input(self):
        """The unconditional setting should draw the whole marginal and exclude the query sample itself."""
        c = np.zeros((100, 1))
        x = np.arange(100)[:, None]
        refs = conditional_reference_samples(
            c, x, np.zeros((2, 1)), n_ref=40, seed=3,
            exclude_indices=np.array([5, 80]))
        self.assertEqual(refs[0].shape, (40, 1))
        self.assertNotIn(5, refs[0][:, 0])
        self.assertNotIn(80, refs[1][:, 0])
        self.assertGreater(len(np.unique(refs[0][:, 0] // 10)), 3)

    def test_one_pass_noise_map_can_preserve_two_modes(self):
        """Theoretical guardrail: an arbitrary single deterministic noise map need not collapse to the conditional mean."""
        x0 = np.linspace(-3, 3, 1000)
        one_pass = np.where(x0 < 0, -2.0, 2.0)
        self.assertAlmostEqual(one_pass.mean(), 0.0, places=12)
        self.assertGreater(one_pass.var(), 3.9)
        self.assertEqual(set(np.unique(one_pass)), {-2.0, 2.0})

    def test_dependence_does_not_imply_nonzero_conditional_mean(self):
        """Dependent coupling can still have mean independence; hence "not independent iff no collapse" does not hold."""
        x0 = np.repeat(np.array([0.5, 1.0, 2.0, 3.0]), 2)
        sign = np.tile(np.array([-1.0, 1.0]), 4)
        x1 = sign * np.abs(x0)
        conditional_means = x1.reshape(-1, 2).mean(axis=1)
        conditional_second_moments = (x1 ** 2).reshape(-1, 2).mean(axis=1)
        self.assertTrue(np.allclose(conditional_means, 0.0))
        self.assertGreater(np.ptp(conditional_second_moments), 8.0)

    def test_knn_variance_correction_uses_total_variance(self):
        """The residual fraction of a Bernoulli mixture coupling is 1-alpha^2."""
        self.assertAlmostEqual(knn_variance_ratio(0.5, 20), 0.2875)
        self.assertAlmostEqual(knn_variance_ratio(0.0, 20), 0.05)
        self.assertAlmostEqual(knn_variance_ratio(1.0, 20), 1.0)


class ResultIntegrityTests(unittest.TestCase):
    def test_legacy_or_cross_revision_merge_is_rejected(self):
        """Old caches without schema/fingerprint must not be silently merged with new seeds."""
        with self.assertRaises(RuntimeError):
            require_merge_compatible({}, "expected", "legacy.json")
        compatible = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "provenance": {"protocol_id": "expected"},
        }
        require_merge_compatible(compatible, "expected", "current.json")

    def test_protocol_fingerprint_changes_with_source(self):
        """The protocol fingerprint used for result merging must genuinely depend on the source file contents."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "protocol.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("VALUE = 1\n")
            first, _ = protocol_fingerprint(tmp, ["protocol.py"])
            with open(path, "w", encoding="utf-8") as f:
                f.write("VALUE = 2\n")
            second, _ = protocol_fingerprint(tmp, ["protocol.py"])
        self.assertNotEqual(first, second)

    def test_training_summaries_equal_per_seed_aggregates(self):
        """Table summaries must be exactly recomputable from the per-seed raw values."""
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, "results", "loss_ladder.json")
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        for method, stored in report["summary"].items():
            values = [row[method]["rho"] for row in report["per_seed"].values()]
            self.assertAlmostEqual(float(np.mean(values)), stored["rho"], places=12)


if __name__ == "__main__":
    unittest.main()
