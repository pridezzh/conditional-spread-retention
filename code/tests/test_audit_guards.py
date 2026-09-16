# -*- coding: utf-8 -*-
"""针对本次科学校核发现的问题做最小回归测试。"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from discriminant import _lam_of, _select_K, marginal_separation  # noqa: E402
from metrics import conditional_reference_samples  # noqa: E402


class DiscriminantAuditTests(unittest.TestCase):
    def test_marginal_separation_is_dimension_invariant_under_feature_duplication(self):
        """重复同一坐标不应因多乘一次维度而让分离度虚高。"""
        rng = np.random.default_rng(4)
        x = np.r_[rng.normal(-3, 0.2, 120), rng.normal(3, 0.2, 120)][:, None]
        sep_1, _ = marginal_separation(x, kmax=3, seed=4, pca_dim=8)
        sep_4, _ = marginal_separation(np.repeat(x, 4, axis=1), kmax=3,
                                       seed=4, pca_dim=8)
        self.assertAlmostEqual(sep_1, sep_4, places=6)
        self.assertGreater(sep_1, 0.95)

    def test_lambda_uses_sample_weighted_pooled_variance(self):
        """小簇不应与大簇等权；公式应等于逐样本池化残差。"""
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
        """K=1 必须是可达结果，不能对任何数据都强制返回 K>=2。"""
        x = np.random.default_rng(12).normal(size=(240, 2))
        self.assertEqual(_select_K(x, kmax=5, seed=12, n_null=49), 1)


class ConditionalMetricAuditTests(unittest.TestCase):
    def test_constant_condition_uses_marginal_reference_without_label_input(self):
        """无条件设置应抽取整个边际，且排除查询样本自身。"""
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
        """理论护栏：任意单次确定性噪声映射不必退化成条件均值。"""
        x0 = np.linspace(-3, 3, 1000)
        one_pass = np.where(x0 < 0, -2.0, 2.0)
        self.assertAlmostEqual(one_pass.mean(), 0.0, places=12)
        self.assertGreater(one_pass.var(), 3.9)
        self.assertEqual(set(np.unique(one_pass)), {-2.0, 2.0})


if __name__ == "__main__":
    unittest.main()
