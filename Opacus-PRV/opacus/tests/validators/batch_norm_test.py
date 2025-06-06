#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch.nn as nn
from opacus.validators.errors import ShouldReplaceModuleError
from opacus.validators.module_validator import ModuleValidator


class BatchNormValidator_test(unittest.TestCase):

    def setUp(self):
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm3d(4)
        self.bns = nn.SyncBatchNorm(4)
        self.mv = ModuleValidator.VALIDATORS
        self.mf = ModuleValidator.FIXERS

    def test_validate(self):
        val1 = self.mv[type(self.bn1)](self.bn1)
        val2 = self.mv[type(self.bn2)](self.bn2)
        val3 = self.mv[type(self.bn3)](self.bn3)
        vals = self.mv[type(self.bns)](self.bns)

        self.assertEqual(len(val1), 1)
        self.assertEqual(len(val2), 1)
        self.assertEqual(len(val3), 1)
        self.assertEqual(len(vals), 1)

        self.assertTrue(isinstance(val1[0], ShouldReplaceModuleError))
        self.assertTrue(isinstance(val2[0], ShouldReplaceModuleError))
        self.assertTrue(isinstance(val3[0], ShouldReplaceModuleError))
        self.assertTrue(isinstance(vals[0], ShouldReplaceModuleError))

    def test_fix(self):
        fix1 = self.mf[type(self.bn1)](self.bn1)
        fix2 = self.mf[type(self.bn2)](self.bn2)
        fix3 = self.mf[type(self.bn3)](self.bn3)
        fixs = self.mf[type(self.bns)](self.bns)

        self.assertTrue(isinstance(fix1, nn.GroupNorm))
        self.assertTrue(isinstance(fix2, nn.GroupNorm))
        self.assertTrue(isinstance(fix3, nn.GroupNorm))
        self.assertTrue(isinstance(fixs, nn.GroupNorm))
