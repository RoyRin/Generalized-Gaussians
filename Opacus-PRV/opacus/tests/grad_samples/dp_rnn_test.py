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

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.layers import DPGRU, DPLSTM, DPRNN
from opacus.utils.packed_sequences import _gen_packed_data

from .common import GradSampleHooks_test

MODELS = [
    DPRNN,
    DPGRU,
    DPLSTM,
]


class DPRNNAdapter(nn.Module):
    """
    Adapter for DPRNN family.
    RNN returns a tuple, but our testing tools need the model to return a single tensor in output.
    We do this adaption here.
    """

    def __init__(self, dp_rnn):
        super().__init__()
        self.dp_rnn = dp_rnn

    def forward(self, x):
        out, _rest = self.dp_rnn(x)
        return out


class RNN_test(GradSampleHooks_test):

    @given(
        model=st.one_of([st.just(model) for model in MODELS]),
        N=st.integers(1, 3),
        T=st.integers(1, 3),
        D=st.integers(4, 5),
        H=st.integers(8, 10),
        num_layers=st.sampled_from([1, 2]),
        bias=st.booleans(),
        batch_first=st.booleans(),
        bidirectional=st.booleans(),
        using_packed_sequences=st.booleans(),
        packed_sequences_sorted=st.booleans(),
    )
    @settings(deadline=30000)
    def test_rnn(
        self,
        model,
        N: int,
        T: int,
        D: int,
        H: int,
        num_layers: int,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
        using_packed_sequences: bool,
        packed_sequences_sorted: bool,
    ):
        torch.use_deterministic_algorithms(False)
        rnn = model(
            D,
            H,
            num_layers=num_layers,
            batch_first=batch_first,
            bias=bias,
            bidirectional=bidirectional,
        )
        rnn = DPRNNAdapter(rnn)

        if using_packed_sequences:
            x = _gen_packed_data(N, T, D, batch_first, packed_sequences_sorted)
        else:
            if batch_first:
                x = torch.randn([N, T, D])
            else:
                x = torch.randn([T, N, D])

        self.run_test(x, rnn, batch_first=batch_first, ew_compatible=False)
