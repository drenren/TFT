"""
Unit tests for TFT core components.

Run with: pytest tests/test_components.py
"""

import torch
import pytest
import sys
sys.path.append('..')

from tft.models.components import (
    GatedLinearUnit,
    GatedResidualNetwork,
    GateAddNorm,
    TimeDistributed,
)


class TestGatedLinearUnit:
    """Tests for Gated Linear Unit."""

    def test_glu_output_shape(self):
        """Test that GLU produces correct output shape."""
        batch_size = 16
        input_dim = 32
        output_dim = 64

        glu = GatedLinearUnit(input_dim, output_dim, dropout=0.1)
        x = torch.randn(batch_size, input_dim)

        output = glu(x)

        assert output.shape == (batch_size, output_dim)

    def test_glu_default_output_dim(self):
        """Test GLU with default output_dim (same as input)."""
        batch_size = 16
        input_dim = 32

        glu = GatedLinearUnit(input_dim)
        x = torch.randn(batch_size, input_dim)

        output = glu(x)

        assert output.shape == (batch_size, input_dim)

    def test_glu_3d_input(self):
        """Test GLU with 3D input (batch, time, features)."""
        batch_size = 16
        time_steps = 10
        input_dim = 32
        output_dim = 64

        glu = GatedLinearUnit(input_dim, output_dim)
        x = torch.randn(batch_size, time_steps, input_dim)

        output = glu(x)

        assert output.shape == (batch_size, time_steps, output_dim)


class TestGatedResidualNetwork:
    """Tests for Gated Residual Network."""

    def test_grn_output_shape(self):
        """Test that GRN produces correct output shape."""
        batch_size = 16
        input_dim = 32
        hidden_dim = 64
        output_dim = 48

        grn = GatedResidualNetwork(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)

        output = grn(x)

        assert output.shape == (batch_size, output_dim)

    def test_grn_with_context(self):
        """Test GRN with context vector."""
        batch_size = 16
        input_dim = 32
        hidden_dim = 64
        output_dim = 48
        context_dim = 16

        grn = GatedResidualNetwork(
            input_dim, hidden_dim, output_dim, context_dim=context_dim
        )
        x = torch.randn(batch_size, input_dim)
        context = torch.randn(batch_size, context_dim)

        output = grn(x, context=context)

        assert output.shape == (batch_size, output_dim)

    def test_grn_skip_connection(self):
        """Test that GRN skip connection works."""
        batch_size = 16
        input_dim = 32
        hidden_dim = 64

        grn = GatedResidualNetwork(input_dim, hidden_dim, input_dim)
        x = torch.randn(batch_size, input_dim)

        output = grn(x)

        # Output should be different from input (not identity)
        assert not torch.allclose(output, x)
        # But should have same shape
        assert output.shape == x.shape

    def test_grn_gradient_flow(self):
        """Test that gradients flow through GRN."""
        input_dim = 32
        hidden_dim = 64

        grn = GatedResidualNetwork(input_dim, hidden_dim, input_dim)
        x = torch.randn(1, input_dim, requires_grad=True)

        output = grn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestGateAddNorm:
    """Tests for GateAddNorm."""

    def test_gate_add_norm_output_shape(self):
        """Test that GateAddNorm produces correct output shape."""
        batch_size = 16
        input_dim = 32

        gan = GateAddNorm(input_dim)
        x = torch.randn(batch_size, input_dim)
        residual = torch.randn(batch_size, input_dim)

        output = gan(x, residual)

        assert output.shape == (batch_size, input_dim)

    def test_gate_add_norm_residual_connection(self):
        """Test that residual connection is applied."""
        batch_size = 16
        input_dim = 32

        gan = GateAddNorm(input_dim, dropout=0.0)  # No dropout for deterministic test
        x = torch.zeros(batch_size, input_dim)
        residual = torch.ones(batch_size, input_dim)

        output = gan(x, residual)

        # Output should be influenced by residual
        assert not torch.allclose(output, x)


class TestTimeDistributed:
    """Tests for TimeDistributed wrapper."""

    def test_time_distributed_linear(self):
        """Test TimeDistributed with Linear layer."""
        batch_size = 16
        time_steps = 10
        input_dim = 32
        output_dim = 64

        linear = torch.nn.Linear(input_dim, output_dim)
        td_linear = TimeDistributed(linear)

        x = torch.randn(batch_size, time_steps, input_dim)
        output = td_linear(x)

        assert output.shape == (batch_size, time_steps, output_dim)

    def test_time_distributed_2d_input(self):
        """Test TimeDistributed with 2D input (no time dimension)."""
        batch_size = 16
        input_dim = 32
        output_dim = 64

        linear = torch.nn.Linear(input_dim, output_dim)
        td_linear = TimeDistributed(linear)

        x = torch.randn(batch_size, input_dim)
        output = td_linear(x)

        assert output.shape == (batch_size, output_dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
