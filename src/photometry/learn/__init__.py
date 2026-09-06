"""Learned components (optional; requires `torch`).

Design rule for this package: a network proposes, physics verifies. The
net amortizes the expensive search stages of the stack (spin-pole grid,
torque-free multi-start) into a millisecond proposal; every proposal is
then scored and polished by the same forward-model cost the classical
pipeline uses, so nothing the net says is trusted without a physics
certificate. The classical stack stays torch-free.
"""
