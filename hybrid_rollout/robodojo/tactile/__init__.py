"""Tactile sensing on the dual ARX X5 grippers.

Every rollout records two sensor models:
  contact  - IsaacLab ContactSensor per finger link, filtered against every task block.
  gelsight - a TacSL GelSight image per fingertip (the image only, as a real sensor gives), from either
             a virtual 1 mm gel ray-cast against the blocks (observation-only; physics unchanged) or,
             with ``mounted``, a compliant gel pad imaged by a depth camera (changes the fingertip contact).
"""
