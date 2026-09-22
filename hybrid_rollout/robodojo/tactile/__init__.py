"""Tactile sensing on the dual ARX X5 grippers (observation-only; never changes physics).

Three sensor models share one rollout:
  contact  - IsaacLab ContactSensor per finger link, filtered against every task block.
  taxel    - the same PhysX contact points binned into a pressure/shear grid on each finger face.
  gelsight - TacSL-style GelSight: a virtual gel on each fingertip, its indentation rendered with
             IsaacLab's TacSL GelSight renderer, plus TacSL's penalty normal/shear force field.
"""
