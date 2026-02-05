# Physical Network Latency Demo – Ball Maze Game


A physical ball maze game where two knobs control the board tilt (X/Y),
and a third knob adjusts *network latency* in real time. Built as a hands-on
demo for explaining why low latency matters (IoT / networking / control systems).

## Concept
- Knob X/Y -> control commands
- Knob Latency -> changes artificial delay
- Commands go through a network latency proxy (delay/jitter/drop can be added)
- Device actuates servos / motors based on delayed commands
