"""Browser demo layer: a serving shell around the project's inference core.

Everything that touches the streaming maths lives in `wanstreamer` proper
(`stream.FewStepStreamer`, `blockcausal`, `kvcache`, `rope`). This package only
drives it: session management, JPEG framing, a websocket, world generation from
text, and the UI.
"""
