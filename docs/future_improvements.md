# Future improvements

## Camera capture timestamps via NTP (instead of a constant measured delay)

**Now:** frames are stamped on arrival at the host; the camera delay (exposure → arrival) is
measured once per setup with a moving ChArUco board (`scripts/estimate_time_offset.py`) and
assumed constant. First measurement (laptop on the camera network, 2026-11-23): 927 ms,
std 5 ms over 30 s; stream pts did not fit better than arrival times.

**Open question:** does the delay stay the same across reconnects, camera restarts and
different machines? Check: several 30 s board recordings, each after a fresh start
(one after a camera restart). If they agree within ~±10 ms, keep the current approach.

**If it varies by tens of ms:** use the camera's own capture timestamps.
- Axis sends RTCP sender reports that map RTP timestamps to the camera's NTP clock, giving a
  capture time per frame. OpenCV/FFmpeg do not expose this; GStreamer `rtspsrc` does
  (reference timestamp meta), so this means a GStreamer-based grabber.
- The camera sits in the private 192.168.1.0/24 network: the workstation must serve NTP to it.
  Camera, recording host and Vicon PC clocks must agree to a few ms.
- Still validate once with the board recording (does the RTP timestamp mark exposure start?
  how well do the clocks agree?).
- Vicon side needs no change: host receive time − SDK-reported latency is already accurate to a
  few ms; an absolute Vicon clock (timecode) would need timecode/genlock hardware.

## Other ideas
- Rolling-shutter readout time: not measurable from the first board recording; needs fast
  vertical image motion. Only matters for fast objects.
- Live latency: ~0.9 s are inside the camera. Try Axis low-latency mode, Zipstream off, shorter
  GOP, then re-measure.
- Rerun logger for debugging (see CLAUDE.md layout).
