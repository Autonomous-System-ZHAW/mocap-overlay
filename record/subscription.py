"""Fan-out queue used by the background streams (camera, Vicon) to feed recorders."""

import queue


class Subscription:
    """Queue of every item (frame / sample). If the consumer falls behind, the oldest frames are dropped and counted."""

    def __init__(self, maxsize: int):
        self.queue: queue.Queue = queue.Queue(maxsize)
        self.dropped = 0

    def put(self, item) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, timeout: float | None = None):
        return self.queue.get(timeout=timeout)
