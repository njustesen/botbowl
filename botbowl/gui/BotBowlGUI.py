import os
import pyglet
import time
from multiprocessing import Process, Pipe
from pyglet import image


def worker(remote, parent_remote, config):
    parent_remote.close()
    width = 1200
    height = 800
    window = pyglet.window.Window(width, height)
    main_batch = pyglet.graphics.Batch()

    file_dir = os.path.dirname(__file__)
    image_path = os.path.join(file_dir, "../web/static/img/")

    # text = pyglet.text.Label(text="", x=100, y=100, batch=main_batch)
    pitch_img = image.load(os.path.join(image_path, 'arenas/pitch/nice-26x15.jpg'))
    pitch_img.anchor_x = pitch_img.width // 2
    pitch_img.anchor_y = pitch_img.height
    dugout_left = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-left.jpg'))
    dugout_left.anchor_x = dugout_left.width
    dugout_left.anchor_y = dugout_left.height
    dugout_right = image.load(os.path.join(image_path, 'arenas/dugouts/dugout-right.jpg'))
    dugout_right.anchor_x = 0
    dugout_right.anchor_y = dugout_right.height

    header_height = 140

    data = None

    def refresh(dt):
        nonlocal data
        print(f"refreshing game: {data}")
        if remote.poll():
            cmd, d = remote.recv()
            print(f"Received command {cmd} with data {d}")
            if cmd == 'update':
                data = d
        if data is None:
            return
        print("Drawing")
        window.clear()
        pitch_img.blit(width / 2, height - header_height)
        dugout_left.blit(width / 2 - pitch_img.width / 2, height - header_height)
        dugout_right.blit(width / 2 + pitch_img.width / 2, height - header_height)
        main_batch.draw()

    pyglet.clock.schedule_interval(refresh, 1 / 120.0)
    pyglet.app.run()


class BotBowlGUI:

    def __init__(self, config):
        self.config = config
        self.remote, work_remote = Pipe()
        self.process = Process(target=worker, args=(work_remote, self.remote, config))
        self.process.daemon = True  # If the main process crashes, we should not cause things to hang
        self.process.start()

    def update(self, game):
        print("Updating")
        self.remote.send(('update', game))

    def close(self):
        self.process.terminate()


if __name__ == '__main__':
    gui1 = BotBowlGUI(None)
    gui2 = BotBowlGUI(None)
    for i in range(1000):
        gui1.update(i)
        gui2.update(i)
        time.sleep(1)
