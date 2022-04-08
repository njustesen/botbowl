import pyglet
import time
from multiprocessing import Process, Pipe


def worker(remote, parent_remote, config):
    parent_remote.close()
    width = 800
    height = 600
    window = pyglet.window.Window(width, height)
    main_batch = pyglet.graphics.Batch()

    text = pyglet.text.Label(text="", x=100, y=100, batch=main_batch)

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
        bg = pyglet.shapes.Rectangle(x=0, y=0, width=width, height=height, color=(0, 0, 0), batch=main_batch)
        circle = pyglet.shapes.Circle(100+20*data, 100, 100, color=(50, 225, 30), batch=main_batch)
        text.text = str(data)
        text.x = 100 + data * 20
        square = pyglet.shapes.Rectangle(200, 200, 200, 200, color=(55, 55, 255), batch=main_batch)
        rectangle = pyglet.shapes.Rectangle(250, 300, 400, 200, color=(255, 22, 20), batch=main_batch)
        rectangle.opacity = 128
        rectangle.rotation = 33
        line = pyglet.shapes.Line(100, 100, 100, 200, width=19, batch=main_batch)
        line2 = pyglet.shapes.Line(150, 150, 444, 111, width=4, color=(200, 20, 20), batch=main_batch)
        star = pyglet.shapes.Star(800, 400, 60, 40, num_spikes=20, color=(255, 255, 0), batch=main_batch)
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
