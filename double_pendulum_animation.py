from generate_time_series import load_double_pendulum_time_series

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == "__main__":
    y = load_double_pendulum_time_series(friction=0.1)
    print(y.shape)

    L1, L2 = 1, 1
    dt = 3e-2

    x1 = L1*np.sin(y[:, 0])
    y1 = L1*np.cos(y[:, 0])

    x2 = L2*np.sin(y[:, 1]) + x1
    y2 = L2*np.cos(y[:, 1]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                                  interval=dt*1000, blit=True, init_func=init)
    plt.show()
