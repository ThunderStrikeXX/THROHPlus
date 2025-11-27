import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import textwrap, sys, os

# ----------------------------------------------------------
# Loader sicuro: ogni file diventa SEMPRE (ntime, nnodes)
# ----------------------------------------------------------
def safe_load_matrix(filename, fill_value=-1e9):
    def parse_line(line):
        return line.replace('-nan(ind)', str(fill_value)) \
                   .replace('nan', str(fill_value)) \
                   .replace('NaN', str(fill_value))
    with open(filename, 'r') as f:
        rows = []
        for line in f:
            line = parse_line(line).strip()
            if not line:
                continue
            parts = line.split()
            rows.append([float(x) for x in parts])
    return np.array(rows, dtype=float)

# ----------------------------------------------------------
# Input
# ----------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python plot_time.py t.txt y1.txt y2.txt ...")
    sys.exit(1)

t_file = sys.argv[1]
y_files = sys.argv[2:]

for f in [t_file] + y_files:
    if not os.path.isfile(f):
        print("Error: file not found ->", f)
        sys.exit(1)

# ----------------------------------------------------------
# Caricamento tempo (colonna singola)
# ----------------------------------------------------------
t = safe_load_matrix(t_file).flatten()
ntime = len(t)

# ----------------------------------------------------------
# Caricamento variabili (matrici)
# ----------------------------------------------------------
Y = []
for f in y_files:
    M = safe_load_matrix(f)
    if M.shape[0] != ntime:
        print("Dimension mismatch:", f, "ha", M.shape[0], "righe, ma il tempo ha", ntime)
        sys.exit(1)
    Y.append(M)

nnodes = Y[0].shape[1]

# ----------------------------------------------------------
# Variabili e unità
# ----------------------------------------------------------
names = [
    "Vapor velocity","Vapor pressure","Vapor temperature","Vapor density","Vapor alpha",
    "Liquid velocity","Liquid pressure","Liquid temperature","Liquid density","Liquid alpha",
    "Wall temperature"
]

units = [
    "[m/s]","[Pa]","[K]","[kg/m³]","[-]",
    "[m/s]","[Pa]","[K]","[kg/m³]","[-]",
    "[K]"
]

# ----------------------------------------------------------
# Y-limits robusti
# ----------------------------------------------------------
def robust_ylim(y):
    lo, hi = np.percentile(y, [1, 99])
    if lo == hi:
        lo, hi = np.min(y), np.max(y)
    m = 0.1*(hi-lo)
    return lo-m, hi+m

# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11,6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.70)

line, = ax.plot([], [], lw=2)
ax.grid(True)
ax.set_xlabel("Time [s]")

# ----------------------------------------------------------
# Slider per nodo
# ----------------------------------------------------------
ax_slider = plt.axes([0.15, 0.1, 0.55, 0.03])
slider = Slider(ax_slider, "Node", 0, nnodes-1, valinit=0, valstep=1)

# ----------------------------------------------------------
# Stato
# ----------------------------------------------------------
current_idx = 0
current_node = 0
paused = [False]
current_frame = [0]

def draw_series():
    y = Y[current_idx][:, current_node]
    line.set_data(t, y)
    ax.set_ylim(*robust_ylim(y))
    ax.set_xlim(t.min(), t.max())
    ax.set_title(f"{names[current_idx]} {units[current_idx]}")

draw_series()

# ----------------------------------------------------------
# Slider update
# ----------------------------------------------------------
def slider_update(val):
    global current_node
    current_node = int(slider.val)
    draw_series()
    fig.canvas.draw_idle()

slider.on_changed(slider_update)

# ----------------------------------------------------------
# Variable buttons
# ----------------------------------------------------------
buttons = []
n_vars = len(Y)
n_cols = 2
button_width = 0.12
button_height = 0.08
col_gap = 0.02
row_gap = 0.10
start_x = 0.73
start_y = 0.86

for i,name in enumerate(names[:n_vars]):
    col = i % n_cols
    row = i // n_cols
    label = "\n".join(textwrap.wrap(name, 15))
    x_pos = start_x + col*(button_width+col_gap)
    y_pos = start_y - row*row_gap
    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, label, hovercolor='0.975')
    btn.label.set_fontsize(8)
    buttons.append(btn)

def change_variable(j):
    global current_idx
    current_idx = j
    draw_series()
    fig.canvas.draw_idle()

for i,btn in enumerate(buttons):
    btn.on_clicked(lambda event,j=i: change_variable(j))

# ----------------------------------------------------------
# Play/pause/reset
# ----------------------------------------------------------
ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
btn_play = Button(ax_play, "Play")

ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
btn_pause = Button(ax_pause, "Pause")

ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset")

def pause(event):
    paused[0] = True

def play(event):
    paused[0] = False

def reset(event):
    paused[0] = True
    current_frame[0] = 0
    slider.set_val(0)

btn_pause.on_clicked(pause)
btn_play.on_clicked(play)
btn_reset.on_clicked(reset)

# ----------------------------------------------------------
# Animazione: aggiorna solo la y (il nodo lo decide lo slider)
# ----------------------------------------------------------
def update_auto(i):
    if not paused[0]:
        current_frame[0] = i
        draw_series()
    return line,

ani = FuncAnimation(fig, update_auto,
                    frames=range(ntime),
                    interval=10000/ntime,
                    blit=False, repeat=True)

plt.show()
