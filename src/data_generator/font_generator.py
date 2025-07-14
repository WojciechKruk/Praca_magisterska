import numpy as np
from PIL import Image, ImageDraw
from font_BIG import DIGIT_GRIDS
from font_SMALL import S_DIGIT_GRIDS

# importy stałych
from src.config import (
    DEFAULT_FONT_TYPE, DEFAULT_TEXT, DEFAULT_TEXT_COLOR,
    BIG_IMG_SIZE, BIG_DOT_RADIUS, BIG_CHAR_WIDTH, BIG_CHAR_HEIGHT, BIG_X_GRID_SPACING,
    BIG_Y_GRID_SPACING, BIG_CHAR_SPACING, BIG_MAX_ERROR, BIG_SCALE_FACTOR,
    SMALL_IMG_SIZE, SMALL_DOT_RADIUS, SMALL_CHAR_WIDTH, SMALL_CHAR_HEIGHT, SMALL_X_GRID_SPACING,
    SMALL_Y_GRID_SPACING, SMALL_CHAR_SPACING, SMALL_MAX_ERROR, SMALL_SCALE_FACTOR
)

FONT = DEFAULT_FONT_TYPE
TEXT = DEFAULT_TEXT
TEXT_COLOR = DEFAULT_TEXT_COLOR

# ======BIG FONT SETTINGS======
IMG_SIZE = BIG_IMG_SIZE
DOT_RADIUS = BIG_DOT_RADIUS
CHAR_WIDTH = BIG_CHAR_WIDTH
CHAR_HEIGHT = BIG_CHAR_HEIGHT
X_GRID_SPACING = BIG_X_GRID_SPACING  # odstęp między punktami w pikselach w osi X
Y_GRID_SPACING = BIG_Y_GRID_SPACING  # odstęp między punktami w pikselach w osi Y
CHAR_SPACING = BIG_CHAR_SPACING
MAX_ERROR = BIG_MAX_ERROR
SCALE_FACTOR = BIG_SCALE_FACTOR

# ======SMALL FONT SETTINGS======
S_IMG_SIZE = SMALL_IMG_SIZE
S_DOT_RADIUS = SMALL_DOT_RADIUS
S_CHAR_WIDTH = SMALL_CHAR_WIDTH
S_CHAR_HEIGHT = SMALL_CHAR_HEIGHT
S_X_GRID_SPACING = SMALL_X_GRID_SPACING  # odstęp między punktami w pikselach w osi X
S_Y_GRID_SPACING = SMALL_Y_GRID_SPACING  # odstęp między punktami w pikselach w osi Y
S_CHAR_SPACING = SMALL_CHAR_SPACING
S_MAX_ERROR = SMALL_MAX_ERROR
S_SCALE_FACTOR = SMALL_SCALE_FACTOR


def draw_character(draw, char, top_left, text_color, x_grid_spacing, y_grid_spacing, dot_radius, digit_grids,
                   max_error):
    grid = digit_grids.get(char)
    if not grid:
        raise ValueError(f"Brak zdefiniowanego znaku: '{char}'")

    x0, y0 = top_left
    line_error = np.array([0.0, 0.0])

    for j, row in enumerate(grid):
        line_error[:] = 0.0
        for i, val in enumerate(row):
            if val == '1':
                # offset = np.random.uniform(-max_error, max_error, size=2)
                offset = np.array([sample_linear_offset(max_error), sample_linear_offset(max_error)])
                line_error += offset

                cx = x0 + i * x_grid_spacing + line_error[0]
                cy = y0 + j * y_grid_spacing + line_error[1]

                draw.ellipse(
                    (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
                    fill=text_color
                    # fill=(0, 0, 0)
                )
            else:
                line_error += np.random.uniform(-max_error, max_error, size=2)  # błąd się propaguje


def sample_linear_offset(max_error, debug=False):
    # Losujemy punkt pod krzywą trójkąta
    if debug:
        print(f"max_error: {max_error}")
    while True:
        x = np.random.uniform(-max_error, max_error)
        y = np.random.uniform(0, 1)
        if debug:
            print("----------------------------")
            print(f"x: {x}\ny: {y} <=? {1 - abs(x) / max_error}")
        if y <= 1 - abs(x) / max_error:
            if debug:
                print(f"wybrano {x}")
            return x


def prepare_canvas(text, img_size, char_width, char_height, x_grig_spacing, y_grig_spacing, char_spacing):
    # canvas = Image.new("RGB", img_size, (255, 255, 255))  # do debugowania
    canvas = Image.new("RGBA", img_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    x = (img_size[0] / 2 - ((len(text) * (char_width * x_grig_spacing + char_spacing) - char_spacing) / 2)
         + (x_grig_spacing / 2))
    y = int((img_size[1] / 2) - ((char_height * y_grig_spacing) / 2) + ((y_grig_spacing) / 2))

    return canvas, draw, x, y


def get_scaled_params(scale_factor):
    return {
        "DOT_RADIUS": DOT_RADIUS * scale_factor,
        "CHAR_WIDTH": CHAR_WIDTH,
        "CHAR_HEIGHT": CHAR_HEIGHT,
        "X_GRID_SPACING": X_GRID_SPACING * scale_factor,
        "Y_GRID_SPACING": Y_GRID_SPACING * scale_factor,
        "CHAR_SPACING": CHAR_SPACING * scale_factor,
        "MAX_ERROR": MAX_ERROR * scale_factor,

        "S_DOT_RADIUS": S_DOT_RADIUS * scale_factor,
        "S_CHAR_WIDTH": S_CHAR_WIDTH,
        "S_CHAR_HEIGHT": S_CHAR_HEIGHT,
        "S_X_GRID_SPACING": S_X_GRID_SPACING * scale_factor,
        "S_Y_GRID_SPACING": S_Y_GRID_SPACING * scale_factor,
        "S_CHAR_SPACING": S_CHAR_SPACING * scale_factor,
        "S_MAX_ERROR": S_MAX_ERROR * scale_factor,
    }


def render_inkjet_text(text, font, text_color, img_size, scale_factor, debug=False):
    params = get_scaled_params(scale_factor)

    if font == "BIG":
        canvas, draw, x, y = prepare_canvas(text, img_size, 15, 7,
                                            params["X_GRID_SPACING"], params["Y_GRID_SPACING"], params["CHAR_SPACING"])
    else:
        canvas, draw, x, y = prepare_canvas(text, img_size, 5, 7,
                                            params["S_X_GRID_SPACING"], params["S_Y_GRID_SPACING"], params["S_CHAR_SPACING"])

    for char in text:
        if font == "BIG":
            draw_character(draw, char, (x, y), text_color,
                           params["X_GRID_SPACING"], params["Y_GRID_SPACING"], params["DOT_RADIUS"],
                           DIGIT_GRIDS, max_error=params["MAX_ERROR"])
            x += 15 * params["X_GRID_SPACING"] + params["CHAR_SPACING"]
        elif font == "SMALL":
            draw_character(draw, char, (x, y), text_color,
                           params["S_X_GRID_SPACING"], params["S_Y_GRID_SPACING"], params["S_DOT_RADIUS"],
                           S_DIGIT_GRIDS, max_error=params["S_MAX_ERROR"])
            x += 5 * params["S_X_GRID_SPACING"] + params["S_CHAR_SPACING"]

    if debug:
        canvas.show()
    return canvas



# Przykładowe użycie
if __name__ == "__main__":
    img = render_inkjet_text(text=TEXT, font=FONT, text_color=TEXT_COLOR, img_size=IMG_SIZE, scale_factor=SCALE_FACTOR)
    img.show()
