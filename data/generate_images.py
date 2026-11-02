import os
import math
import numpy as np
from PIL import Image, ImageDraw

IMG_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Galaxy
def create_galaxy(path):
    img = Image.new("RGB", (256, 256), color=(8, 5, 20))
    draw = ImageDraw.Draw(img)
    # Background stars
    np.random.seed(42)
    for _ in range(120):
        x, y = np.random.randint(0, 256, 2)
        b = np.random.randint(180, 255)
        draw.point((x, y), fill=(b, b, b))
    # Spiral arms
    cx, cy = 128, 128
    for arm in [0, math.pi]:
        for theta in np.linspace(0, 4 * math.pi, 300):
            r = 8 * math.sqrt(theta) * 4.5
            x = cx + r * math.cos(theta + arm) + np.random.normal(0, 4)
            y = cy + r * math.sin(theta + arm) + np.random.normal(0, 4)
            if 0 <= x < 256 and 0 <= y < 256:
                draw.ellipse([x-2, y-2, x+2, y+2], fill=(160, 120, 240))
    # Glowing nucleus
    for r in range(25, 0, -2):
        alpha = int(255 * (1 - r / 25))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 240, 200))
    img.save(path)

# 2. Sunset Beach
def create_sunset(path):
    img = Image.new("RGB", (256, 256))
    draw = ImageDraw.Draw(img)
    # Sky gradient
    for y in range(160):
        r = int(255 - (y / 160) * 40)
        g = int(80 + (y / 160) * 100)
        b = int(20 + (y / 160) * 30)
        draw.line([(0, y), (256, y)], fill=(r, g, b))
    # Glowing Sun
    draw.ellipse([100, 80, 156, 136], fill=(255, 240, 100))
    # Ocean
    for y in range(160, 256):
        r = int(20 + (y - 160) * 0.4)
        g = int(50 + (y - 160) * 0.5)
        b = int(120 - (y - 160) * 0.6)
        draw.line([(0, y), (256, y)], fill=(r, g, b))
    # Waves
    for y in range(170, 250, 12):
        draw.arc([30, y, 226, y+8], start=0, end=180, fill=(240, 240, 255), width=2)
    img.save(path)

# 3. Pine Forest
def create_forest(path):
    img = Image.new("RGB", (256, 256), color=(180, 220, 250))
    draw = ImageDraw.Draw(img)
    # Green ground
    draw.rectangle([0, 180, 256, 256], fill=(40, 110, 40))
    # Pine trees
    tree_xs = [30, 80, 130, 180, 225, 55, 105, 160, 205]
    for x in tree_xs:
        # Trunk
        draw.rectangle([x-4, 150, x+4, 190], fill=(80, 45, 20))
        # Pine foliage layers
        for tier, y in enumerate([110, 130, 150]):
            w = 16 + tier * 10
            draw.polygon([(x, y - 25), (x - w, y + 10), (x + w, y + 10)], fill=(20, 75 + tier * 15, 30))
    img.save(path)

# 4. Snowy Mountain
def create_mountain(path):
    img = Image.new("RGB", (256, 256), color=(135, 195, 245))
    draw = ImageDraw.Draw(img)
    # Distant peaks
    draw.polygon([(10, 200), (90, 80), (170, 200)], fill=(120, 130, 150))
    draw.polygon([(90, 80), (130, 135), (90, 125)], fill=(255, 255, 255))
    # Main peak
    draw.polygon([(60, 240), (160, 50), (250, 240)], fill=(90, 100, 120))
    # Snow cap
    draw.polygon([(160, 50), (120, 115), (145, 105), (160, 120), (185, 100), (200, 115)], fill=(255, 255, 255))
    # Valley
    draw.rectangle([0, 220, 256, 256], fill=(220, 235, 245))
    img.save(path)

# 5. Circuit Board
def create_circuit(path):
    img = Image.new("RGB", (256, 256), color=(15, 30, 20))
    draw = ImageDraw.Draw(img)
    # Grid lines
    for i in range(20, 256, 30):
        draw.line([(i, 0), (i, 256)], fill=(25, 60, 35), width=1)
        draw.line([(0, i), (256, i)], fill=(25, 60, 35), width=1)
    # Central chip
    draw.rectangle([88, 88, 168, 168], fill=(30, 30, 35), outline=(0, 230, 120), width=2)
    draw.text((105, 120), "CPU", fill=(0, 255, 140))
    # Neon green bus traces
    traces = [
        [(88, 100), (30, 100), (30, 40)],
        [(88, 140), (40, 140), (40, 220)],
        [(168, 100), (220, 100), (220, 50)],
        [(168, 140), (210, 140), (210, 210)],
        [(128, 88), (128, 30)],
        [(128, 168), (128, 230)],
    ]
    for t in traces:
        draw.line(t, fill=(0, 255, 128), width=3)
        draw.ellipse([t[-1][0]-4, t[-1][1]-4, t[-1][0]+4, t[-1][1]+4], fill=(255, 215, 0))
    img.save(path)

# 6. Coffee Cup
def create_coffee(path):
    img = Image.new("RGB", (256, 256), color=(220, 200, 180))
    draw = ImageDraw.Draw(img)
    # Saucer
    draw.ellipse([48, 48, 208, 208], fill=(245, 245, 245), outline=(190, 180, 170), width=3)
    # Cup rim
    draw.ellipse([64, 64, 192, 192], fill=(255, 255, 255), outline=(180, 170, 160), width=2)
    # Coffee liquid
    draw.ellipse([76, 76, 180, 180], fill=(70, 40, 25))
    # Heart latte art
    draw.ellipse([100, 105, 135, 140], fill=(245, 230, 215))
    draw.ellipse([121, 105, 156, 140], fill=(245, 230, 215))
    draw.polygon([(103, 126), (153, 126), (128, 160)], fill=(245, 230, 215))
    img.save(path)

create_galaxy(os.path.join(IMG_DIR, "galaxy.png"))
create_sunset(os.path.join(IMG_DIR, "sunset_beach.png"))
create_forest(os.path.join(IMG_DIR, "pine_forest.png"))
create_mountain(os.path.join(IMG_DIR, "snowy_mountain.png"))
create_circuit(os.path.join(IMG_DIR, "cyber_circuit.png"))
create_coffee(os.path.join(IMG_DIR, "coffee_cup.png"))
print("Generated all 6 gallery images successfully.")
