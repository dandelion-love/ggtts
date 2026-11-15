# Renders the demo app's icon: 粵 (Cantonese) in Noto Sans HK (SIL Open Font License) on the app's light-blue accent.
#   python scripts/android_icon.py android/demo/src/main/res [preview.png]
# ICON_FONT names the font file (Noto Sans HK, from Google Fonts or the system). Needs Pillow.
import os, pathlib, sys
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(sys.argv[1])  # android/demo/src/main/res
FONT = os.environ.get("ICON_FONT", r"C:\Windows\Fonts\NotoSansHK-VF.ttf")
BG = (0xD3, 0xE3, 0xFD, 255)
FG = (0x0B, 0x1A, 0x2E, 255)
GLYPH = "粵"
DENS = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

def font(px):
    f = ImageFont.truetype(FONT, px)
    try:
        f.set_variation_by_name("Medium")
    except Exception:
        pass
    return f

def glyph(img, box_px, centre):
    d = ImageDraw.Draw(img)
    f = font(int(box_px))
    l, t, r, b = d.textbbox((0, 0), GLYPH, font=f)
    x = centre[0] - (l + r) / 2
    y = centre[1] - (t + b) / 2
    d.text((x, y), GLYPH, font=f, fill=FG)

for name, k in DENS.items():
    d = OUT / f"mipmap-{name}"
    d.mkdir(parents=True, exist_ok=True)
    # Adaptive foreground: 108 dp, the glyph inside the 66 dp safe zone.
    s = round(108 * k)
    fg = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    glyph(fg, 46 * k, (s / 2, s / 2))
    fg.save(d / "ic_launcher_foreground.png", optimize=True)
    # Legacy icon (before Android 8): a 48 dp circle.
    s = round(48 * k)
    big = 4 * s
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((4, 4, big - 4, big - 4), fill=BG)
    glyph(im, big * 0.52, (big / 2, big / 2))
    im.resize((s, s), Image.LANCZOS).save(d / "ic_launcher.png", optimize=True)

(OUT / "mipmap-anydpi-v26").mkdir(exist_ok=True)
(OUT / "mipmap-anydpi-v26/ic_launcher.xml").write_text("""<?xml version="1.0" encoding="utf-8"?>
<!-- The app icon: 粵 (Cantonese) in Noto Sans HK (SIL Open Font License), rendered by scripts/android_icon.py. -->
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@mipmap/ic_launcher_foreground" />
</adaptive-icon>
""", encoding="utf-8", newline="\n")
(OUT / "values/ic_launcher.xml").write_text("""<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="ic_launcher_background">#D3E3FD</color>
</resources>
""", encoding="utf-8", newline="\n")
# A large preview for the PR.
p = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
ImageDraw.Draw(p).ellipse((0, 0, 511, 511), fill=BG)
glyph(p, 270, (256, 256))
if len(sys.argv) > 2:
    p.save(sys.argv[2])
print("icons written")
