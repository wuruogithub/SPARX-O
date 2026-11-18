"""Render the two README figures with Pillow. Run from any directory."""
import cmath
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"
INK, MUTED, LINE = "#F5EFE6", "#C9BDCC", "#63576B"
ORCHID, COPPER, SLATE_BLUE = "#DDAED2", "#E3AE75", "#91BCDC"
PAPER = "#251F2C"


def font(size, bold=False):
    for name in (("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else
                 ("segoeui.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(d, xy, value, size=22, color=INK, bold=False):
    d.text(xy, value, font=font(size, bold), fill=color)


def canvas(kicker, title):
    im = Image.new("RGB", (1024, 440), PAPER)
    d = ImageDraw.Draw(im)
    label(d, (36, 16), kicker, 18, ORCHID, True)
    label(d, (34, 46), title, 34, INK, True)
    d.line((36, 103, 988, 103), fill=LINE, width=2)
    return im, d


def save(frames, durations, name):
    frames[0].save(ASSETS / name, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=True)


def shapes():
    steps = [
        ("Acquisitions", "1 × 4 × 10 × 32 × 32", "Ten spectral bands per date", 4, 2),
        ("Frame embedding + date encoding", "1 × 4 × 160 × 32 × 32", "Each acquisition gains 160 local features.", 4, 6),
        ("TI-SSM updates each pixel sequence", "1 × 4 × 160 × 32 × 32", "Two updates preserve the four date slots.", 4, 6),
        ("Grouped temporal selection", "1 × 160 × 32 × 32", "Eight groups pool valid dates into one map.", 1, 6),
        ("Time-guided spatial prediction", "1 × 20 × 32 × 32", "The decoder produces one score per class.", 1, 3),
        ("Pixelwise class selection", "1 × 32 × 32", "Argmax gives one class ID per pixel.", 1, 1),
    ]
    frames, durations = [], []
    for i, (title, shape, note, dates, depth) in enumerate(steps):
        # Bring all date stacks to the same location before revealing the pooled map.
        transitions = 17 if i == 3 else 1
        for frame in range(transitions):
            t = frame / (transitions - 1) if transitions > 1 else 1
            pooling = i == 3 and frame < transitions - 1
            im, d = canvas("SPARX   /   TENSOR SHAPES", title)
            for j in range(4 if pooling else dates):
                x = 55 + j * 107 if dates > 1 or pooling else 210
                if pooling:
                    x = round(x + (210 - x) * t)
                color = SLATE_BLUE if dates > 1 or pooling else COPPER
                for k in range(depth):
                    y = 160 - k * 5
                    xx = x + k * 3
                    d.rectangle((xx, y, xx+68, y+72), fill="#393748", outline=color, width=2)
                for n in (1, 2, 3):
                    d.line((xx+n*17,y,xx+n*17,y+72),fill="#84798E")
                    d.line((xx,y+n*18,xx+68,y+n*18),fill="#84798E")
                if not pooling or frame == 0:
                    label(d, (x, 245), f"date {j+1}" if dates > 1 or pooling else "one map", 18, MUTED)
            label(d, (515, 143), "TENSOR SHAPE", 18, ORCHID, True)
            label(d, (512, 182), steps[2][1] if pooling else shape, 32, INK, True)
            axes = "Batch · dates · channels · height · width" if dates > 1 or pooling else (
                "Batch · channels · height · width" if i < 5 else "Batch · height · width")
            label(d, (515, 244), axes, 20, MUTED)
            label(d, (38, 320), note, 25)
            for j in range(len(steps)):
                x = 38 + j*160
                d.rounded_rectangle((x,397,x+147,405),radius=4,fill=ORCHID if j<=i else LINE)
            frames.append(im)
            durations.append(75 if pooling else 1900)
    save(frames, durations, "sparx-tensor-dimensions.gif")


def arrow(d, start, end, color):
    d.line((start,end),fill=color,width=4)
    dx,dy=end[0]-start[0],end[1]-start[1]
    if math.hypot(dx,dy)>6:
        a=math.atan2(dy,dx)
        d.polygon([end]+[(end[0]-12*math.cos(a+s),end[1]-12*math.sin(a+s))
                         for s in (-0.48,0.48)],fill=color)
    else:
        d.ellipse((end[0]-4,end[1]-4,end[0]+4,end[1]+4),fill=color)


def states():
    days = [10,40,60,130]
    timeline = [62,218,322,686]
    frames, durations = [], []
    center=(285,277)
    def point(z):
        return (center[0]+330*z.real, center[1]-330*z.imag)
    def scene(index,title,history,propagated,new,lines):
        im,d=canvas("TI-SSM   /   ONE COMPLEX STATE CHANNEL",title)
        d.line((62,141,686,141),fill=LINE,width=3)
        for j,x in enumerate(timeline):
            col=COPPER if j==2 else ORCHID
            d.ellipse((x-6,135,x+6,147),fill=col if j<=index else LINE)
            label(d,(x-20,151),str(days[j]),17,col if j==index else MUTED)
        label(d,(750,126),"Acquisition day",19,MUTED)
        d.line((100,277,460,277),fill=LINE,width=2)
        d.line((285,195,285,356),fill=LINE,width=2)
        label(d,(456,269),"Re",16,MUTED)
        label(d,(292,187),"Im",16,MUTED)
        arrow(d,point(0),point(history),SLATE_BLUE)
        arrow(d,point(history),point(propagated),COPPER)
        arrow(d,point(propagated),point(new),ORCHID)
        for j,(s,col) in enumerate(lines):
            label(d,(520,205+j*44),s,22,col,j==0)
        for j,(s,c) in enumerate([("Previous state",SLATE_BLUE),("Decay + rotation",COPPER),("Input increment",ORCHID)]):
            x=45+j*327
            d.line((x,397,x+25,397),fill=c,width=4)
            label(d,(x+34,381),s,20,c)
        return im
    z=0j
    prev_day=None
    prev_v=0.
    for i,(day,v) in enumerate(zip(days,[2.,2.5,None,1.6])):
        if i==0 or v is None:
            title="Save the first valid input" if i==0 else "Skip the masked observation"
            lines=[("Day 10 · first valid date",INK),("Elapsed time = 0 days",MUTED),("State starts at zero.",MUTED)] if i==0 else [
                ("Day 60 · masked",COPPER),("Keep the state and input.",MUTED),("Day 40 stays the last valid date.",MUTED)]
            frames.append(scene(i,title,z,z,z,lines))
            durations.append(2200)
            if i==0:
                prev_day,prev_v=day,v
            continue
        gap=day-prev_day
        tau=gap/365.2425
        a=cmath.exp(complex(-2.5,2*math.pi)*tau)
        p=a*z
        increment=tau*(a*prev_v+v)/2
        for step in range(21):
            f=step/20
            moved=cmath.exp(complex(-2.5,2*math.pi)*tau*min(2*f,1))*z
            new=moved+max(2*f-1,0)*increment
            frames.append(scene(i,"Advance to the next valid acquisition",z,moved,new,[
                (f"Day {prev_day} → {day} · {gap} days",INK),
                ("Propagate the previous state." if f<0.5 else "Add both endpoint inputs.",COPPER if f<0.5 else ORCHID),
                (f"Retention {abs(a):.2f}   ·   rotation {360*tau:.0f}°",MUTED)]))
            durations.append(1400 if step==20 else 80)
        z,prev_day,prev_v=p+increment,day,v
    save(frames,durations,"ti-ssm-update.gif")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    shapes()
    states()
