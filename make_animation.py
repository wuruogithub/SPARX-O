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
    stages = [
        ("Sentinel-2 observations", "1 × 4 × 10 × 32 × 32"),
        ("Embed each observation and its date", "1 × 4 × 160 × 32 × 32"),
        ("TI-SSM updates the valid dates", "1 × 4 × 160 × 32 × 32"),
        ("Score the dates for each group", "One pixel  ·  learned projection from 160 features to 8 scores"),
        ("Pool dates within each group", "Weighted sum across valid dates  →  20 channels per group"),
        ("Concatenate along the channel axis", "8 pooled groups × 20 channels  →  160 channels"),
        ("Time-guided spatial prediction", "1 × 20 × 32 × 32"),
        ("Choose the highest-scoring class", "1 × 32 × 32"),
    ]
    frames, durations = [], []
    starts = [144, 390, 636, 882]
    days = [10, 40, 60, 130]

    def card(d, x, y, value, color, half=101):
        x, y = round(x), round(y)
        d.rounded_rectangle((x-half,y-22,x+half,y+22),radius=8,
                            fill="#393342",outline=color,width=2)
        d.text((x,y),value,font=font(21,True),fill=color,anchor="mm")

    def tile(d, x, y, depth, color):
        for k in range(depth):
            left, top = round(x-37+k*3), round(y-k*3)
            d.rectangle((left,top,left+74,top+55),fill="#393748",outline=color,width=2)
        for n in range(1,4):
            d.line((left+n*18,top,left+n*18,top+55),fill=LINE)
            d.line((left,top+n*14,left+74,top+n*14),fill=LINE)

    for stage,(title,shape) in enumerate(stages):
        for tick in range(19):
            t=tick/18
            im,d=canvas("SPARX   /   TENSOR FLOW",title)
            label(d,(38,116),shape,24,COPPER,True)
            if stage<=2:
                for j,x in enumerate(starts):
                    color=MUTED if j==2 else SLATE_BLUE
                    label(d,(x-60,164),f"Day {days[j]}" + (" · masked" if j==2 else ""),18,color)
                    depth=2 if stage==0 else 6
                    if stage==1:
                        depth=2+round(4*t)
                    offset=(1-t)*45 if stage==0 else 0
                    tile(d,x,214+offset,depth,color)
                    if stage==2 and j!=2:
                        # The valid-date scan bypasses the masked slot.
                        end=starts[0]+(starts[-1]-starts[0])*t
                        d.line((starts[0],286,end,286),fill=ORCHID,width=3)
                        if end>=x:
                            d.ellipse((x-5,281,x+5,291),fill=ORCHID)
            elif 3 <= stage <= 5:
                # Dates run left to right; channel groups retain their vertical order.
                cols = [224, 405, 586, 767]
                top, pitch = 201, 26
                pooled_x = 925
                for j,x in enumerate(cols):
                    if stage != 5:
                        d.text((x,171),["Day 10","Day 40","Day 60 · masked","Day 130"][j],
                               anchor="mm",font=font(18),fill=MUTED if j==2 else SLATE_BLUE)
                for g in range(8):
                    y=top+pitch*g
                    d.text((72,y),f"G{g+1}",anchor="mm",font=font(18,True),fill=ORCHID)
                    if stage==5:
                        # Stack the eight vectors in this order into one channel vector.
                        yy=y+(246+g*14-y)*t
                        x=pooled_x+(512-pooled_x)*t
                        d.rounded_rectangle((x-75,yy-6,x+75,yy+6),radius=3,
                                            fill=ORCHID if g%2==0 else SLATE_BLUE)
                        if g==7:
                            label(d,(max(620,x-300),385),"160 channels" if t==1 else "8 groups of 20",19,COPPER)
                        continue
                    for j,x in enumerate(cols):
                        col=MUTED if j==2 else SLATE_BLUE
                        if stage==3:
                            if t<0.42:
                                width=round(63*t/0.42)
                                d.line((x-width,y,x+width,y),fill=col,width=2)
                            else:
                                d.rounded_rectangle((x-66,y-11,x+66,y+11),radius=5,fill="#393342",outline=col)
                                scores=["0","ln 2","masked","ln 7"] if g==0 else [
                                    f"s({g+1},1)",f"s({g+1},2)","masked",f"s({g+1},4)"]
                                d.text((x,y),scores[j],anchor="mm",font=font(16),fill=col)
                        elif j!=2:
                            xx=x+(pooled_x-x)*t
                            d.line((x,y,xx,y),fill=LINE,width=1)
                            d.ellipse((xx-6,y-6,xx+6,y+6),fill=col)
                    if stage==4 and t>0.85:
                        d.rounded_rectangle((pooled_x-28,y-10,pooled_x+28,y+10),
                                            radius=4,fill=COPPER)
                        d.text((pooled_x,y),"20",anchor="mm",font=font(15,True),fill=PAPER)
                if stage==3:
                    label(d,(39,402),"One score per valid date and group. G1 shows illustrative values.",19,MUTED)
                if stage==4:
                    label(d,(39,402),"G1 example: 0.10 × 2 + 0.20 × 4 + 0.70 × 6 = 5.20",19,MUTED)
            else:
                tile(d,230+(170*t),230,6-round((3 if stage==6 else 2)*t) if stage==6 else 3-round(2*t),COPPER)
                card(d,740,250,"20 class scores" if stage==6 else "1 class ID / pixel",ORCHID)
                d.line((453,250,453+175*t,250),fill=ORCHID,width=3)
            frames.append(im)
            durations.append(1300 if tick==18 else 65)
    save(frames,durations,"sparx-tensor-dimensions.gif")


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
