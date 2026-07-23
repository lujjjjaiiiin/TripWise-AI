"""Generate the TripWise boarding splash as one self-contained HTML string.

Everything is vector: the cabin, the window mouldings, the shades and the sky
are SVG shapes and filters, so the scene stays sharp at any resolution and the
whole thing costs a few kilobytes instead of a megabyte of base64 photo.

Clouds come from <feTurbulence> shaped by <feColorMatrix> and masked to a deck
below the horizon, which is what the view actually looks like at altitude:
clear sky on top, a bright cloud layer underneath.
"""

W, H = 1200, 620
WIN_W, WIN_H = 300, 430
WIN_Y = 78
WIN_X = [72, 450, 828]

# glass opening, relative to each window's top-left
GX, GY, GW, GH = 40, 46, 220, 326
GRX, GRY = 62, 50

CLOUD_LAYERS = [
    # id, baseFrequency, octaves, seed, matrix slope, matrix offset, opacity,
    # mask id, drift seconds, drift px
    ("cA", 0.009, 5, 11, -1.9, 1.18, 0.85, "mDeep", 26, 16),
    ("cB", 0.017, 6, 5, -2.5, 1.45, 1.00, "mDeck", 19, 26),
    ("cC", 0.034, 5, 29, -3.0, 1.72, 0.90, "mLow", 13, 38),
    ("cD", 0.013, 4, 41, -2.6, 1.30, 0.28, "mHigh", 34, 12),
]


def cloud_filters() -> str:
    out = []
    for cid, freq, octv, seed, slope, off, *_ in CLOUD_LAYERS:
        out.append(
            f'<filter id="{cid}" x="0" y="0" width="100%" height="100%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{freq}" '
            f'numOctaves="{octv}" seed="{seed}" stitchTiles="stitch"/>'
            f'<feColorMatrix type="matrix" values="'
            f'0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  {slope} 0 0 0 {off}"/>'
            f"</filter>"
        )
    return "".join(out)


def deck_masks() -> str:
    """Vertical ramps that place each cloud layer relative to the horizon."""
    ramps = {
        "mDeep": [(0.40, 0), (0.66, 1), (1.0, 1)],
        "mDeck": [(0.44, 0), (0.70, 1), (1.0, 1)],
        "mLow": [(0.58, 0), (0.84, 1), (1.0, 1)],
        "mHigh": [(0.0, 1), (0.30, 0), (1.0, 0)],
    }
    out = []
    for name, stops in ramps.items():
        s = "".join(
            f'<stop offset="{p * 100:.0f}%" stop-color="#fff" stop-opacity="{v}"/>'
            for p, v in stops
        )
        out.append(
            f'<linearGradient id="{name}g" x1="0" y1="0" x2="0" y2="1">{s}</linearGradient>'
            f'<mask id="{name}" maskUnits="userSpaceOnUse" x="-40" y="0" '
            f'width="{GW + 80}" height="{GH}">'
            f'<rect x="-40" y="0" width="{GW + 80}" height="{GH}" fill="url(#{name}g)"/>'
            f"</mask>"
        )
    return "".join(out)


def window_svg(i: int, x: int) -> str:
    """One window: moulded surround, recessed well, glass, sky, shade."""
    gid = f"glass{i}"
    seed_shift = i * 60
    delay = 1.0 + i * 0.12

    clouds = []
    for n, (cid, freq, octv, seed, slope, off, op, mask, dur, dist) in enumerate(CLOUD_LAYERS):
        # a per-window filter so each window shows a different piece of sky
        wid = f"{cid}_{i}"
        clouds.append(
            f'<g class="drift" style="animation-duration:{dur + i * 2}s;'
            f"animation-delay:-{n * 3 + i}s;--dx:{dist}px\">"
            f'<rect x="{GX - 40}" y="{GY}" width="{GW + 80}" height="{GH}" '
            f'filter="url(#{wid})" mask="url(#{mask})" opacity="{op}" '
            f'transform="translate(0,0)"/>'
            f"</g>"
        )

    return f"""
  <g transform="translate({x},{WIN_Y})">
    <rect x="0" y="0" width="{WIN_W}" height="{WIN_H}" rx="96" ry="78" fill="url(#bezel)"
          filter="url(#winShadow)"/>
    <rect x="0" y="0" width="{WIN_W}" height="{WIN_H}" rx="96" ry="78" fill="none"
          stroke="#aab3bd" stroke-width="1.1" opacity=".5"/>
    <rect x="24" y="24" width="{WIN_W - 48}" height="{WIN_H - 48}" rx="76" ry="60" fill="url(#well)"/>
    <rect x="32" y="34" width="{WIN_W - 64}" height="{WIN_H - 68}" rx="68" ry="54"
          fill="#5c6672" opacity=".5"/>

    <clipPath id="{gid}">
      <rect x="{GX}" y="{GY}" width="{GW}" height="{GH}" rx="{GRX}" ry="{GRY}"/>
    </clipPath>

    <g clip-path="url(#{gid})">
      <rect x="{GX}" y="{GY}" width="{GW}" height="{GH}" fill="url(#sky)"/>
      <g transform="translate({GX},{GY})">{''.join(clouds)}</g>
      <ellipse cx="{GX + GW * 0.74}" cy="{GY + GH * 0.34}" rx="{GW * 0.7}" ry="{GH * 0.42}"
               fill="url(#sun)" opacity=".55"/>

      <!-- shade: starts closed, glides up out of the opening -->
      <g class="shade" style="animation-delay:{delay}s">
        <rect x="{GX - 4}" y="{GY - 2}" width="{GW + 8}" height="{GH + 4}" fill="url(#shade)"/>
        <rect x="{GX - 4}" y="{GY + GH - 40}" width="{GW + 8}" height="42" fill="url(#lipG)"/>
        <rect x="{GX + GW / 2 - 30}" y="{GY + GH - 26}" width="60" height="9" rx="4.5"
              fill="url(#gripG)"/>
      </g>

      <!-- glass depth + reflection, always above the shade -->
      <rect x="{GX}" y="{GY}" width="{GW}" height="26" fill="#33404f" opacity=".34"/>
      <rect x="{GX}" y="{GY}" width="13" height="{GH}" fill="#33404f" opacity=".2"/>
      <rect x="{GX + GW - 13}" y="{GY}" width="13" height="{GH}" fill="#33404f" opacity=".2"/>
      <path d="M{GX} {GY + 250} L{GX + 150} {GY} L{GX + 210} {GY} L{GX} {GY + GH} Z"
            fill="#fff" opacity=".08"/>
    </g>

    <path d="M{GX + 18} {GY + 6} C{GX + 60} {GY - 26} {GX + GW - 60} {GY - 26} {GX + GW - 18} {GY + 6}
             C{GX + GW - 60} {GY - 12} {GX + 60} {GY - 12} {GX + 18} {GY + 6} Z" fill="url(#lipHi)"/>
  </g>"""


def build() -> str:
    windows = "".join(window_svg(i, x) for i, x in enumerate(WIN_X))

    # per-window copies of each cloud filter, re-seeded so no two windows match
    per_window = []
    for i in range(3):
        for cid, freq, octv, seed, slope, off, *_ in CLOUD_LAYERS:
            per_window.append(
                f'<filter id="{cid}_{i}" x="0" y="0" width="100%" height="100%">'
                f'<feTurbulence type="fractalNoise" baseFrequency="{freq}" '
                f'numOctaves="{octv}" seed="{seed + i * 60}" stitchTiles="stitch"/>'
                f'<feColorMatrix type="matrix" values="'
                f'0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  {slope} 0 0 0 {off}"/>'
                f"</filter>"
            )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%;overflow:hidden}}
body{{
  display:grid;place-items:center;
  background:linear-gradient(180deg,#f1f4f7 0%,#e3e9ef 42%,#ced6df 100%);
  animation:out .8s ease 4.5s forwards;
}}
@keyframes out{{to{{opacity:0}}}}

.scene{{width:min(96vw,1240px);aspect-ratio:{W}/{H};
  animation:scenein 1.2s cubic-bezier(.2,.7,.3,1) both}}
@keyframes scenein{{from{{opacity:0;transform:scale(1.05)}}to{{opacity:1;transform:none}}}}
svg{{width:100%;height:100%;display:block;
  filter:drop-shadow(0 30px 64px rgba(66,80,96,.22))}}

/* shades: hold closed, then glide up out of the opening */
.shade{{transform:translateY(0);animation:up 2.2s cubic-bezier(.72,0,.18,1) forwards}}
@keyframes up{{to{{transform:translateY(-{GH + 60}px)}}}}

/* clouds drift sideways at different speeds for parallax */
.drift{{animation-name:dr;animation-timing-function:ease-in-out;
  animation-iteration-count:infinite;animation-direction:alternate}}
@keyframes dr{{from{{transform:translateX(calc(var(--dx) * -1))}}
              to{{transform:translateX(var(--dx))}}}}

.mark{{position:absolute;left:0;right:0;bottom:6%;text-align:center;opacity:0;
  animation:mk 1s cubic-bezier(.2,.7,.3,1) 2.8s both}}
.mark .t{{font-family:'Outfit',sans-serif;font-weight:800;
  font-size:clamp(1.5rem,3.6vw,2.45rem);letter-spacing:-.005em;
  background:linear-gradient(96deg,#0EA5E9,#2563EB);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  color:#2563EB;
  filter:drop-shadow(0 0 22px rgba(14,165,233,.45)) drop-shadow(0 2px 3px rgba(255,255,255,.95))}}
.mark .s{{font-family:'Inter',sans-serif;margin-top:.5rem;color:#5A6B7E;font-weight:500;
  font-size:clamp(.62rem,1.35vw,.8rem);letter-spacing:.34em;text-transform:uppercase;
  text-shadow:0 1px 2px rgba(255,255,255,.9)}}
@keyframes mk{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}

@media (prefers-reduced-motion:reduce){{
  .shade{{animation-duration:.01s}}
  .drift{{animation:none}}
  body{{animation:out .5s linear 2s forwards}}
}}
</style></head><body>
<div class="scene">
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
<defs>
  <linearGradient id="cabin" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#f1f4f7"/><stop offset=".42" stop-color="#e3e9ef"/>
    <stop offset="1" stop-color="#ced6df"/>
  </linearGradient>
  <radialGradient id="vign" cx="50%" cy="42%" r="72%">
    <stop offset="55%" stop-color="#000" stop-opacity="0"/>
    <stop offset="100%" stop-color="#48535f" stop-opacity=".16"/>
  </radialGradient>
  <filter id="winShadow" x="-25%" y="-25%" width="150%" height="150%">
    <feDropShadow dx="0" dy="12" stdDeviation="15" flood-color="#54626f" flood-opacity=".33"/>
  </filter>
  <linearGradient id="bezel" x1=".25" y1="0" x2=".75" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset=".5" stop-color="#f4f6f8"/>
    <stop offset="1" stop-color="#dbe1e7"/>
  </linearGradient>
  <linearGradient id="well" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6f7883"/><stop offset=".28" stop-color="#98a1ab"/>
    <stop offset=".75" stop-color="#c9d0d8"/><stop offset="1" stop-color="#eef1f4"/>
  </linearGradient>
  <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#1a4a8c"/><stop offset=".34" stop-color="#4886c4"/>
    <stop offset=".56" stop-color="#9ec8e8"/><stop offset=".72" stop-color="#d9eaf6"/>
    <stop offset="1" stop-color="#c2d9ec"/>
  </linearGradient>
  <radialGradient id="sun">
    <stop offset="0" stop-color="#fff6dc" stop-opacity=".85"/>
    <stop offset="55%" stop-color="#ffe9b8" stop-opacity=".22"/>
    <stop offset="100%" stop-color="#ffe9b8" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="shade" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#d9dee4"/><stop offset=".08" stop-color="#f2f4f7"/>
    <stop offset=".5" stop-color="#fafbfc"/><stop offset=".92" stop-color="#eceff3"/>
    <stop offset="1" stop-color="#d3d9e0"/>
  </linearGradient>
  <linearGradient id="lipG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#d3d9e0"/><stop offset="1" stop-color="#b3bbc5"/>
  </linearGradient>
  <linearGradient id="gripG" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#98a0aa"/><stop offset="1" stop-color="#727a85"/>
  </linearGradient>
  <linearGradient id="lipHi" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff" stop-opacity=".92"/>
    <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </linearGradient>
  {deck_masks()}
  {''.join(per_window)}
</defs>
  <rect width="{W}" height="{H}" fill="url(#cabin)"/>
  <rect x="0" y="{H*0.055:.0f}" width="{W}" height="2" fill="#c3cbd4" opacity=".55"/>
  <rect x="0" y="{H*0.055+3:.0f}" width="{W}" height="1" fill="#ffffff" opacity=".9"/>
  <rect x="0" y="{H*0.905:.0f}" width="{W}" height="2" fill="#c3cbd4" opacity=".5"/>
  <rect x="0" y="{H*0.905+3:.0f}" width="{W}" height="1" fill="#ffffff" opacity=".85"/>
{windows}
  <rect width="{W}" height="{H}" fill="url(#vign)" pointer-events="none"/>
</svg>
</div>
<div class="mark">
  <div class="t">TripWise AI</div>
  <div class="s">Plan smarter &middot; Travel further</div>
</div>
</body></html>"""


