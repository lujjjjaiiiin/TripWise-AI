"""HTML component builders.

Every visible surface is assembled here and handed to Streamlit as markup, so
pages stay declarative and no markup is duplicated between them. Functions take
plain data and return strings; none of them touch Streamlit or pandas.
"""

from __future__ import annotations

from html import escape

# --------------------------------------------------------------------------- #
# icons — single source, stroked to match the design system
# --------------------------------------------------------------------------- #

ICONS = {
    "compass": '<circle cx="12" cy="12" r="9"/><path d="m15.6 8.4-2 5.2-5.2 2 2-5.2z"/>',
    "sparkle": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/><circle cx="12" cy="12" r="3"/>',
    "wallet": '<path d="M3 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M16 11h5v4h-5a2 2 0 0 1 0-4z"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.4M12 19.6V22M2 12h2.4M19.6 12H22M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M19.1 4.9l-1.7 1.7M6.6 17.4l-1.7 1.7"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3.5 9h17M3.5 15h17M12 3a15 15 0 0 1 0 18A15 15 0 0 1 12 3z"/>',
    "shield": '<path d="M12 3l7 3v6c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6z"/><path d="m9.2 12 2 2 3.6-4"/>',
    "plane": '<path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>',
    "pin": '<path d="M12 21s-7-5.6-7-11a7 7 0 1 1 14 0c0 5.4-7 11-7 11z"/><circle cx="12" cy="10" r="2.6"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="16" rx="2.5"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "tag": '<path d="M3 12.5V5a2 2 0 0 1 2-2h7.5L21 11.5 12.5 20z"/><circle cx="7.8" cy="7.8" r="1.4"/>',
    "bulb": '<path d="M9.2 17h5.6M10 21h4"/><path d="M12 3a6 6 0 0 1 3.6 10.8c-.5.4-.8 1-.8 1.6H9.2c0-.6-.3-1.2-.8-1.6A6 6 0 0 1 12 3z"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "layers": '<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5M3 18l9 5 9-5"/>',
}


def icon(name: str, cls: str = "") -> str:
    body = ICONS.get(name, ICONS["sparkle"])
    c = f' class="{cls}"' if cls else ""
    return f'<svg viewBox="0 0 24 24"{c}>{body}</svg>'


def _e(v) -> str:
    return escape(str(v), quote=True)


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #

NAV_ITEMS = [
    ("Home", "?view=home", False),
    ("Features", "?view=home#features", False),
    ("AI Planner", "?view=planner", False),
    ("About", "?view=home#about", False),
    ("Contact", "?view=home#contact", False),
]


def navbar(active: str = "home") -> str:
    links = ""
    for label, href, _ in NAV_ITEMS:
        key = "planner" if "planner" in href else "home"
        cls = " is-active" if (key == active and "#" not in href) else ""
        links += f'<a href="{href}" target="_top" class="{cls.strip()}">{label}</a>'
    return f"""
<div class="tw-nav">
  <a class="tw-nav__brand" href="?view=home" target="_top">
    <svg viewBox="0 0 24 24" fill="none" stroke="url(#navg)" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
      <defs><linearGradient id="navg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
      </linearGradient></defs>
      <path d="M2 13.4 21 4l-4.6 16.2-4.1-6.1z"/><path d="m12.3 14.1-3.4 6.9-1-5"/>
    </svg>
    <b>TripWise <span class="tw-grad">AI</span></b>
  </a>
  <nav class="tw-nav__links">
    {links}
    <a class="tw-btn tw-btn--sm" href="?view=planner" target="_top">Start planning</a>
  </nav>
</div>"""


def section_head(eyebrow: str, title: str, blurb: str = "", anchor: str = "") -> str:
    a = f' id="{anchor}"' if anchor else ""
    p = f"<p>{blurb}</p>" if blurb else ""
    return f"""
<div class="tw-section"{a}>
  <div class="tw-section__head tw-rise">
    <span class="tw-eyebrow">{eyebrow}</span>
    <h2>{title}</h2>{p}
  </div>"""


def close_section() -> str:
    return "</div>"


# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #

def hero(stats: list[tuple[str, str]]) -> str:
    stat_html = "".join(
        f'<div class="tw-stat"><b>{_e(v)}</b><span>{_e(lbl)}</span></div>'
        for v, lbl in stats
    )
    return f"""
<div class="tw-hero">
  <div class="tw-hero__plane tw-float">
    <svg viewBox="0 0 420 260" fill="none">
      <defs>
        <linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0EA5E9"/><stop offset="100%" stop-color="#2563EB"/>
        </linearGradient>
        <linearGradient id="trailg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#0EA5E9" stop-opacity="0"/>
          <stop offset="100%" stop-color="#2563EB" stop-opacity=".55"/>
        </linearGradient>
      </defs>
      <path d="M10 210 C120 200 210 150 290 70" stroke="url(#trailg)" stroke-width="3"
            stroke-dasharray="2 12" stroke-linecap="round"/>
      <g transform="translate(300,62) rotate(-38)">
        <path d="M46 0 L16 -7 L-6 -34 L-18 -34 L-9 -7 L-30 -7 L-40 -17 L-48 -17
                 L-42 -5 L-42 5 L-48 17 L-40 17 L-30 7 L-9 7 L-18 34 L-6 34 L16 7 Z"
              fill="url(#pg)"/>
        <path d="M46 0 L16 -7 L-9 -7 L-9 7 L16 7 Z" fill="#fff" opacity=".28"/>
      </g>
      <circle cx="300" cy="62" r="46" fill="url(#pg)" opacity=".1"/>
    </svg>
  </div>
  <div class="tw-hero__inner">
    <span class="tw-eyebrow tw-rise d1">Intelligent travel planning</span>
    <h1 class="tw-rise d2">Plan Smarter with<br/><span class="tw-grad">TripWise AI</span></h1>
    <p class="tw-lede tw-rise d3">
      Tell us how you like to travel and TripWise ranks real destinations against
      your answers — with the cost, the climate, the right season and the nearest
      airport worked out before you commit to anything.
    </p>
    <div class="tw-hero__cta tw-rise d4">
      <a class="tw-btn" href="?view=planner" target="_top">Start planning &nbsp;&rarr;</a>
      <a class="tw-btn tw-btn--ghost" href="#features" target="_top">See how it works</a>
    </div>
    <div class="tw-hero__stats tw-rise d5">{stat_html}</div>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# cards
# --------------------------------------------------------------------------- #

def feature_card(icon_name: str, title: str, body: str, delay: int = 1) -> str:
    return f"""
<div class="tw-card tw-feat tw-rise d{delay}">
  <div class="tw-feat__icon">{icon(icon_name)}</div>
  <h3>{_e(title)}</h3>
  <p>{_e(body)}</p>
</div>"""


def insight_card(title: str, body: str, delay: int = 1) -> str:
    return f"""
<div class="tw-card tw-insight tw-rise d{delay}">
  <div class="tw-insight__dot">{icon('bulb')}</div>
  <div><h4>{_e(title)}</h4><p>{_e(body)}</p></div>
</div>"""


def destination_card(d: dict, delay: int = 1) -> str:
    """`d` is the plain dict assembled by app.py, never a raw dataframe row."""
    facts = f"""
    <div class="tw-facts">
      <div><span>Budget</span><b>{_e(d['tier'])}</b></div>
      <div><span>Per day</span><b>${_e(d['daily'])}</b></div>
      <div><span>Climate</span><b>{_e(d['temp'])}&deg;C</b></div>
    </div>"""

    chips = "".join(f'<span class="tw-chip">{_e(s)}</span>' for s in d["style"])
    if d.get("region"):
        chips += f'<span class="tw-chip tw-chip--muted">{_e(d["region"])}</span>'

    rows = []
    if d.get("airport"):
        rows.append((("plane"), "Nearest airport", d["airport"]))
    else:
        rows.append((("plane"), "Nearest airport", "None recorded for this city"))
    rows.append((("calendar"), "Best season", d["season"]))
    rows.append((("sun"), "Weather", d["climate"]))
    if d.get("hotel"):
        rows.append((("tag"), "Suggested stay", d["hotel"]))
    if d.get("attractions"):
        rows.append((("pin"), "Nearby", ", ".join(d["attractions"])))
    rows.append((("bulb"), "Tip", d["tip"]))

    meta = "".join(
        f'<div class="tw-meta__row">{icon(ic)}<div><b>{_e(lbl)}</b> — {_e(val)}</div></div>'
        for ic, lbl, val in rows
    )

    return f"""
<div class="tw-card tw-dest tw-rise d{delay}">
  <div class="tw-dest__art">
    {d['art']}
    <div class="tw-dest__scrim"></div>
    <div class="tw-dest__match">{_e(d['match'])}% match</div>
    <div class="tw-dest__place">
      <h3>{_e(d['city'])}</h3><span>{_e(d['country'])}</span>
    </div>
  </div>
  <div class="tw-dest__body">
    {facts}
    <div class="tw-chips">{chips}</div>
    <div class="tw-why">{_e(d['why'])}</div>
    <div class="tw-meta">{meta}</div>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# static content blocks
# --------------------------------------------------------------------------- #

def about_block() -> str:
    return """
<div class="tw-grid tw-grid--2">
  <div class="tw-card tw-feat tw-rise d1">
    <div class="tw-feat__icon">%s</div>
    <h3>Recommendations you can trace</h3>
    <p>TripWise compares your answers against every destination in its catalogue
       across nine travel dimensions, climate, budget tier and accommodation
       standard, then ranks by how closely each one sits to what you described.
       Every card explains its own reasoning in plain language.</p>
  </div>
  <div class="tw-card tw-feat tw-rise d2">
    <div class="tw-feat__icon">%s</div>
    <h3>Grounded in real data</h3>
    <p>Destinations, airports, accommodation standards and climate averages come
       from a curated catalogue rather than generated text, so a recommendation
       always points at somewhere that exists — and tells you when a detail is
       missing instead of inventing one.</p>
  </div>
</div>""" % (icon("sparkle"), icon("shield"))


def contact_block() -> str:
    return """
<div class="tw-card tw-feat tw-rise d1" style="padding:2.4rem 2rem;">
  <div class="tw-grid tw-grid--2" style="align-items:center;">
    <div>
      <h3 style="font-size:1.5rem;margin-bottom:.7rem;">Bring TripWise to your travellers</h3>
      <p>Agencies, airlines and tourism boards can run TripWise against their own
         catalogue — your destinations, your inventory, your branding.</p>
    </div>
    <div style="display:flex;gap:.7rem;flex-wrap:wrap;justify-content:flex-end;">
      <a class="tw-btn" href="mailto:hello@tripwise.ai" target="_top">Talk to us</a>
      <a class="tw-btn tw-btn--ghost" href="?view=planner" target="_top">Try the planner</a>
    </div>
  </div>
</div>"""


def footer() -> str:
    return """
<div class="tw-foot">
  <div class="tw-foot__note">
    <b style="color:var(--ink);">TripWise AI</b><br/>
    Destination matching, cost estimates and climate guidance from a curated
    travel catalogue. Figures are planning estimates, not quotes — confirm
    prices and seasons before booking.
  </div>
  <div class="tw-foot__note" style="text-align:right;">
    <a href="?view=home" target="_top">Home</a> &nbsp;·&nbsp;
    <a href="?view=planner" target="_top">AI Planner</a> &nbsp;·&nbsp;
    <a href="mailto:hello@tripwise.ai">Contact</a>
  </div>
</div>"""
