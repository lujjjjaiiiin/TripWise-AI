# TripWise AI

Destination matching for travellers. Describe how you like to travel and
TripWise ranks real destinations against your answers, with the cost, climate,
season and nearest airport worked out for each one.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Two builds ship here and behave identically:

| Build | Use |
| --- | --- |
| `app.py` + `tripwise/` | Development. Modular, and the version that is maintained. |
| `single_file/app.py` | Deployment. The package inlined into one file. |

The single-file build exists because cloud deployments regularly lose the
package folder during upload. Edit the modules and regenerate it:

```bash
python build_single.py
```

## Architecture

```
app.py                    routing, caching, form handling, page assembly
tripwise/
    config.py             schema, labels, tunables — one place to retune
    errors.py             logging, safe() decorator, guard() boundaries
    loader.py             CSV loading, validation, coercion, health report
    airports.py           airport resolution (direct -> name -> distance)
    models.py             fitted StandardScaler + KMeans bundle
    recommender.py        preference vector, cosine ranking, diversification
    insights.py           cost, season, explanations, travel intelligence
    validation.py         user input validation and filter relaxation
    theme.py              design tokens and every CSS rule
    ui.py                 HTML component builders
    art.py                generated SVG destination artwork
    splash.py             the opening window-shade animation
```

Each layer depends only on the ones above it. No module imports `streamlit`
except `app.py`, so the whole backend is testable without a Streamlit runtime.

## How recommendations work

1. **Validate** — answers are clamped to legal ranges; anything repaired is
   reported rather than silently corrected.
2. **Vectorise** — answers become a point in the same feature space as the
   catalogue. Coordinates use the catalogue centroid, not zero: zero is a real
   place in the Atlantic and would drag results toward whatever sits near it.
3. **Score** — cosine similarity against every destination, using row norms
   precomputed at fit time.
4. **Diversify** — K-Means groups destinations into profiles, and at most
   `MAX_PER_CLUSTER` results may come from one profile. Without this, cosine
   similarity returns six variations of the same place, because near-identical
   destinations sit at near-identical angles.

Self-similarity is exactly 1.0: feeding a destination's own row back in ranks
that destination first.

## Airports

The notebook joins airports on an exact `(city, country)` string match, which
misses on any spelling difference and fills the misses with `"Unknown"`.
`has_airport` is computed *before* that fill, so a row can hold a good airport
name while its flag reads 0 — which is why gating the display on that flag hid
almost every airport.

Resolution is now a three-step ladder:

1. **Direct** — the row's own join succeeded.
2. **Normalised name** — retry with case, accents and punctuation stripped, so
   `Al-'Ula`, `AlUla` and `al ula` all match.
3. **Geographic** — nearest airport by great-circle distance, reported with
   that distance.

Every destination carrying coordinates resolves to an airport. Only a catalogue
with no airport columns at all produces none, and the interface says so plainly.

## Connecting your own data

`tripwise_data.csv` — the export of the notebook's `final_df` — goes beside
`app.py`. Without it a demo catalogue loads and the interface says so.

Required:

| Column | Meaning |
| --- | --- |
| `city`, `country` | Destination identity |
| `latitude`, `longitude` | Coordinates |
| `culture` … `seclusion` | The nine taste scores, 1–5 |
| `temp_avg_yearly` | Average yearly temperature, °C |
| `budget_level_encoded` | 1 budget, 2 mid-range, 3 luxury |

Optional — each lights up more of the card:

| Column | Adds |
| --- | --- |
| `name`, `iata`, `icao` | Airport names and codes |
| `latitude_airport`, `longitude_airport` | More accurate distance fallback |
| `HotelName` | Suggested stay |
| `Attractions` | Nearby attractions |
| `HotelRating_encoded` | Sharper cost estimates |
| `region_*` | Region chip and clustering insight |

Missing columns are omitted from the card, never guessed at. Malformed numbers
are coerced, and the repair is reported in the loader's health notice.

## Reliability

- No user input can raise: values are clamped, and a filter that would empty the
  catalogue is relaxed with an explanation instead of returning nothing.
- Per-row derivations are wrapped by `safe()`, so a malformed cell costs one
  line on one card rather than the page.
- Page sections are wrapped by `guard()`, so a failing section degrades alone.
- Errors are logged with context to stderr.

## Performance

- `@st.cache_data` on catalogue loading — parsed and cleaned once.
- `@st.cache_resource` on the fitted models and the airport index, keyed by a
  catalogue fingerprint rather than the DataFrame, which is unhashable.
- Row norms are precomputed at fit time, so scoring is one matrix-vector product.
- Distance search is vectorised across the whole airport index.

## Notes

Costs are planning estimates derived from budget tier and accommodation
standard, not quotes. Seasons are inferred from latitude and average
temperature. Confirm both before booking.
