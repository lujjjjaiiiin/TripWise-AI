# TripWise AI

Destination matching for travellers. Describe how you like to travel and
TripWise ranks real destinations against your answers, with the cost, climate,
season and nearest airport worked out for each one.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints.

## Connecting your own data

The app looks for `tripwise_data.csv` beside `app.py` — the export of the
notebook's `final_df`. Without it, a small demo catalogue loads instead and the
interface says so.

Required columns:

| Column | Meaning |
| --- | --- |
| `city`, `country` | Destination identity |
| `latitude`, `longitude` | Coordinates |
| `culture` … `seclusion` | The nine taste scores, 1–5 |
| `temp_avg_yearly` | Average yearly temperature, °C |
| `budget_level_encoded` | 1 budget, 2 mid-range, 3 luxury |

Optional but recommended — each one lights up more of the destination card:

| Column | Adds |
| --- | --- |
| `name`, `iata` | Nearest airport and its code |
| `HotelName` | Suggested stay |
| `Attractions` | Nearby attractions |
| `HotelRating_encoded` | Sharper cost estimates |
| `region_*` | Region chip and clustering insight |

Anything missing is omitted from the card rather than guessed at.

## Structure

```
app.py              routing, the planner form, page assembly
tripwise/
    theme.py        design tokens and every CSS rule
    ui.py           HTML component builders
    art.py          generated SVG destination artwork
    engine.py       ranking, cost, season, explanations, insights
    data.py         catalogue loading and validation
    splash.py       the opening window-shade animation
```

Python handles data, ranking and derived intelligence. Everything visible is
custom markup from `ui.py` styled by `theme.py`; Streamlit's own interface is
hidden. To restyle the product, `theme.py` is the only file you need.

## Notes

Costs are planning estimates derived from budget tier and accommodation
standard, not quotes. Seasons are inferred from latitude and average
temperature. Confirm both before booking.
