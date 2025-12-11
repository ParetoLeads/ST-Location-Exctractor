# Location Search Term Filter

Streamlit app to upload Google Ads search term CSVs, geocode detected locations, check whether they fall inside a target area, and suggest keep vs exclude lists.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
- Required columns: `search_term`, `impressions` (case-insensitive aliases like `term`, `impr` are accepted).

## How it works
- Geocodes the target area once (OpenStreetMap Nominatim, free/rate-limited) and builds a polygon/bbox.
- Geocodes each search term; classifies locations as inside/outside the target area; caches requests per session.
- Shows aggregated impressions for keep vs exclude and provides CSV downloads.

### If geocoding is blocked on your host
- Set an alternate geocoder endpoint via environment variables when deploying:
  - `GEOCODER_URL=https://geocode.maps.co/search` (or another OSM-compatible endpoint)
  - `GEOCODER_API_KEY=<your key>` if the service requires it (maps.co supports free API keys).
- Default remains `https://nominatim.openstreetmap.org/search`.

## Hosting / sharing
- Easiest: Streamlit Community Cloud (free for small workloads). Push this repo to GitHub and deploy via https://share.streamlit.io/.
- Other options: run `streamlit run app.py` on a small VM (Fly.io/Render/EC2) with HTTPS via a reverse proxy; keep Nominatim rate limits in mind.

## Notes
- Free Nominatim is rate-limited; adjust the in-app throttle slider if you hit limits.
- For higher reliability/volume, swap in a paid geocoder (e.g., Google Geocoding/Places) and update `geocode` in `app.py`.

