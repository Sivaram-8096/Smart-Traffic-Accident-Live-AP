SMART TRAFFIC AP - FIXED APP

Replace your existing app.py with this app.py.
Run from the folder containing app.py:

python -m py_compile app.py
streamlit run app.py --server.port 8505

Fixes: duplicate/malformed Live Weather block removed; indentation repaired; unsupported datetime_input replaced with date_input + time_input; TomTom API key redacted from HTTP errors.

Keep your existing artifacts folder, traffic_accident.csv, and reported_incidents.csv if you have them. Do not upload or share your TomTom API key.
