from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, os
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # TODO: change for production
DB_FILE = "database.db"

# ---------- DB ----------
def conn():
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = conn()
    cur = c.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS search_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        source TEXT,
        destination TEXT,
        date TEXT,
        time TEXT,
        vehicle_type TEXT,
        predicted_price TEXT,
        predicted_time TEXT,
        timestamp TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    c.commit(); c.close()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
# ---------- Auth helper ----------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapper

# ---------- Routes ----------
@app.route('/')
@login_required
def home():
    if 'user_id' in session:
        return render_template('index.html', google_api_key=GOOGLE_MAPS_API_KEY)
    return redirect(url_for('login'))

@app.route('/whoami')
def whoami():
    if 'user_id' not in session:
        return jsonify({"logged_in": False}), 200
    c = conn()
    u = c.execute("SELECT username FROM users WHERE id=?", (session['user_id'],)).fetchone()
    c.close()
    return jsonify({"logged_in": True, "username": u['username'] if u else "user"}), 200

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or "").strip()
        password = request.form.get('password') or ""
        if not username or not password:
            return render_template('login.html', error="Please enter username and password.")
        c = conn()
        user = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        c.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('login.html', tab="signup")

    username = (request.form.get('username') or "").strip()
    password = request.form.get('password') or ""
    confirm  = request.form.get('confirm') or ""

    if len(username) < 3:
        return render_template('login.html', tab="signup", error="Username must be at least 3 characters.")
    if len(password) < 6:
        return render_template('login.html', tab="signup", error="Password must be at least 6 characters.")
    if password != confirm:
        return render_template('login.html', tab="signup", error="Passwords do not match.")

    conn_obj = sqlite3.connect(DB_FILE)
    conn_obj.row_factory = sqlite3.Row
    cursor = conn_obj.cursor()

    existing_user = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if existing_user:
        conn_obj.close()
        return render_template('login.html', tab="signup", error="Username already exists.")

    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                   (username, generate_password_hash(password)))
    conn_obj.commit()
    conn_obj.close()

    return render_template('login.html', success="Account created! Please log in.")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------- APIs (auth required) ----------
from datetime import datetime, timezone  # <-- ensure this line exists at top

@app.route('/save_search', methods=['POST'])
@login_required
def save_search():
    data = request.json or {}
    c = conn()
    c.execute("""INSERT INTO search_history
        (user_id, source, destination, date, time, vehicle_type, predicted_price, predicted_time, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)""", (
            session['user_id'],
            data.get('source', ''),
            data.get('destination', ''),
            data.get('date', ''),
            data.get('time', ''),
            data.get('vehicle_type', ''),
            data.get('predicted_price', ''),
            data.get('predicted_time', ''),
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")  # ✅ fixed
        ))
    c.commit()
    c.close()
    return jsonify({"ok": True})


@app.route('/history')
@login_required
def history():
    c = conn()
    rows = c.execute("""SELECT id, source, destination, date, time, vehicle_type,
                        predicted_price, predicted_time, timestamp
                        FROM search_history
                        WHERE user_id=? ORDER BY timestamp DESC""",
                     (session['user_id'],)).fetchall()
    c.close()
    return jsonify([dict(r) for r in rows])

@app.route('/predict_cancellation', methods=['POST'])
@login_required
def predict_cancellation():
    import os, pickle, pandas as pd, traceback
    try:
        data = request.json or {}
        pickup = data.get('pickup')
        drop = data.get('drop')
        date = data.get('date')
        time = data.get('time')
        car_type = data.get('car_type')

        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, "preprocessing.pkl"), "rb") as f:
            preproc = pickle.load(f)
        with open(os.path.join(base_path, "driver_cancellation_model.pkl"), "rb") as f:
            model = pickle.load(f)

        scaler = preproc["scaler"]
        encoders = preproc["encoders"]

        # ✅ Construct feature frame
        df = pd.DataFrame([{
            "Pickup Location": pickup,
            "Drop Location": drop,
            "Vehicle Type": car_type,
            "Time": int(time.split(":")[0]),
            "day_of_week": pd.to_datetime(date).dayofweek,
            "month": pd.to_datetime(date).month
        }])

        # ✅ Encode categorical columns safely
        for col in ['Pickup Location', 'Drop Location', 'Vehicle Type']:
            le = encoders[col]
            val = df.at[0, col]
            if val not in le.classes_:
                print(f"⚠️ unseen value for {col}: {val}, defaulting to {le.classes_[0]}")
                val = le.classes_[0]
            df[col] = le.transform([val])

        # ✅ Align dataframe with scaler’s expected columns
        if hasattr(scaler, "feature_names_in_"):
            print("✅ Scaler expects columns:", list(scaler.feature_names_in_))
            df = df[scaler.feature_names_in_]
        else:
            # fallback for older sklearn
            df = df[['Pickup Location', 'Drop Location', 'Vehicle Type', 'Time', 'day_of_week', 'month']]

        print("🚀 Final data sent to model:\n", df)

        # ✅ Scale and predict
        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        return jsonify({
            "ok": True,
            "cancel_pred": int(pred),
            "probability": round(float(prob) * 100, 2)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500



if __name__ == '__main__':
    if not os.path.exists(DB_FILE):
        init_db()
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT
    app.run(host="0.0.0.0", port=port)
