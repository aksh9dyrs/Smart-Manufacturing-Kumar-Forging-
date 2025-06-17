from flask import Flask, jsonify, render_template_string, request
import psycopg2
import json
import requests

app = Flask(__name__)

# --- Database connection ---
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="postgres",
    port="5132"
)

# --- LLaMA/Ollama Integration ---
def ask_llama(prompt):
    try:
        response = requests.post(
            'http://localhost:5132/api/generate',
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )
        data = response.json()
        return data.get("response", "No answer generated.")
    except Exception as e:
        return f"Error querying LLaMA: {str(e)}"

# --- HTML Template (NAV bar updated + Ask section added) ---
html_template = """ 
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      background: linear-gradient(rgba(0,0,0,.7), rgba(0,0,0,.7)),
        url('https://images.unsplash.com/photo-1557804506-669a67965ba0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
      background-size: cover;
      color: #ffffff;
      font-family: 'Segoe UI', sans-serif;
    }
    header {
      text-align: center;
      padding: 30px 20px 10px;
    }
    header img { width: 80px; height: auto; }
    header h1 { margin: 10px 0 5px; font-size: 2em; color: #00bcd4; }
    header p { margin: 0; font-size: 1.1em; color: #b2ebf2; }
    nav {
      background-color: #1e3a8a;
      padding: 10px 20px;
      display: flex;
      gap: 20px;
      align-items: center;
    }
    nav a {
      color: white;
      text-decoration: none;
      font-weight: bold;
      font-family: sans-serif;
    }
    nav a:hover { text-decoration: underline; }
    .container {
      max-width: 950px;
      margin: 20px auto;
      background: rgba(0, 0, 0, 0.7);
      padding: 30px 40px;
      border-radius: 12px;
      box-shadow: 0 0 15px rgba(0, 200, 255, 0.4);
    }
    input, button {
      padding: 10px;
      margin: 5px;
      border-radius: 5px;
      border: none;
      font-size: 1em;
    }
    button {
      background-color: #00acc1;
      color: white;
      cursor: pointer;
    }
    button:hover { background-color: #007c91; }
    .card {
      background: rgba(255,255,255,0.1);
      padding: 15px;
      margin: 15px 0;
      border-radius: 8px;
      box-shadow: 0 0 5px rgba(255,255,255,0.2);
    }
  </style>
</head>
<body>

<nav>
  <a href="/">🏠 Home</a>
  <a href="/breakdown">⚠️ Breakdown</a>
  <a href="/production">🏭 Production</a>
  <a href="/events">📋 All Events</a>
  <a href="/cosine">🔁 Cosine Similarity</a>
  <a href="/ask">🧠 Ask LLaMA</a>
</nav>

<header>
  <img src="https://img.icons8.com/ios-filled/100/00bcd4/factory.png" alt="Logo">
  <h1>Smart Manufacturing Dashboard</h1>
  <p>Real-time insights. Intelligent decisions. Efficient outcomes.</p>
</header>

<div class="container">
  <h2>{{ title }}</h2>

  {% if cosine %}
    <input type="number" id="id1" placeholder="Event ID 1">
    <input type="number" id="id2" placeholder="Event ID 2">
    <button onclick="getCosine()">Compute Cosine</button>
    <div id="cosine_result"></div>

  {% elif show_input %}
    <input type="number" id="inputId" placeholder="Enter Event ID">
    <button onclick="findSimilar()">Find Similar</button>

  {% elif ask %}
    <input type="text" id="userPrompt" placeholder="Ask your manufacturing question..." style="width: 80%;">
    <button onclick="askLlama()">Ask</button>
    <div id="llama_result" style="margin-top: 20px;"></div>
  {% endif %}

  {% if endpoint %}
    <button onclick="loadData()">Load Events</button>
  {% endif %}
  
  <div id="results"></div>
</div>

<script>
  const base = '';

  function loadData() {
    fetch(base + '{{ endpoint }}')
      .then(res => res.json())
      .then(showResults);
  }

  function findSimilar() {
    const id = document.getElementById('inputId').value;
    fetch(`${base}{{ endpoint }}/similar/` + id)
      .then(res => res.json())
      .then(showResults);
  }

  function getCosine() {
    const id1 = document.getElementById('id1').value;
    const id2 = document.getElementById('id2').value;
    fetch(`/api/cosine/${id1}/${id2}`)
      .then(res => res.json())
      .then(data => {
        document.getElementById('cosine_result').innerHTML = `<strong>Cosine Similarity:</strong> ${data.similarity}`;
      });
  }

  function askLlama() {
    const prompt = document.getElementById('userPrompt').value;
    fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById('llama_result').innerHTML = `<strong>Response:</strong> ${data.response}`;
    });
  }

  function showResults(data) {
    const div = document.getElementById('results');
    div.innerHTML = '';
    data.forEach(e => {
      const el = document.createElement('div');
      el.className = 'card';
      el.innerHTML = `<strong>ID:</strong> ${e.id}<br>
                      <strong>Type:</strong> ${e.event_type}<br>
                      <strong>Machine:</strong> ${e.machine_name}<br>
                      <strong>Duration:</strong> ${e.duration_minutes} mins<br>
                      <strong>Timestamp:</strong> ${new Date(e.timestamp).toLocaleString()}<br>
                      <strong>Notes:</strong> ${e.notes}<br>
                      <strong>Embedding:</strong> ${e.embedding}`;
      div.appendChild(el);
    });
  }
</script>

</body>
</html>
"""

# --- Web Pages ---
@app.route('/')
def index():
    return render_template_string(html_template, title="Home", endpoint="/api/events", show_input=False)

@app.route('/breakdown')
def breakdown_page():
    return render_template_string(html_template, title="Breakdown Events", endpoint="/api/breakdown", show_input=True)

@app.route('/production')
def production_page():
    return render_template_string(html_template, title="Production Events", endpoint="/api/production", show_input=True)

@app.route('/events')
def all_events_page():
    return render_template_string(html_template, title="All Events", endpoint="/api/events", show_input=False)

@app.route('/cosine')
def cosine_page():
    return render_template_string(html_template, title="Cosine Similarity Between Events", endpoint="/api/events", cosine=True)

@app.route('/ask')
def ask_page():
    return render_template_string(html_template, title="Ask LLaMA AI", ask=True)

# --- DB Helper ---
def query_db(query, params=()):
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        result = [dict(zip(colnames, row)) for row in rows]
        for r in result:
            if 'embedding' in r and isinstance(r['embedding'], list):
                r['embedding'] = json.dumps(r['embedding'])
        return result

# --- API Endpoints ---
@app.route('/api/events')
def get_events():
    return jsonify(query_db("SELECT * FROM manufacturing_events ORDER BY timestamp DESC"))

@app.route('/api/breakdown')
def get_breakdowns():
    return jsonify(query_db("SELECT * FROM manufacturing_events WHERE event_type = 'Breakdown' ORDER BY timestamp DESC"))

@app.route('/api/production')
def get_productions():
    return jsonify(query_db("SELECT * FROM manufacturing_events WHERE event_type = 'Production' ORDER BY timestamp DESC"))

@app.route('/api/<event_type>/similar/<int:event_id>')
def get_similar(event_type, event_id):
    result = query_db(
        """
        SELECT * FROM manufacturing_events
        WHERE event_type = %s AND id != %s
        ORDER BY embedding <-> (SELECT embedding FROM manufacturing_events WHERE id = %s)
        LIMIT 5
        """,
        (event_type.capitalize(), event_id, event_id)
    )
    return jsonify(result)

@app.route('/api/cosine/<int:id1>/<int:id2>')
def cosine_similarity(id1, id2):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 - (e1.embedding <=> e2.embedding) AS similarity
            FROM manufacturing_events e1, manufacturing_events e2
            WHERE e1.id = %s AND e2.id = %s
        """, (id1, id2))
        result = cur.fetchone()
    return jsonify({"similarity": round(result[0], 4) if result else None})

@app.route('/api/ask', methods=['POST'])
def llama_api():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response = ask_llama(prompt)
    return jsonify({"response": response})

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)
