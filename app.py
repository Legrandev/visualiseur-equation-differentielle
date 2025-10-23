from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def euler_method(f, y0, t):
    """Méthode d'Euler pour résoudre dy/dt = f(t,y)"""
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y

def runge_kutta_4(f, y0, t):
    """Méthode de Runge-Kutta d'ordre 4"""
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + dt/2, y[i-1] + dt/2 * k1)
        k3 = f(t[i-1] + dt/2, y[i-1] + dt/2 * k2)
        k4 = f(t[i-1] + dt, y[i-1] + dt * k3)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_equation():
    try:
        data = request.get_json()
        print("Données reçues:", data)  # Debug
        
        # Paramètres avec valeurs par défaut
        t0 = data.get('t0', 0)
        tf = data.get('tf', 5.0)
        y0 = data.get('y0', 1.0)
        k = data.get('k', 1.0)
        n_points = data.get('n_points', 100)
        
        # Équation différentielle : croissance exponentielle
        def equation(t, y):
            return k * y  # dy/dt = k*y
        
        # Discrétisation du temps
        t = np.linspace(float(t0), float(tf), int(n_points))
        
        # Résolution avec différentes méthodes
        y_euler = euler_method(equation, float(y0), t)
        y_rk4 = runge_kutta_4(equation, float(y0), t)
        
        # Solution exacte (pour comparaison)
        y_exact = float(y0) * np.exp(float(k) * t)
        
        # Création du graphique
        plt.figure(figsize=(10, 6))
        plt.plot(t, y_euler, 'b-', label='Méthode d\'Euler', linewidth=2)
        plt.plot(t, y_rk4, 'r-', label='Runge-Kutta 4', linewidth=2)
        plt.plot(t, y_exact, 'g--', label='Solution exacte', linewidth=1.5)
        plt.xlabel('Temps (t)')
        plt.ylabel('y(t)')
        plt.title(f'Résolution de dy/dt = {k}*y avec y(0) = {y0}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Conversion en base64 pour l'affichage web
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'status': 'success',
            'plot_url': f"data:image/png;base64,{plot_url}"
        })
    
    except Exception as e:
        print("Erreur:", str(e))  # Debug
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)