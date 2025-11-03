from flask import Flask, render_template, request, redirect, url_for, session
import os
from genetic_algorithm import run_ga  # تأكد أن run_ga ترجع dict كما شرحنا

app = Flask(__name__)
app.secret_key = 'secret'  # مطلوب لتخزين النتائج مؤقتًا

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# إنشاء المجلدات إذا لم تكن موجودة
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            session['results'] = run_ga(filepath)  # حفظ النتائج مؤقتًا
            return redirect(url_for('results'))    # redirect لتجنب POST عند Refresh
    return render_template('index.html', results=None)

@app.route('/results', methods=['GET'])
def results():
    results = session.pop('results', None)  # جلب النتائج وحذفها بعد العرض
    if results is None:
        return redirect(url_for('index'))    # إذا لم توجد نتائج، ارجع للصفحة الرئيسية
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
