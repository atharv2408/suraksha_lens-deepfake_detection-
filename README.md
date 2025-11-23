# 🔍 SurakshaLens AI  
### AI-Powered Deepfake Detection, Cyber Complaint Generator & Legal Assistance (in progress)

SurakshaLens AI is a full-stack platform designed to **detect deepfakes**, **assist victims of cyber harassment**, and **generate structured cyber-complaints** that can be filed with official authorities.  
It combines **FastAPI (Python)** backend + **React (Vite + Tailwind)** frontend.

---

## 🚀 Features

### ✅ **Deepfake Detection**
- Upload an image and instantly detect:
  - Whether it is AI-generated / manipulated
  - Confidence score (0–100%)
- Powered by:
  - **EfficientNet-B0** deepfake model  
  - Custom preprocessing + accuracy-tuned pipeline

---

### ✍️ **Cyber Complaint Generator**
Automatically generates:
- Victim details section  
- Incident description  
- Platform details  
- Evidence list  
- Legal-style formatted text  
Ready for:
- National Cybercrime Reporting Portal  
- Local police station  
- Digital submission or print  

---

### 🧠 **Legal AI Assistance (Coming Soon)**
- Explain cyber laws in simple language  
- Suggest correct sections / acts  
- Provide next-step guidance  

---

## 🏗️ Tech Stack

### **Frontend**
- React (Vite)
- TailwindCSS
- Axios  
- Modern UI (glassmorphism + neon theme)

### **Backend (FastAPI)**
- FastAPI + Uvicorn
- Pydantic
- PyTorch
- EfficientNet deepfake model
- CORS enabled for local development

### **Model**
- EfficientNet-B0  
- Pretrained deepfake weights  
- Custom preprocessing

---

## 📦 Project Structure
surakshalens-ai/
│
├── backend/
│ ├── app/
│ │ ├── main.py
│ │ ├── api/routes.py
│ │ ├── core/config.py
│ │ ├── services/deepfake_detector.py
│ │ └── models/weights/deepfake_efficientnet_best.pth
│
└── surakshalens-frontend/
├── src/
│ ├── pages/
│ ├── components/
│ ├── utils/api.js
│ └── App.jsx


---

## ⚙️ Installation & Setup

### **Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
Backend runs on:
👉 http://127.0.0.1:8000

cd surakshalens-frontend
npm install
npm run dev

Frontend runs on:
👉 http://127.0.0.1:5173

API Routes
Route	Method	Description
/api/v1/deepfake/analyze	POST	Analyze image deepfake
/api/v1/complaint/generate	POST	Generate complaint draft

📄 License

MIT License

👨‍💻 Author

Atharv S. Munj
BSC TYCS • AI & Deep Learning Enthusiast
📧 atharvmunj24@gmail.com

⭐ Support the project

If you like this project, give the repository a Star ⭐ on GitHub!


---

If you want, I can also generate:

✅ A **project banner image**  
✅ A **logo for SurakshaLens**  
✅ A **badges section** (version, license, tech stack icons)  
Just tell me — type **2** for next step!

