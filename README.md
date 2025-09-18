# 🎵 LangChain Database Analytics

A full-stack application that allows you to query your Chinook music database using natural language and get beautiful visualizations. Built with Flask (backend) and Next.js (frontend).

## 🌟 Features

- **Natural Language Querying**: Ask questions in plain English
- **Smart SQL Generation**: Uses OpenAI GPT-4 to convert questions to SQL
- **Beautiful Visualizations**: Interactive charts using Chart.js
- **Real-time Results**: Get instant insights from your database
- **Modern UI**: Responsive design with Tailwind CSS
- **Error Handling**: Graceful error management and user feedback

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 16+** with npm
3. **MySQL Server** with Chinook database
4. **OpenAI API Key**

### Setup Instructions

1. **Clone and Navigate**
   ```bash
   # Navigate to your project directory
   cd "langchain+zigment"
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Create .env file with your OpenAI API key
   echo OPENAI_API_KEY=your_openai_api_key_here > .env
   ```

3. **Frontend Setup**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Database Configuration**
   
   Update the MySQL connection string in `backend/app.py` if needed:
   ```python
   mysql_uri = "mysql+pymysql://username:password@localhost:3306/chinook"
   ```

### 🏃‍♂️ Running the Application

#### Option 1: Using Batch Files (Windows)
```bash
# Terminal 1: Start Backend
start_backend.bat

# Terminal 2: Start Frontend
start_frontend.bat
```

#### Option 2: Manual Start
```bash
# Terminal 1: Backend
cd backend
python start.py

# Terminal 2: Frontend  
cd frontend
npm run dev
```

### 🌐 Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 📊 Sample Questions

Try these example queries:

- "Show me top 5 most popular albums with their number of songs"
- "Which artists have sold the most tracks?"
- "What are the most popular music genres by sales?"
- "Show me customer purchases by country"
- "Which employees have the highest sales?"
- "What are the longest tracks in the database?"

## 🏗️ Architecture

```
┌─────────────────┐    HTTP Requests    ┌──────────────────┐
│                 │ ───────────────────► │                  │
│  Next.js        │                      │  Flask           │
│  Frontend       │ ◄─────────────────── │  Backend         │
│  (Port 3000)    │    JSON Response     │  (Port 5000)     │
└─────────────────┘                      └──────────────────┘
         │                                         │
         │                                         │
         ▼                                         ▼
┌─────────────────┐                      ┌──────────────────┐
│                 │                      │                  │
│  Chart.js       │                      │  LangChain +     │
│  Visualizations │                      │  OpenAI GPT-4    │
│                 │                      │                  │
└─────────────────┘                      └──────────────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │                  │
                                         │  MySQL           │
                                         │  Chinook DB      │
                                         │                  │
                                         └──────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework
- **LangChain**: LLM orchestration framework  
- **OpenAI**: GPT-4 for natural language processing
- **PyMySQL**: MySQL database connector
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **Next.js 14**: React framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Data visualization library
- **Axios**: HTTP client for API requests

## 📁 Project Structure

```
langchain+zigment/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── start.py            # Startup script
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   └── ChartComponent.tsx
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx       # Main page
│   ├── package.json       # Node.js dependencies
│   └── README.md         # Frontend documentation
├── main.ipynb            # Original Jupyter notebook
├── Chinook_MySql.sql     # Database schema
├── start_backend.bat     # Windows batch file
├── start_frontend.bat    # Windows batch file
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

Create `backend/.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Database Connection

Update `backend/app.py`:
```python
mysql_uri = "mysql+pymysql://username:password@localhost:3306/chinook"
```

## 🎯 API Endpoints

- `GET /health` - Health check
- `POST /api/ask` - Process natural language questions
- `POST /api/execute-sql` - Execute raw SQL (debugging)
- `GET /api/schema` - Get database schema

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Make sure `.env` file exists in `backend/` directory
   - Verify your API key is valid and has sufficient credits

2. **Database Connection Error**
   - Check if MySQL server is running
   - Verify database credentials in `app.py`
   - Ensure Chinook database is properly installed

3. **CORS Issues**
   - Backend includes CORS support for `localhost:3000`
   - If using different ports, update CORS configuration

4. **Module Not Found**
   - Run `pip install -r requirements.txt` in backend directory
   - Run `npm install` in frontend directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Enjoy querying your database with natural language! 🎉**
