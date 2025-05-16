# LuckyBikey Server

This is the backend service for **LuckyBikey** — a smart cycling assistant.  
It supports user preference management, route recommendation, user feedback collection, and clustering logic to personalize the user experience.

## 🛠 Features

- **Flask-based HTTP API endpoints**
- **Custom route recommendation algorithms**
- **User preference management**
- **User feedback collection and processing**
- **User clustering logic** to support personalization and insights
- **Google Cloud Functions** deployment using Python

## ⚙️ Architecture

The backend is written in **Python** and designed to run on **Google Cloud Functions** for scalable, serverless execution.  
While Firebase Functions officially support Node.js and TypeScript, Python Cloud Functions can be deployed within Firebase projects using the Google Cloud Console or CLI.

## 📦 Requirements

- Python 3.9+  
- [Flask](https://flask.palletsprojects.com/)  
- [Google Cloud SDK](https://cloud.google.com/sdk)  
- [firebase_functions](https://firebase.google.com/docs/functions?hl=ko)
- [numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [scikit-learn](https://scikit-learn.org/)
- [geopy](https://geopy.readthedocs.io/en/stable/)

## ☁️ Deploying to Google Cloud Functions

To deploy all functions:

```bash
firebase deploy --only functions
```

To deploy a specific function:

```bash
firebase deploy --only functions:function1
```