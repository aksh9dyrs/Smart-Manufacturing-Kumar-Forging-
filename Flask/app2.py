import streamlit as st
import pandas as pd
import psycopg2
import ast
from datetime import datetime
import umap
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
from google.api_core import retry
from functools import lru_cache
from collections import deque
import threading
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
import os
import wikipedia
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import concurrent.futures

# Configure Wikipedia
wikipedia.set_lang("en")

# ----------------- Helper Functions for Web Search -----------------
def search_wikipedia(query, max_results=1):
    """Search Wikipedia for relevant results"""
    try:
        search_results = wikipedia.search(query, results=max_results)
        articles = []
        
        for title in search_results:
            try:
                # Get only the first sentence of the summary for faster processing
                summary = wikipedia.summary(title, sentences=1)
                articles.append({
                    'title': title,
                    'summary': summary,
                    'url': wikipedia.page(title).url
                })
            except:
                continue
                
        return articles
    except Exception as e:
        return []

def search_google(query, num_results=1):
    """Search Google for relevant results"""
    try:
        results = []
        for url in search(query, num_results=num_results):
            try:
                # Reduced timeout to 1 second
                response = requests.get(url, timeout=1)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else url
                description = soup.find('meta', {'name': 'description'})
                description = description['content'] if description else "No description available"
                
                results.append({
                    'title': title,
                    'url': url,
                    'description': description[:100]  # Limit description length
                })
            except:
                continue
        return results
    except Exception as e:
        return []

def fetch_web_data(query):
    """Fetch data from Wikipedia and Google in parallel"""
    try:
        # Use ThreadPoolExecutor with shorter timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            wiki_future = executor.submit(search_wikipedia, query)
            google_future = executor.submit(search_google, query)
            
            try:
                wiki_results = wiki_future.result(timeout=2)
                google_results = google_future.result(timeout=2)
            except concurrent.futures.TimeoutError:
                return {'wikipedia': [], 'google': []}
        
        return {
            'wikipedia': wiki_results,
            'google': google_results
        }
    except Exception as e:
        return {'wikipedia': [], 'google': []}

# Add caching for web searches
@st.cache_data(ttl=1800)  # Cache for 30 minutes instead of 1 hour
def cached_wikipedia_search(query, max_results=1):
    """Cached Wikipedia search"""
    return search_wikipedia(query, max_results)

@st.cache_data(ttl=1800)  # Cache for 30 minutes instead of 1 hour
def cached_google_search(query, num_results=1):
    """Cached Google search"""
    return search_google(query, num_results)

@st.cache_data(ttl=1800)  # Cache for 30 minutes instead of 1 hour
def fetch_web_data(query):
    """Fetch data from Wikipedia and Google in parallel with caching"""
    try:
        # Use cached search functions
        wiki_results = cached_wikipedia_search(query)
        google_results = cached_google_search(query)
        
        return {
            'wikipedia': wiki_results,
            'google': google_results
        }
    except Exception as e:
        return {'wikipedia': [], 'google': []}

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Smart Manufacturing Dashboard",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="collapsed"  # Changed to collapsed since we're moving menu to top
)

# ----------------- API Keys -----------------
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://engineering.cmu.edu/techspark/_files/images/facilities/machine-shop/1-student-shop.png');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Add overlay to ensure content readability */
    .main > div {
        background-color: rgba(30, 30, 30, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    
    /* Adjust navigation bar for better visibility */
    .nav-container {
        background-color: rgba(45, 45, 45, 0.95);
        padding: 10px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid #4CAF50;
        backdrop-filter: blur(5px);
    }
    
    /* Adjust feature boxes for better visibility */
    .feature-box {
        background-color: rgba(45, 45, 45, 0.95);
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }
    
    /* Adjust stats boxes for better visibility */
    .stats-box {
        background-color: rgba(45, 45, 45, 0.95);
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
        backdrop-filter: blur(5px);
    }
    
    /* Existing styles ... */
    .main-header {
        text-align: center;
        color: #FFD600 !important;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 0 0 10px #FFD600, 0 0 20px #FFD600, 2px 2px 4px rgba(0,0,0,0.7);
    }
    .sub-header {
        text-align: center;
        color: #FFFFFF;
        font-size: 24px;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .feature-title {
        color: #4CAF50;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .feature-content {
        color: #FFFFFF;
        font-size: 16px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .stats-title {
        color: #4CAF50;
        font-size: 18px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .stats-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .nav-item {
        display: inline-block;
        padding: 10px 20px;
        color: #FFFFFF !important;
        background: rgba(45, 45, 45, 0.95) !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
        margin: 0 5px;
        font-weight: bold;
        font-size: 18px;
        text-decoration: none;
        transition: filter 0.2s, background 0.2s, color 0.2s, border 0.2s;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .nav-item:hover, .nav-item.active {
        filter: brightness(1.15);
        color: #FFD600 !important;
        border-color: #FFD600 !important;
    }
    .kfi-logo {
        width: 260px;
        height: 260px;
        object-fit: cover;
        border: 6px solid #4CAF50;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.25), 0 2px 8px rgba(0,0,0,0.18);
        margin-bottom: 20px;
        background: #222;
        display: inline-block;
    }
    
    /* Global text color override */
    .stApp * {
        color: #FFFFFF !important;
    }
    
    /* Override specific Streamlit elements */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li {
        color: #FFFFFF !important;
    }
    
    /* DataFrame specific overrides */
    .stDataFrame div[data-testid="stDataFrame"] * {
        color: #FFFFFF !important;
    }
    .stDataFrame div[data-testid="stDataFrame"] table * {
        color: #FFFFFF !important;
    }
    .stDataFrame div[data-testid="stDataFrame"] th {
        color: #FFFFFF !important;
        background-color: rgba(76, 175, 80, 0.2) !important;
    }
    .stDataFrame div[data-testid="stDataFrame"] td {
        color: #FFFFFF !important;
    }
    .stDataFrame div[data-testid="stDataFrame"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Metric specific overrides */
    .stMetric div[data-testid="stMetricValue"], 
    .stMetric div[data-testid="stMetricLabel"],
    .stMetric div[data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }
    
    /* Expander specific overrides */
    .stExpander div[data-testid="stExpander"] * {
        color: #FFFFFF !important;
    }
    .stExpander div[data-testid="stExpander"] summary {
        color: #FFFFFF !important;
    }
    .stExpander div[data-testid="stExpander"] div {
        color: #FFFFFF !important;
    }
    
    /* Selectbox specific overrides - Enhanced */
    .stSelectbox div[data-testid="stSelectbox"] * {
        color: #FFFFFF !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] select {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] option {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Override Streamlit's selectbox dropdown styles */
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] {
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div:hover {
        background-color: rgba(76, 175, 80, 0.2) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div[aria-selected="true"] {
        background-color: rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Override Streamlit's selectbox placeholder */
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] {
        color: #FFFFFF !important;
    }
    
    /* Override Streamlit's selectbox dropdown arrow */
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] svg {
        color: #FFFFFF !important;
    }
    
    /* Override Streamlit's selectbox dropdown menu */
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] {
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Override Streamlit's selectbox dropdown items */
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div[role="option"] {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: rgba(76, 175, 80, 0.2) !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div[role="option"][aria-selected="true"] {
        background-color: rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Slider specific overrides */
    .stSlider div[data-testid="stSlider"] * {
        color: #FFFFFF !important;
    }
    .stSlider div[data-testid="stSlider"] input {
        color: #FFFFFF !important;
    }
    
    /* Checkbox specific overrides */
    .stCheckbox div[data-testid="stCheckbox"] * {
        color: #FFFFFF !important;
    }
    .stCheckbox div[data-testid="stCheckbox"] label {
        color: #FFFFFF !important;
    }
    
    /* Progress specific overrides */
    .stProgress div[data-testid="stProgress"] * {
        color: #FFFFFF !important;
    }
    
    /* Alert specific overrides */
    .stAlert div[data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }
    
    /* Info specific overrides */
    .stInfo div[data-testid="stInfo"] * {
        color: #FFFFFF !important;
    }
    
    /* Warning specific overrides */
    .stWarning div[data-testid="stWarning"] * {
        color: #FFFFFF !important;
    }
    
    /* Error specific overrides */
    .stError div[data-testid="stError"] * {
        color: #FFFFFF !important;
    }
    
    /* Success specific overrides */
    .stSuccess div[data-testid="stSuccess"] * {
        color: #FFFFFF !important;
    }
    
    /* Text input specific overrides */
    .stTextInput div[data-testid="stTextInput"] * {
        color: #FFFFFF !important;
    }
    .stTextInput div[data-testid="stTextInput"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Text area specific overrides */
    .stTextArea div[data-testid="stTextArea"] * {
        color: #FFFFFF !important;
    }
    .stTextArea div[data-testid="stTextArea"] textarea {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Button specific overrides */
    .stButton button {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .stButton button:hover {
        color: #FFFFFF !important;
        background-color: #4CAF50 !important;
    }
    
    /* Number input specific overrides */
    .stNumberInput div[data-testid="stNumberInput"] * {
        color: #FFFFFF !important;
    }
    .stNumberInput div[data-testid="stNumberInput"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Radio specific overrides */
    .stRadio div[data-testid="stRadio"] * {
        color: #FFFFFF !important;
    }
    .stRadio div[data-testid="stRadio"] label {
        color: #FFFFFF !important;
    }
    
    /* Multiselect specific overrides */
    .stMultiSelect div[data-testid="stMultiSelect"] * {
        color: #FFFFFF !important;
    }
    .stMultiSelect div[data-testid="stMultiSelect"] select {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Date input specific overrides */
    .stDateInput div[data-testid="stDateInput"] * {
        color: #FFFFFF !important;
    }
    .stDateInput div[data-testid="stDateInput"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Time input specific overrides */
    .stTimeInput div[data-testid="stTimeInput"] * {
        color: #FFFFFF !important;
    }
    .stTimeInput div[data-testid="stTimeInput"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* File uploader specific overrides */
    .stFileUploader div[data-testid="stFileUploader"] * {
        color: #FFFFFF !important;
    }
    .stFileUploader div[data-testid="stFileUploader"] button {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Color picker specific overrides */
    .stColorPicker div[data-testid="stColorPicker"] * {
        color: #FFFFFF !important;
    }
    .stColorPicker div[data-testid="stColorPicker"] input {
        color: #FFFFFF !important;
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    
    /* Sidebar specific overrides */
    .css-1d391kg {
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .css-1d391kg * {
        color: #FFFFFF !important;
    }
    
    /* Main content area overrides */
    .main .block-container {
        background-color: rgba(45, 45, 45, 0.95) !important;
    }
    .main .block-container * {
        color: #FFFFFF !important;
    }
    
    /* Override Streamlit's default text colors */
    .stText, .stMarkdown, .stDataFrame, .stMetric, .stExpander, 
    .stTextInput, .stTextArea, .stButton, .stSelectbox, .stSlider, 
    .stCheckbox, .stProgress, .stAlert, .stInfo, .stWarning, 
    .stError, .stSuccess, .stNumberInput, .stRadio, .stMultiSelect, 
    .stDateInput, .stTimeInput, .stFileUploader, .stColorPicker {
        color: #FFFFFF !important;
    }
    
    /* Override any remaining text elements */
    p, h1, h2, h3, h4, h5, h6, span, div, label, input, textarea, select, option {
        color: #FFFFFF !important;
    }
    
    /* Selectbox specific overrides - Black text */
    .stSelectbox div[data-testid="stSelectbox"] {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] select {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] option {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div {
        color: #000000 !important;
    }
    .stSelectbox div[data-testid="stSelectbox"] div[data-baseweb="popover"] div[role="option"] {
        color: #000000 !important;
    }

    /* Selectbox specific overrides - Black text with higher specificity */
    div[data-testid="stSelectbox"] *,
    div[data-testid="stSelectbox"] select,
    div[data-testid="stSelectbox"] option,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"],
    div[data-testid="stSelectbox"] div[data-baseweb="popover"],
    div[data-testid="stSelectbox"] div[data-baseweb="popover"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="popover"] div[role="option"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] span,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] p {
        color: #000000 !important;
    }

    /* Override any potential conflicting styles */
    .stSelectbox *,
    .stSelectbox select,
    .stSelectbox option,
    .stSelectbox div[data-baseweb="select"] *,
    .stSelectbox div[data-baseweb="popover"] * {
        color: #000000 !important;
    }

    /* Force black text for selectbox content */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] {
        color: #000000 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[aria-selected="false"] * {
        color: #000000 !important;
    }

    /* Make text black in light boxes */
    .stSelectbox div[data-testid="stSelectbox"],
    .stTextInput div[data-testid="stTextInput"],
    .stTextArea div[data-testid="stTextArea"],
    .stNumberInput div[data-testid="stNumberInput"],
    .stDateInput div[data-testid="stDateInput"],
    .stTimeInput div[data-testid="stTimeInput"],
    .stFileUploader div[data-testid="stFileUploader"],
    .stColorPicker div[data-testid="stColorPicker"],
    .stDataFrame div[data-testid="stDataFrame"],
    .stMetric div[data-testid="stMetricValue"],
    .stMetric div[data-testid="stMetricLabel"],
    .stMetric div[data-testid="stMetricDelta"],
    .stExpander div[data-testid="stExpander"],
    .stAlert div[data-testid="stAlert"],
    .stInfo div[data-testid="stInfo"],
    .stWarning div[data-testid="stWarning"],
    .stError div[data-testid="stError"],
    .stSuccess div[data-testid="stSuccess"],
    .stProgress div[data-testid="stProgress"],
    .stCheckbox div[data-testid="stCheckbox"],
    .stRadio div[data-testid="stRadio"],
    .stMultiSelect div[data-testid="stMultiSelect"],
    .stSlider div[data-testid="stSlider"] {
        color: #000000 !important;
    }

    /* Make text black in light box inputs and their children */
    .stSelectbox div[data-testid="stSelectbox"] *,
    .stTextInput div[data-testid="stTextInput"] *,
    .stTextArea div[data-testid="stTextArea"] *,
    .stNumberInput div[data-testid="stNumberInput"] *,
    .stDateInput div[data-testid="stDateInput"] *,
    .stTimeInput div[data-testid="stTimeInput"] *,
    .stFileUploader div[data-testid="stFileUploader"] *,
    .stColorPicker div[data-testid="stColorPicker"] *,
    .stDataFrame div[data-testid="stDataFrame"] *,
    .stMetric div[data-testid="stMetricValue"] *,
    .stMetric div[data-testid="stMetricLabel"] *,
    .stMetric div[data-testid="stMetricDelta"] *,
    .stExpander div[data-testid="stExpander"] *,
    .stAlert div[data-testid="stAlert"] *,
    .stInfo div[data-testid="stInfo"] *,
    .stWarning div[data-testid="stWarning"] *,
    .stError div[data-testid="stError"] *,
    .stSuccess div[data-testid="stSuccess"] *,
    .stProgress div[data-testid="stProgress"] *,
    .stCheckbox div[data-testid="stCheckbox"] *,
    .stRadio div[data-testid="stRadio"] *,
    .stMultiSelect div[data-testid="stMultiSelect"] *,
    .stSlider div[data-testid="stSlider"] * {
        color: #000000 !important;
    }

    /* Specific overrides for input elements */
    input, textarea, select, option {
        color: #000000 !important;
    }

    /* Keep other text white */
    .stApp *:not(.stSelectbox *):not(.stTextInput *):not(.stTextArea *):not(.stNumberInput *):not(.stDateInput *):not(.stTimeInput *):not(.stFileUploader *):not(.stColorPicker *):not(.stDataFrame *):not(.stMetric *):not(.stExpander *):not(.stAlert *):not(.stInfo *):not(.stWarning *):not(.stError *):not(.stSuccess *):not(.stProgress *):not(.stCheckbox *):not(.stRadio *):not(.stMultiSelect *):not(.stSlider *) {
        color: #FFFFFF !important;
    }

    /* Make text black in question and filter sections */
    .stTextArea textarea,
    .stTextArea div[data-testid="stTextArea"] *,
    .stTextInput input,
    .stTextInput div[data-testid="stTextInput"] *,
    .stSelectbox select,
    .stSelectbox div[data-testid="stSelectbox"] *,
    .stSelectbox div[data-baseweb="select"] *,
    .stSelectbox div[data-baseweb="popover"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] *,
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] *,
    .stSelectbox div[data-baseweb="select"] div[role="option"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="true"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] span,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] div,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] p,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] {
        color: #000000 !important;
    }

    /* Override Streamlit's default text colors for these sections */
    div[data-testid="stTextArea"] *,
    div[data-testid="stTextInput"] *,
    div[data-testid="stSelectbox"] * {
        color: #000000 !important;
    }

    /* Force black text for all input elements */
    input, textarea, select, option {
        color: #000000 !important;
    }

    /* Force black text for all Streamlit components with light backgrounds */
    .stTextArea, .stTextInput, .stSelectbox, .stNumberInput, .stDateInput, .stTimeInput,
    .stFileUploader, .stColorPicker, .stDataFrame, .stMetric, .stExpander, .stAlert,
    .stInfo, .stWarning, .stError, .stSuccess, .stProgress, .stCheckbox, .stRadio,
    .stMultiSelect, .stSlider {
        color: #000000 !important;
    }

    /* Force black text for all children of Streamlit components */
    .stTextArea *, .stTextInput *, .stSelectbox *, .stNumberInput *, .stDateInput *, .stTimeInput *,
    .stFileUploader *, .stColorPicker *, .stDataFrame *, .stMetric *, .stExpander *, .stAlert *,
    .stInfo *, .stWarning *, .stError *, .stSuccess *, .stProgress *, .stCheckbox *, .stRadio *,
    .stMultiSelect *, .stSlider * {
        color: #000000 !important;
    }

    /* Restore white text for all elements except inputs and selectboxes */
    .stMarkdown, .stDataFrame, .stMetric, .stExpander, .stAlert, .stInfo, .stWarning, 
    .stError, .stSuccess, .stProgress, .stCheckbox, .stRadio, .stMultiSelect, .stSlider,
    .stMarkdown *, .stDataFrame *, .stMetric *, .stExpander *, .stAlert *, .stInfo *, 
    .stWarning *, .stError *, .stSuccess *, .stProgress *, .stCheckbox *, .stRadio *, 
    .stMultiSelect *, .stSlider * {
        color: #FFFFFF !important;
    }

    /* Keep black text only for input elements and selectboxes */
    .stTextArea textarea,
    .stTextArea div[data-testid="stTextArea"] *,
    .stTextInput input,
    .stTextInput div[data-testid="stTextInput"] *,
    .stSelectbox select,
    .stSelectbox div[data-testid="stSelectbox"] *,
    .stSelectbox div[data-baseweb="select"] *,
    .stSelectbox div[data-baseweb="popover"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] *,
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] *,
    .stSelectbox div[data-baseweb="select"] div[role="option"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="true"] *,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] span,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] div,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] p,
    .stSelectbox div[data-baseweb="select"] div[aria-selected="false"] {
        color: #000000 !important;
    }

    /* Override Streamlit's default text colors for these sections */
    div[data-testid="stTextArea"] *,
    div[data-testid="stTextInput"] *,
    div[data-testid="stSelectbox"] * {
        color: #000000 !important;
    }

    /* Force black text for all input elements */
    input, textarea, select, option {
        color: #000000 !important;
    }

    /* Force black text for summary boxes and their children */
    .summary-box, .summary-box * {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- Rate Limiter -----------------
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()
            if len(self.requests) >= self.max_requests:
                return False
            self.requests.append(now)
            return True

rate_limiter = RateLimiter(max_requests=15, time_window=60)

# ----------------- Gemini API Config -----------------
genai.configure(api_key="AIzaSyA1mWUjRsSeyjeOn7F_1aPtm2KOb3j8K3w")

# ----------------- Local Fallback Mode -----------------
def local_summarize_events(events_df):
    event_types = events_df['event_type'].value_counts()
    summary = f"Found {len(events_df)} events with {len(event_types)} different types.\n"
    summary += "Most common event types:\n"
    for event_type, count in event_types.head(3).items():
        summary += f"- {event_type}: {count} events\n"
    return summary

def local_explain_similarity(event1, event2, similarity_score):
    return f"Events have a similarity score of {similarity_score:.4f}. These events are {'very similar' if similarity_score > 0.8 else 'somewhat similar' if similarity_score > 0.5 else 'not very similar'}."

def local_generate_event_report(events_df):
    report = f"Event Report Summary:\n"
    report += f"Total Events: {len(events_df)}\n"
    report += f"Date Range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}\n"
    report += f"Unique Event Types: {events_df['event_type'].nunique()}\n"
    return report

# ----------------- Error Handling & Caching -----------------
def handle_api_error():
    return "Sorry, the AI service is currently unavailable. Please try again later or switch to local mode."

@lru_cache(maxsize=200)
def cached_generate_content(prompt, max_retries=3):
    if not rate_limiter.acquire():
        return handle_api_error()
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(prompt, generation_config={
                "temperature": 0.3,  # Lower temperature for faster, more focused responses
                "max_output_tokens": 500,  # Limit response length
                "top_p": 0.8,
                "top_k": 40
            })
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"AI service error: {str(e)}")
                return handle_api_error()
            time.sleep(2 ** attempt)

# ----------------- Gemini Helpers -----------------
def summarize_events(events_df):
    if st.session_state.get('use_local_mode', False):
        return local_summarize_events(events_df)
    prompt = f"""Summarize the following manufacturing events in plain English:\n{events_df.to_string(index=False)}"""
    return cached_generate_content(prompt)

def explain_similarity(event1, event2, similarity_score):
    if st.session_state.get('use_local_mode', False):
        return local_explain_similarity(event1, event2, similarity_score)
    prompt = f"""
Two manufacturing events have a cosine similarity of {similarity_score:.4f}.
Event 1: {event1.to_dict()}
Event 2: {event2.to_dict()}
Explain in plain English what this similarity might indicate.
"""
    return cached_generate_content(prompt)

def generate_event_report(events_df):
    if st.session_state.get('use_local_mode', False):
        return local_generate_event_report(events_df)
    prompt = f"""Create a high-level summary report of the following manufacturing events dataset:\n{events_df.to_string(index=False)}"""
    return cached_generate_content(prompt)

# ----------------- DB Connection -----------------
@st.cache_resource
def get_connection():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5132"
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Database connection error: {str(e)}")
        st.error("Please check if PostgreSQL is running and the connection details are correct.")
        raise
    except Exception as e:
        st.error(f"Unexpected error while connecting to database: {str(e)}")
        raise

# ----------------- DB Helpers -----------------
def fetch_df(conn, query, params=None):
    return pd.read_sql(query, conn, params=params)

def get_city_for_event(event_id):
    """Assign a city based on event ID for demonstration"""
    cities = ["Delhi", "Mumbai", "Pune", "Hyderabad", "Bengaluru"]
    return cities[event_id % len(cities)]

def get_event_statistics(conn):
    query = """
        SELECT 
            event_type,
            COUNT(*) as frequency,
            MIN(timestamp) as first_occurrence,
            MAX(timestamp) as last_occurrence,
            AVG(duration_minutes) as avg_duration,
            MAX(duration_minutes) as max_duration,
            MIN(duration_minutes) as min_duration
        FROM manufacturing_events
        GROUP BY event_type
        ORDER BY frequency DESC;
    """
    df = fetch_df(conn, query)
    # Add city information
    df['city'] = df.index.map(lambda x: get_city_for_event(x))
    return df

def get_machine_statistics(conn):
    query = """
        SELECT 
            machine_name,
            COUNT(*) as total_events,
            COUNT(DISTINCT event_type) as unique_event_types,
            AVG(duration_minutes) as avg_duration,
            MAX(duration_minutes) as max_duration,
            MIN(duration_minutes) as min_duration
        FROM manufacturing_events
        GROUP BY machine_name
        ORDER BY total_events DESC;
    """
    df = fetch_df(conn, query)
    return df

def get_recent_events(conn, limit=100):
    query = """
        SELECT 
            id,
            event_type,
            machine_name,
            notes,
            timestamp,
            duration_minutes,
            embedding
        FROM manufacturing_events
        ORDER BY timestamp DESC
        LIMIT %s;
    """
    df = fetch_df(conn, query)
    # Add city information
    df['city'] = df['id'].map(get_city_for_event)
    return df

def execute_custom_query(conn, query):
    try:
        # Basic query validation
        if not query.strip().upper().startswith('SELECT'):
            return None, "Only SELECT queries are allowed for security reasons."
        
        # Execute the query
        df = fetch_df(conn, query)
        return df, None
    except Exception as e:
        return None, str(e)

def check_database_structure(conn):
    try:
        with conn.cursor() as cur:
            # Check if the table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'manufacturing_events'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                return "Table 'manufacturing_events' does not exist"
            
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'manufacturing_events';
            """)
            columns = cur.fetchall()
            
            # Check if vector extension is installed
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """)
            vector_ext_exists = cur.fetchone()[0]
            
            return {
                "table_exists": table_exists,
                "columns": columns,
                "vector_extension": vector_ext_exists
            }
    except Exception as e:
        return f"Error checking database: {str(e)}"

def get_similar_events(conn, event_id):
    try:
        # First check if the event exists and get its embedding
        check_query = """
            SELECT id, embedding 
            FROM manufacturing_events 
            WHERE id = %s;
        """
        with conn.cursor() as cur:
            cur.execute(check_query, (event_id,))
            result = cur.fetchone()
            
            if not result:
                return None, f"Event ID {event_id} not found"
            
            # Now find similar events using cosine similarity
            query = """
                WITH event_embedding AS (
                    SELECT embedding 
                    FROM manufacturing_events 
                    WHERE id = %s
                )
                SELECT 
                    m.id,
                    m.event_type,
                    m.machine_name,
                    m.notes as event_description,
                    m.embedding,
                    1 - (m.embedding <=> e.embedding) as cosine_similarity
                FROM manufacturing_events m
                CROSS JOIN event_embedding e
                WHERE m.id != %s
                ORDER BY m.embedding <=> e.embedding
                LIMIT 10;
            """
            return fetch_df(conn, query, (event_id, event_id)), None
    except Exception as e:
        return None, f"Error in similarity search: {str(e)}"

def get_similar_by_embedding(conn, embedding, event_type=None):
    if event_type:
        query = """
            SELECT * FROM manufacturing_events
            WHERE event_type = %s
            ORDER BY embedding <-> %s::vector
            LIMIT 5;
        """
        return fetch_df(conn, query, (event_type, embedding))
    else:
        query = """
            SELECT * FROM manufacturing_events
            ORDER BY embedding <-> %s::vector
            LIMIT 5;
        """
        return fetch_df(conn, query, (embedding,))

def get_event_details(conn, event_id):
    query = """
        SELECT id, event_type, machine_name, notes, embedding
        FROM manufacturing_events
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (event_id,))
        result = cur.fetchone()
        if result:
            return {
                'id': result[0],
                'event_type': result[1],
                'machine_name': result[2],
                'notes': result[3],
                'embedding': result[4]
            }
    return None

def get_event_embedding(conn, event_id):
    query = """
        SELECT id, event_type, machine_name, notes, embedding
        FROM manufacturing_events
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (event_id,))
        result = cur.fetchone()
        if result:
            return {
                'id': result[0],
                'event_type': result[1],
                'machine_name': result[2],
                'notes': result[3],
                'embedding': result[4]
            }
    return None

def get_similar_events_by_embedding(conn, embedding, limit=5):
    query = """
        SELECT 
            id,
            event_type,
            machine_name,
            notes,
            embedding,
            1 - (embedding <=> %s::vector) as similarity
        FROM manufacturing_events
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    return fetch_df(conn, query, (embedding, embedding, limit))

def visualize_embeddings(conn):
    query = """
        SELECT 
            id,
            event_type,
            machine_name,
            notes,
            embedding
        FROM manufacturing_events
        WHERE embedding IS NOT NULL
        LIMIT 100
    """
    events = fetch_df(conn, query)
    return events

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embedding vectors"""
    try:
        # Convert string representation of embeddings to actual lists if needed
        if isinstance(embedding1, str):
            embedding1 = ast.literal_eval(embedding1)
        if isinstance(embedding2, str):
            embedding2 = ast.literal_eval(embedding2)
            
        # Convert to numpy arrays for vector operations
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cosine_similarity = dot_product / (norm1 * norm2)
        return float(cosine_similarity)
    except Exception as e:
        st.error(f"Error calculating cosine similarity: {str(e)}")
        return None

def get_similar_events_by_embedding(conn, embedding, limit=5):
    """Find similar events using the given embedding"""
    try:
        # First get all events with embeddings
        query = """
            SELECT 
                id,
                event_type,
                machine_name,
                notes as description,
                embedding
            FROM manufacturing_events
            WHERE embedding IS NOT NULL;
        """
        all_events = fetch_df(conn, query)
        
        if all_events.empty:
            return None
            
        # Calculate cosine similarity for each event
        similarities = []
        for _, event in all_events.iterrows():
            similarity = calculate_cosine_similarity(embedding, event['embedding'])
            if similarity is not None:
                similarities.append({
                    'id': event['id'],
                    'event_type': event['event_type'],
                    'machine_name': event['machine_name'],
                    'description': event['description'],
                    'similarity': similarity
                })
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similarities[:limit])
    except Exception as e:
        st.error(f"Error finding similar events: {str(e)}")
        return None

def visualize_embeddings(conn):
    df = fetch_df(conn, "SELECT id, embedding, event_type FROM manufacturing_events LIMIT 500")
    embeddings = df['embedding'].apply(ast.literal_eval).tolist()
    labels = df['event_type'].astype('category').cat.codes
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(embedding_2d[:,0], embedding_2d[:,1], c=labels, cmap='Spectral', s=10)
    plt.colorbar(scatter, label='Event Type Code')
    plt.title("UMAP Projection of Manufacturing Events")
    st.pyplot(plt)

def get_question_embedding(question):
    """Convert a question into an embedding vector using Gemini"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = f"""Convert this manufacturing question into a vector embedding.
        Question: {question}
        Return ONLY a Python list of 3 numbers between 0 and 1, like this: [0.1, 0.2, 0.3]"""
        
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.1,  # Very low temperature for consistent embeddings
            "max_output_tokens": 50,  # Very short response needed
            "top_p": 0.1,
            "top_k": 1
        })
        embedding_str = response.text.strip()
        
        # Clean and parse the response
        embedding_str = embedding_str.replace('\n', '').replace(' ', '')
        start_idx = embedding_str.find('[')
        end_idx = embedding_str.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            return [0.5, 0.5, 0.5]  # Default embedding if parsing fails
            
        embedding_str = embedding_str[start_idx:end_idx]
        try:
            embedding = ast.literal_eval(embedding_str)
            if not isinstance(embedding, list) or len(embedding) != 3:
                return [0.5, 0.5, 0.5]
            return [float(x) for x in embedding]
        except:
            return [0.5, 0.5, 0.5]
            
    except Exception as e:
        return [0.5, 0.5, 0.5]

# ----------------- Streamlit App -----------------
def main():
    conn = get_connection()
    
    # --- CONDITIONAL BACKGROUND CSS (must be first Streamlit call) ---
    import streamlit as st
    # Navigation menu
    menu = ["🏠 Home", "📄 All Events", "🔍 Similar Events", "📐 Cosine Similarity", "📊 Visualize Embeddings", "📝 Event Summary Report", "💬 Ask a Question", "🤖 Ask with Ragas", "🤖 AI Agent Assistant"]
    if 'nav_choice' not in st.session_state:
        st.session_state['nav_choice'] = menu[0]

    # Display navigation bar
    st.markdown("""
    <div class="nav-container">
        <div style="text-align: center;">
    """, unsafe_allow_html=True)
    cols = st.columns(len(menu))
    for i, item in enumerate(menu):
        with cols[i]:
            if st.button(item, key=f"nav_{i}"):
                st.session_state['nav_choice'] = item
    st.markdown("</div></div>", unsafe_allow_html=True)
    choice = st.session_state['nav_choice']

    # Inject background CSS before any content
    if choice == "🏠 Home":
        st.markdown(
            '''
            <style>
            .stApp {
                background-image: url('https://i.postimg.cc/0yN8ttJf/logo-for-KUMAR-FORGING-INDUSTRIES.jpg');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .main-header {
                text-align: center;
                color: #FFD600 !important;
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 20px;
                text-shadow: 0 0 10px #FFD600, 0 0 20px #FFD600, 2px 2px 4px rgba(0,0,0,0.7);
            }
            .sub-header {
                text-align: center;
                color: #FFFFFF;
                font-size: 24px;
                margin-bottom: 30px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .feature-box {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
            }
            .stats-box {
                background-color: rgba(45, 45, 45, 0.95);
                border: 1px solid #4CAF50;
                border-radius: 8px;
                padding: 15px;
                margin: 5px 0;
                backdrop-filter: blur(5px);
            }
            .kfi-logo {
                width: 260px;
                height: 260px;
                object-fit: cover;
                border: 6px solid #4CAF50;
                border-radius: 24px;
                box-shadow: 0 8px 32px rgba(76, 175, 80, 0.25), 0 2px 8px rgba(0,0,0,0.18);
                margin-bottom: 20px;
                background: #222;
                display: inline-block;
            }
            .nav-container {
                background-color: rgba(45, 45, 45, 0.95);
                padding: 10px 0;
                margin-bottom: 20px;
                border-bottom: 2px solid #4CAF50;
                backdrop-filter: blur(5px);
            }
            .nav-item {
                display: inline-block;
                padding: 10px 20px;
                color: #FFFFFF !important;
                background: rgba(45, 45, 45, 0.95) !important;
                border: 2px solid #4CAF50 !important;
                border-radius: 10px !important;
                margin: 0 5px;
                font-weight: bold;
                font-size: 18px;
                text-decoration: none;
                transition: filter 0.2s, background 0.2s, color 0.2s, border 0.2s;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .nav-item:hover, .nav-item.active {
                filter: brightness(1.15);
                color: #FFD600 !important;
                border-color: #FFD600 !important;
            }
            </style>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '''
            <style>
            .stApp {
                background-color: #1E1E1E !important;
                background-image: none !important;
            }
            .main-header {
                text-align: center;
                color: #FFD600 !important;
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 20px;
                text-shadow: 0 0 10px #FFD600, 0 0 20px #FFD600, 2px 2px 4px rgba(0,0,0,0.7);
            }
            .sub-header {
                text-align: center;
                color: #FFFFFF;
                font-size: 24px;
                margin-bottom: 30px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .feature-box {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
            }
            .stats-box {
                background-color: rgba(45, 45, 45, 0.95);
                border: 1px solid #4CAF50;
                border-radius: 8px;
                padding: 15px;
                margin: 5px 0;
                backdrop-filter: blur(5px);
            }
            .nav-container {
                background-color: rgba(45, 45, 45, 0.95);
                padding: 10px 0;
                margin-bottom: 20px;
                border-bottom: 2px solid #4CAF50;
                backdrop-filter: blur(5px);
            }
            .nav-item {
                display: inline-block;
                padding: 10px 20px;
                color: #FFFFFF !important;
                background: rgba(45, 45, 45, 0.95) !important;
                border: 2px solid #4CAF50 !important;
                border-radius: 10px !important;
                margin: 0 5px;
                font-weight: bold;
                font-size: 18px;
                text-decoration: none;
                transition: filter 0.2s, background 0.2s, color 0.2s, border 0.2s;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .nav-item:hover, .nav-item.active {
                filter: brightness(1.15);
                color: #FFD600 !important;
                border-color: #FFD600 !important;
            }
            .stMarkdown {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stDataFrame {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stMetric {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stExpander {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stTextInput, .stTextArea {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stButton button {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 10px 20px;
                margin: 10px 0;
                color: #FFFFFF !important;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #4CAF50;
                color: #FFFFFF !important;
                border-color: #FFD600;
            }
            .stSelectbox {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stSlider {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stCheckbox {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stProgress {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stAlert {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stInfo {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stWarning {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #FFD600;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stError {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #FF0000;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            .stSuccess {
                background-color: rgba(45, 45, 45, 0.95);
                border: 2px solid #00FF00;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                backdrop-filter: blur(5px);
                color: #FFFFFF !important;
            }
            /* Add styles for DataFrame elements */
            .stDataFrame div[data-testid="stDataFrame"] {
                color: #FFFFFF !important;
            }
            .stDataFrame div[data-testid="stDataFrame"] table {
                color: #FFFFFF !important;
            }
            .stDataFrame div[data-testid="stDataFrame"] th {
                color: #FFFFFF !important;
                background-color: rgba(76, 175, 80, 0.2) !important;
            }
            .stDataFrame div[data-testid="stDataFrame"] td {
                color: #FFFFFF !important;
            }
            /* Add styles for metric elements */
            .stMetric div[data-testid="stMetricValue"] {
                color: #FFFFFF !important;
            }
            .stMetric div[data-testid="stMetricLabel"] {
                color: #FFFFFF !important;
            }
            /* Add styles for expander elements */
            .stExpander div[data-testid="stExpander"] {
                color: #FFFFFF !important;
            }
            .stExpander div[data-testid="stExpander"] summary {
                color: #FFFFFF !important;
            }
            /* Add styles for selectbox elements */
            .stSelectbox div[data-testid="stSelectbox"] {
                color: #FFFFFF !important;
            }
            .stSelectbox div[data-testid="stSelectbox"] select {
                color: #FFFFFF !important;
                background-color: rgba(45, 45, 45, 0.95) !important;
            }
            /* Add styles for slider elements */
            .stSlider div[data-testid="stSlider"] {
                color: #FFFFFF !important;
            }
            .stSlider div[data-testid="stSlider"] input {
                color: #FFFFFF !important;
            }
            /* Add styles for checkbox elements */
            .stCheckbox div[data-testid="stCheckbox"] {
                color: #FFFFFF !important;
            }
            .stCheckbox div[data-testid="stCheckbox"] label {
                color: #FFFFFF !important;
            }
            /* Add styles for progress elements */
            .stProgress div[data-testid="stProgress"] {
                color: #FFFFFF !important;
            }
            /* Add styles for alert elements */
            .stAlert div[data-testid="stAlert"] {
                color: #FFFFFF !important;
            }
            /* Add styles for info elements */
            .stInfo div[data-testid="stInfo"] {
                color: #FFFFFF !important;
            }
            /* Add styles for warning elements */
            .stWarning div[data-testid="stWarning"] {
                color: #FFFFFF !important;
            }
            /* Add styles for error elements */
            .stError div[data-testid="stError"] {
                color: #FFFFFF !important;
            }
            /* Add styles for success elements */
            .stSuccess div[data-testid="stSuccess"] {
                color: #FFFFFF !important;
            }
            /* Add styles for text input elements */
            .stTextInput div[data-testid="stTextInput"] {
                color: #FFFFFF !important;
            }
            .stTextInput div[data-testid="stTextInput"] input {
                color: #FFFFFF !important;
                background-color: rgba(45, 45, 45, 0.95) !important;
            }
            /* Add styles for text area elements */
            .stTextArea div[data-testid="stTextArea"] {
                color: #FFFFFF !important;
            }
            .stTextArea div[data-testid="stTextArea"] textarea {
                color: #FFFFFF !important;
                background-color: rgba(45, 45, 45, 0.95) !important;
            }
            </style>
            ''',
            unsafe_allow_html=True
        )

    # Local mode toggle in a small sidebar
    with st.sidebar:
        st.toggle("Use Local Mode (No API)", key="use_local_mode", help="Switch to local mode to avoid API quota limits")

    if choice == "🏠 Home":
        # Main header with custom logo in a feature box for visibility
        st.markdown("""
        <style>
        .kfi-logo {
            width: 260px;
            height: 260px;
            object-fit: cover;
            border: 6px solid #4CAF50;
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(76, 175, 80, 0.25), 0 2px 8px rgba(0,0,0,0.18);
            margin-bottom: 20px;
            background: #222;
            display: inline-block;
        }
        </style>
        <div class='feature-box' style='text-align: center; margin-bottom: 30px;'>
            <img src='https://i.postimg.cc/0yN8ttJf/logo-for-KUMAR-FORGING-INDUSTRIES.jpg' class='kfi-logo'>
            <h1 class='main-header'>Kumar Forging Industries</h1>
            <p class='sub-header'>Advanced Manufacturing Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        # Main content in columns
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class='feature-box'>
                <div class='feature-title'>🏭 Smart Manufacturing Solutions</div>
                <div class='feature-content'>
                    Our advanced manufacturing analytics platform provides real-time insights into your forging operations:
                    <ul>
                        <li>Real-time event monitoring and analysis</li>
                        <li>Predictive maintenance insights</li>
                        <li>Quality control automation</li>
                        <li>Performance optimization</li>
                    </ul>
                </div>
            </div>
            
            <div class='feature-box'>
                <div class='feature-title'>📊 Key Features</div>
                <div class='feature-content'>
                    <ul>
                        <li>Event Analysis & Pattern Recognition</li>
                        <li>Machine Performance Monitoring</li>
                        <li>Quality Control Analytics</li>
                        <li>AI-Powered Insights</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='stats-box'>
                <div class='stats-title'>Active Machines</div>
                <div class='stats-value'>8</div>
            </div>
            
            <div class='stats-box'>
                <div class='stats-title'>Production Cities</div>
                <div class='stats-value'>5</div>
            </div>
            
            <div class='stats-box'>
                <div class='stats-title'>Event Types</div>
                <div class='stats-value'>12+</div>
            </div>
            
            <div class='stats-box'>
                <div class='stats-title'>Real-time Monitoring</div>
                <div class='stats-value'>24/7</div>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://www.industryweek.com/sites/industryweek.com/files/styles/article_featured_retina/public/uploads/2019/11/smart-manufacturing.jpg", 
                    caption="Smart Manufacturing in Industry 4.0")
        st.markdown("""
        <div class='feature-box'>
            <div class='feature-title'>🚀 Getting Started</div>
            <div class='feature-content'>
                Use the sidebar to navigate through different analysis tools:
                <ul>
                    <li>View all manufacturing events</li>
                    <li>Analyze machine performance</li>
                    <li>Monitor quality metrics</li>
                    <li>Get AI-powered insights</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif choice == "📄 All Events":
        st.subheader("All Manufacturing Events")
        
        # Get all events data
        all_events = fetch_df(conn, """
            SELECT 
                id,
                event_type,
                machine_name,
                notes,
                timestamp,
                duration_minutes,
                embedding
            FROM manufacturing_events 
            ORDER BY timestamp DESC
        """)
        
        # Add city information
        all_events['city'] = all_events['id'].map(get_city_for_event)
        
        # Reorder columns to put machine_name second
        cols = all_events.columns.tolist()
        cols.remove('machine_name')
        cols.insert(1, 'machine_name')
        all_events = all_events[cols]
        
        # Display summary statistics
        st.markdown("### 📊 Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", len(all_events))
            st.metric("Unique Event Types", all_events['event_type'].nunique())
        with col2:
            st.metric("Unique Machines", all_events['machine_name'].nunique())
            st.metric("Unique Cities", all_events['city'].nunique())
        with col3:
            st.metric("Avg Duration (min)", f"{all_events['duration_minutes'].mean():.2f}")
            st.metric("Max Duration (min)", f"{all_events['duration_minutes'].max():.2f}")
        
        # Add filters
        st.markdown("### 🔍 Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_city = st.selectbox("Select City", ["All"] + sorted(all_events['city'].unique().tolist()))
        with col2:
            selected_event_type = st.selectbox("Select Event Type", ["All"] + sorted(all_events['event_type'].unique().tolist()))
        with col3:
            selected_machine = st.selectbox("Select Machine", ["All"] + sorted(all_events['machine_name'].unique().tolist()))
        
        # Apply filters
        filtered_events = all_events.copy()
        if selected_city != "All":
            filtered_events = filtered_events[filtered_events['city'] == selected_city]
        if selected_event_type != "All":
            filtered_events = filtered_events[filtered_events['event_type'] == selected_event_type]
        if selected_machine != "All":
            filtered_events = filtered_events[filtered_events['machine_name'] == selected_machine]
        
        # Display filtered results
        st.markdown(f"### 📋 Events ({len(filtered_events)} found)")
        st.dataframe(filtered_events)
        
        # Display events grouped by city
        st.markdown("### 📍 Events by City")
        for city in sorted(all_events['city'].unique()):
            city_events = all_events[all_events['city'] == city]
            st.markdown(f"#### 📍 {city} ({len(city_events)} events)")
            
            # Calculate city-specific statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Event Types", city_events['event_type'].nunique())
            with col2:
                st.metric("Machines", city_events['machine_name'].nunique())
            with col3:
                st.metric("Avg Duration", f"{city_events['duration_minutes'].mean():.2f} min")
            
            # Display city events
            st.dataframe(city_events)
            st.markdown("---")
        
        # Add event type analysis
        st.markdown("### 📊 Event Type Analysis")
        event_type_stats = all_events.groupby('event_type').agg({
            'id': 'count',
            'duration_minutes': ['mean', 'min', 'max'],
            'machine_name': 'nunique',
            'city': 'nunique'
        }).round(2)
        
        event_type_stats.columns = ['Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Unique Machines', 'Unique Cities']
        st.dataframe(event_type_stats)
        
        # Add machine analysis
        st.markdown("### 🏭 Machine Analysis")
        machine_stats = all_events.groupby('machine_name').agg({
            'id': 'count',
            'duration_minutes': ['mean', 'min', 'max'],
            'event_type': 'nunique',
            'city': 'nunique'
        }).round(2)
        
        machine_stats.columns = ['Event Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Unique Event Types', 'Unique Cities']
        st.dataframe(machine_stats)
        
        # Add duration analysis
        st.markdown("### ⏱️ Duration Analysis")
        duration_stats = all_events['duration_minutes'].describe().round(2)
        st.dataframe(duration_stats)
        
        # Add download option
        st.markdown("### 💾 Download Data")
        csv = all_events.to_csv(index=False)
        st.download_button(
            label="Download Complete Dataset",
            data=csv,
            file_name="manufacturing_events.csv",
            mime="text/csv"
        )

    elif choice == "🔍 Similar Events":
        st.subheader("Find Similar Events")
        
        # Add database structure check
        db_info = check_database_structure(conn)
        if isinstance(db_info, str):
            st.error(db_info)
        else:
            if not db_info["vector_extension"]:
                st.error("Vector extension is not installed in the database. Please install the 'vector' extension.")
            elif not db_info["table_exists"]:
                st.error("Table 'manufacturing_events' does not exist.")
            else:
                # Show table structure in expander
                with st.expander("Database Structure"):
                    st.write("Columns:", db_info["columns"])
                    st.write("Vector Extension:", "Installed" if db_info["vector_extension"] else "Not Installed")
        
        event_id = st.number_input("Enter Event ID (e.g., 333)", min_value=1)
        if st.button("Find Similar"):
            try:
                df, error = get_similar_events(conn, event_id)
                if error:
                    st.error(error)
                elif df is not None and not df.empty:
                    st.success(f"Found {len(df)} similar events to Event ID {event_id}")
                    
                    # Display the reference event first
                    ref_event = fetch_df(conn, f"SELECT * FROM manufacturing_events WHERE id = {event_id}")
                    if not ref_event.empty:
                        ref_event['city'] = ref_event['id'].map(get_city_for_event)
                        # Reorder columns to put machine_name second
                        cols = ref_event.columns.tolist()
                        cols.remove('machine_name')
                        cols.insert(1, 'machine_name')
                        ref_event = ref_event[cols]
                        
                        st.markdown("### Reference Event")
                        st.markdown(f"""
                        <div class="event-details">
                            <p><strong>ID:</strong> {ref_event.iloc[0]['id']}</p>
                            <p><strong>Machine:</strong> {ref_event.iloc[0]['machine_name']}</p>
                            <p><strong>Type:</strong> {ref_event.iloc[0]['event_type']}</p>
                            <p><strong>Description:</strong> {ref_event.iloc[0]['notes']}</p>
                            <p><strong>City:</strong> {ref_event.iloc[0]['city']}</p>
                            <p><strong>Embedding:</strong> {ref_event.iloc[0]['embedding']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display similar events with similarity scores
                    st.markdown("### Similar Events (Ordered by Similarity)")
                    df['city'] = df['id'].map(get_city_for_event)
                    # Reorder columns to put machine_name second
                    cols = df.columns.tolist()
                    cols.remove('machine_name')
                    cols.insert(1, 'machine_name')
                    df = df[cols]
                    
                    for _, row in df.iterrows():
                        st.markdown(f"""
                        <div class="similarity-box">
                            <p><strong>Event ID:</strong> {row['id']}</p>
                            <p><strong>Machine:</strong> {row['machine_name']}</p>
                            <p><strong>Type:</strong> {row['event_type']}</p>
                            <p><strong>Description:</strong> {row['event_description']}</p>
                            <p><strong>City:</strong> {row['city']}</p>
                            <p class="similarity-score">Similarity Score: {row['cosine_similarity']*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show raw data in expander
                    with st.expander("View Raw Data with Embeddings"):
                        display_df = df.drop('machine_id', axis=1)
                        st.dataframe(display_df)
                else:
                    st.warning(f"No similar events found for Event ID {event_id}")
            except Exception as e:
                st.error(f"Error finding similar events: {str(e)}")

    elif choice == "📐 Cosine Similarity":
        st.subheader("Compare Two Events")
        col1, col2 = st.columns(2)
        
        with col1:
            id1 = st.number_input("Event ID 1", min_value=1)
            if id1:
                # Verify event exists and has embedding
                verify_query = """
                    SELECT id, event_type, machine_name, notes, embedding IS NOT NULL as has_embedding
                    FROM manufacturing_events
                    WHERE id = %s;
                """
                event1 = fetch_df(conn, verify_query, (id1,))
                if not event1.empty:
                    st.markdown("### Event 1 Details")
                    st.markdown(f"""
                    <div class="event-details">
                        <p><strong>ID:</strong> {event1.iloc[0]['id']}</p>
                        <p><strong>Machine:</strong> {event1.iloc[0]['machine_name']}</p>
                        <p><strong>Type:</strong> {event1.iloc[0]['event_type']}</p>
                        <p><strong>Description:</strong> {event1.iloc[0]['notes']}</p>
                        <p><strong>Has Embedding:</strong> {'Yes' if event1.iloc[0]['has_embedding'] else 'No'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Event {id1} not found")
        
        with col2:
            id2 = st.number_input("Event ID 2", min_value=1)
            if id2:
                # Verify event exists and has embedding
                verify_query = """
                    SELECT id, event_type, machine_name, notes, embedding IS NOT NULL as has_embedding
                    FROM manufacturing_events
                    WHERE id = %s;
                """
                event2 = fetch_df(conn, verify_query, (id2,))
                if not event2.empty:
                    st.markdown("### Event 2 Details")
                    st.markdown(f"""
                    <div class="event-details">
                        <p><strong>ID:</strong> {event2.iloc[0]['id']}</p>
                        <p><strong>Machine:</strong> {event2.iloc[0]['machine_name']}</p>
                        <p><strong>Type:</strong> {event2.iloc[0]['event_type']}</p>
                        <p><strong>Description:</strong> {event2.iloc[0]['notes']}</p>
                        <p><strong>Has Embedding:</strong> {'Yes' if event2.iloc[0]['has_embedding'] else 'No'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Event {id2} not found")
        
        if st.button("Compare Events"):
            if id1 and id2:
                if id1 == id2:
                    st.warning("Please select two different events to compare")
                else:
                    # First verify both events have embeddings
                    verify_query = """
                        SELECT id, event_type, embedding IS NOT NULL as has_embedding
                        FROM manufacturing_events
                        WHERE id IN (%s, %s);
                    """
                    verification = fetch_df(conn, verify_query, (id1, id2))
                    
                    if len(verification) != 2:
                        st.error("One or both events not found in database")
                    elif not all(verification['has_embedding']):
                        missing_embeddings = verification[~verification['has_embedding']]
                        st.error(f"Events {', '.join(missing_embeddings['id'].astype(str))} do not have embeddings")
                    else:
                        # Calculate similarity
                        query = """
                            WITH event_embeddings AS (
                                SELECT id, embedding 
                                FROM manufacturing_events 
                                WHERE id IN (%s, %s)
                            )
                            SELECT 
                                e1.id as event1_id,
                                e2.id as event2_id,
                                e1.embedding as embedding1,
                                e2.embedding as embedding2,
                                1 - (e1.embedding <=> e2.embedding) as cosine_similarity
                            FROM event_embeddings e1
                            CROSS JOIN event_embeddings e2
                            WHERE e1.id = %s AND e2.id = %s;
                        """
                        df = fetch_df(conn, query, (id1, id2, id1, id2))
                        
                        if not df.empty:
                            similarity = df.iloc[0]["cosine_similarity"]
                            st.markdown("### Comparison Result")
                            st.markdown(f"""
                            <div class="similarity-box">
                                <p class="similarity-score">Cosine Similarity: {similarity*100:.2f}%</p>
                                <p><strong>Interpretation:</strong> {
                                    'Very Similar' if similarity > 0.8 
                                    else 'Somewhat Similar' if similarity > 0.5 
                                    else 'Not Very Similar'
                                }</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show raw comparison data
                            with st.expander("View Raw Comparison Data"):
                                st.dataframe(df)
                        else:
                            st.warning("Could not calculate similarity between these events")
            else:
                st.warning("Please enter both event IDs")

    elif choice == "📊 Visualize Embeddings":
        st.subheader("UMAP Visualization of Event Embeddings")
        visualize_embeddings(conn)

    elif choice == "📝 Event Summary Report":
        st.subheader("AI Summary of All Recent Events")
        df = fetch_df(conn, "SELECT * FROM manufacturing_events ORDER BY id DESC LIMIT 100")
        st.dataframe(df)

    elif choice == "💬 Ask a Question":
        st.subheader("Ask a Question About Manufacturing Events")
        user_query = st.text_area("Enter your question", height=100)
        
        if st.button("Get Answer"):
            if user_query:
                try:
                    # Get ALL events data with all columns
                    all_events = fetch_df(conn, """
                        SELECT 
                            id,
                            event_type,
                            machine_name,
                            notes,
                            timestamp,
                            duration_minutes,
                            embedding
                        FROM manufacturing_events 
                        ORDER BY timestamp DESC
                    """)
                    
                    # Add city information
                    all_events['city'] = all_events['id'].map(get_city_for_event)
                    
                    # Extract event IDs from the question if present
                    event_ids = []
                    if "event" in user_query.lower():
                        words = user_query.split()
                        for i, word in enumerate(words):
                            if word.isdigit():
                                event_ids.append(int(word))
                    
                    # If specific events are mentioned, get their details
                    events_data = []
                    if event_ids:
                        # Get list of available event IDs
                        available_ids = all_events['id'].tolist()
                        
                        for event_id in event_ids:
                            if event_id in available_ids:
                                event_data = all_events[all_events['id'] == event_id]
                                events_data.append(event_data.iloc[0].to_dict())
                            else:
                                st.error(f"Event ID {event_id} not found in the database.")
                                # Show some available event IDs
                                st.info(f"Available event IDs: {', '.join(map(str, available_ids[:5]))} ...")
                                st.info("Please try again with one of the available event IDs.")
                                return
                    
                    # Create a focused context with explicit column names
                    if events_data:
                        context = "Specific Events Mentioned:\n"
                        for event in events_data:
                            context += f"""
                            Event Details:
                            - Event ID: {event['id']}
                            - Machine Name: {event['machine_name']}
                            - Event Type: {event['event_type']}
                            - Description: {event['notes']}
                            - City: {event['city']}
                            - Duration: {event['duration_minutes']} minutes
                            - Timestamp: {event['timestamp']}
                            """
                    else:
                        # If no specific events mentioned, use all events
                        context = "Complete Events Dataset:\n"
                        context += f"Total number of events: {len(all_events)}\n"
                        context += f"Date range: from {all_events['timestamp'].min()} to {all_events['timestamp'].max()}\n"
                        context += f"Number of unique event types: {all_events['event_type'].nunique()}\n"
                        context += f"Number of unique machines: {all_events['machine_name'].nunique()}\n"
                        context += f"Number of unique cities: {all_events['city'].nunique()}\n"
                        context += f"Average event duration: {all_events['duration_minutes'].mean():.2f} minutes\n"
                        context += f"Maximum event duration: {all_events['duration_minutes'].max():.2f} minutes\n"
                        context += f"Minimum event duration: {all_events['duration_minutes'].min():.2f} minutes\n\n"
                        
                        # Add machine-specific statistics
                        context += "Machine-wise Statistics:\n"
                        for machine in sorted(all_events['machine_name'].unique()):
                            machine_events = all_events[all_events['machine_name'] == machine]
                            context += f"""
                            {machine}:
                            - Number of events: {len(machine_events)}
                            - Unique event types: {machine_events['event_type'].nunique()}
                            - Cities involved: {machine_events['city'].nunique()}
                            - Average duration: {machine_events['duration_minutes'].mean():.2f} minutes
                            - Maximum duration: {machine_events['duration_minutes'].max():.2f} minutes
                            - Minimum duration: {machine_events['duration_minutes'].min():.2f} minutes
                            """
                        
                        # Add city-specific statistics
                        context += "\nCity-wise Statistics:\n"
                        for city in sorted(all_events['city'].unique()):
                            city_events = all_events[all_events['city'] == city]
                            context += f"""
                            {city}:
                            - Number of events: {len(city_events)}
                            - Unique event types: {city_events['event_type'].nunique()}
                            - Unique machines: {city_events['machine_name'].nunique()}
                            - Average duration: {city_events['duration_minutes'].mean():.2f} minutes
                            - Maximum duration: {city_events['duration_minutes'].max():.2f} minutes
                            - Minimum duration: {city_events['duration_minutes'].min():.2f} minutes
                            """
                        
                        # Add event type statistics
                        context += "\nEvent Type Statistics:\n"
                        event_type_stats = all_events.groupby('event_type').agg({
                            'id': 'count',
                            'duration_minutes': ['mean', 'min', 'max'],
                            'machine_name': 'nunique',
                            'city': 'nunique'
                        }).round(2)
                        
                        for event_type, stats in event_type_stats.iterrows():
                            context += f"""
                            {event_type}:
                            - Count: {stats[('id', 'count')]}
                            - Average duration: {stats[('duration_minutes', 'mean')]:.2f} minutes
                            - Unique machines: {stats[('machine_name', 'nunique')]}
                            - Cities involved: {stats[('city', 'nunique')]}
                            """
                    
                    # Generate answer with optimized prompt
                    prompt = f"""Based on this manufacturing data, answer: {user_query}

                    Available Data Columns:
                    - Event ID: Unique identifier for each event
                    - Machine Name: Name of the manufacturing machine
                    - Event Type: Type of manufacturing event (e.g., breakdown, production)
                    - Description: Detailed description of the event
                    - City: Location of the event
                    - Duration: Duration of the event in minutes
                    - Timestamp: When the event occurred

                    Data:
                    {context}

                    For event type specific questions (e.g., breakdown, production):
                    1. First, filter events by the specific event type
                    2. Show ALL individual event details for the filtered events:
                       - Event ID
                       - Machine Name
                       - City
                       - Timestamp
                       - Description
                       - Duration
                    3. Then provide aggregate statistics for the filtered events:
                       - Total number of events
                       - Machine distribution
                       - City distribution
                       - Duration statistics (min, max, avg)
                       - Time-based patterns
                    4. Compare with:
                       - Other event types
                       - Overall statistics
                    5. Include specific event details in the response

                    For duration-related questions:
                    1. Analyze ALL events' durations
                    2. Consider the complete distribution of durations
                    3. Include statistics like:
                       - Average duration across all events
                       - Maximum and minimum durations
                       - Duration patterns by machine, city, or event type
                       - Number of events exceeding certain duration thresholds
                    4. If the question mentions specific machines or event types, focus on those while still considering the complete duration data

                    For notes/description analysis:
                    1. Analyze the content of event descriptions
                    2. Group similar descriptions
                    3. Identify common patterns or issues
                    4. Provide frequency of different types of descriptions
                    5. Include specific examples from the notes

                    For machine-city specific questions:
                    1. First, filter events by both machine AND city
                    2. Show ALL individual event details for the filtered events:
                       - Event ID
                       - Timestamp
                       - Description
                       - Event Type
                       - Duration
                    3. Then provide aggregate statistics for the filtered events:
                       - Total number of events
                       - Event type distribution
                       - Duration statistics
                       - Time-based patterns
                    4. Compare with:
                       - Same machine in other cities
                       - Other machines in the same city
                       - Overall statistics
                    5. Include specific event details in the response

                    For machine or city-specific questions:
                    1. Analyze ALL events for the specified machine/city
                    2. Consider ALL columns in the analysis:
                       - Event types and their frequencies
                       - Duration patterns
                       - Timestamp patterns
                       - Description patterns
                    3. Provide detailed breakdowns:
                       - Event type distribution
                       - Duration statistics (min, max, avg)
                       - Time-based patterns
                       - Common event descriptions
                    4. Compare with other machines/cities when relevant
                    5. Include specific event details when available

                    Always include:
                    1. Raw data tables showing the relevant events
                    2. Statistical summaries
                    3. Visual comparisons where applicable
                    4. Specific examples from the data
                    5. Clear breakdowns by all relevant dimensions (machine, city, event type, duration)

                    Provide a comprehensive answer using the complete dataset. Include relevant statistics and patterns if applicable. If the question asks about specific events, focus on those events. If it's a general question, use the complete dataset to provide a thorough answer."""
                    
                    with st.spinner("Generating answer..."):
                        answer = cached_generate_content(prompt)
                    
                    # Display the answer
                    st.markdown(f"""
                    <div class="answer-box">
                        <h3>🧠 Answer</h3>
                        <div class="answer-content">
                            {answer}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show supporting data in expanders
                    with st.expander("Supporting Data"):
                        if events_data:
                            display_df = pd.DataFrame(events_data)
                            # Reorder columns to put machine_name second
                            cols = display_df.columns.tolist()
                            cols.remove('machine_name')
                            cols.insert(1, 'machine_name')
                            display_df = display_df[cols]
                            st.dataframe(display_df)
                        else:
                            # Extract event type, machine, and city from the question if mentioned
                            question_lower = user_query.lower()
                            mentioned_event_type = None
                            mentioned_machine = None
                            mentioned_city = None
                            
                            # Check for event type mentions
                            for event_type in all_events['event_type'].unique():
                                if event_type.lower() in question_lower:
                                    mentioned_event_type = event_type
                                    break
                            
                            # Check for machine mentions
                            for machine in all_events['machine_name'].unique():
                                if machine.lower() in question_lower:
                                    mentioned_machine = machine
                                    break
                            
                            # Check for city mentions
                            for city in all_events['city'].unique():
                                if city.lower() in question_lower:
                                    mentioned_city = city
                                    break
                            
                            # Show relevant data based on the question
                            if mentioned_event_type:
                                # Show event type specific data
                                event_type_events = all_events[all_events['event_type'] == mentioned_event_type]
                                st.markdown(f"### 📊 {mentioned_event_type} Events")
                                
                                # Show event type statistics
                                st.markdown(f"#### 📈 {mentioned_event_type} Statistics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Events", len(event_type_events))
                                    st.metric("Unique Machines", event_type_events['machine_name'].nunique())
                                with col2:
                                    st.metric("Unique Cities", event_type_events['city'].nunique())
                                    st.metric("Avg Duration", f"{event_type_events['duration_minutes'].mean():.2f} min")
                                with col3:
                                    st.metric("Max Duration", f"{event_type_events['duration_minutes'].max():.2f} min")
                                    st.metric("Min Duration", f"{event_type_events['duration_minutes'].min():.2f} min")
                                
                                # Show machine-wise breakdown
                                st.markdown(f"#### 🏭 {mentioned_event_type} by Machine")
                                machine_stats = event_type_events.groupby('machine_name').agg({
                                    'id': 'count',
                                    'duration_minutes': ['mean', 'min', 'max'],
                                    'city': 'nunique'
                                }).round(2)
                                machine_stats.columns = ['Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Cities']
                                st.dataframe(machine_stats)
                                
                                # Show city-wise breakdown
                                st.markdown(f"#### 📍 {mentioned_event_type} by City")
                                city_stats = event_type_events.groupby('city').agg({
                                    'id': 'count',
                                    'duration_minutes': ['mean', 'min', 'max'],
                                    'machine_name': 'nunique'
                                }).round(2)
                                city_stats.columns = ['Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Machines']
                                st.dataframe(city_stats)
                                
                                # Show duration analysis
                                st.markdown(f"#### ⏱️ {mentioned_event_type} Duration Analysis")
                                duration_stats = event_type_events['duration_minutes'].describe().round(2)
                                st.dataframe(duration_stats)
                                
                                # Show the complete event type data
                                st.markdown(f"#### 📋 All {mentioned_event_type} Events")
                                st.dataframe(event_type_events)
                            
                            if mentioned_machine and mentioned_city:
                                # Show machine-city specific data
                                machine_city_events = all_events[(all_events['machine_name'] == mentioned_machine) & 
                                                              (all_events['city'] == mentioned_city)]
                                st.markdown(f"### 📊 {mentioned_machine} in {mentioned_city}")
                                
                                # Show event type distribution
                                st.markdown(f"#### 📈 Event Type Distribution")
                                event_type_dist = machine_city_events['event_type'].value_counts()
                                st.dataframe(event_type_dist)
                                
                                # Show duration statistics
                                st.markdown(f"#### ⏱️ Duration Statistics")
                                duration_stats = machine_city_events['duration_minutes'].describe().round(2)
                                st.dataframe(duration_stats)
                                
                                # Show the complete data
                                st.dataframe(machine_city_events)
                                
                            elif mentioned_machine:
                                # Show machine-specific data
                                machine_events = all_events[all_events['machine_name'] == mentioned_machine]
                                st.markdown(f"### 📊 {mentioned_machine} Events")
                                
                                # Show event type distribution
                                st.markdown(f"#### 📈 Event Type Distribution")
                                event_type_dist = machine_events['event_type'].value_counts()
                                st.dataframe(event_type_dist)
                                
                                # Show city-wise breakdown
                                st.markdown(f"#### 📍 Events by City")
                                city_stats = machine_events.groupby('city').agg({
                                    'id': 'count',
                                    'duration_minutes': ['mean', 'min', 'max'],
                                    'event_type': 'nunique'
                                }).round(2)
                                city_stats.columns = ['Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Event Types']
                                st.dataframe(city_stats)
                                
                                # Show the complete data
                                st.dataframe(machine_events)
                                
                            elif mentioned_city:
                                # Show city-specific data
                                city_events = all_events[all_events['city'] == mentioned_city]
                                st.markdown(f"### 📊 Events in {mentioned_city}")
                                
                                # Show event type distribution
                                st.markdown(f"#### 📈 Event Type Distribution")
                                event_type_dist = city_events['event_type'].value_counts()
                                st.dataframe(event_type_dist)
                                
                                # Show machine-wise breakdown
                                st.markdown(f"#### 🏭 Events by Machine")
                                machine_stats = city_events.groupby('machine_name').agg({
                                    'id': 'count',
                                    'duration_minutes': ['mean', 'min', 'max'],
                                    'event_type': 'nunique'
                                }).round(2)
                                machine_stats.columns = ['Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Event Types']
                                st.dataframe(machine_stats)
                                
                                # Show the complete data
                                st.dataframe(city_events)
                            
                            # Show relevant summary statistics
                            st.markdown("### 📊 Relevant Statistics")
                            if mentioned_event_type:
                                # Event type specific statistics
                                event_type_events = all_events[all_events['event_type'] == mentioned_event_type]
                                st.markdown(f"""
                                <div class="summary-box">
                                    <h3>📊 {mentioned_event_type} Statistics</h3>
                                    <div class="summary-metric">
                                        <strong>Total Events:</strong> {len(event_type_events)}
                                    </div>
                                    <div class="summary-metric">
                                        <strong>Machines:</strong> {event_type_events['machine_name'].nunique()}
                                    </div>
                                    <div class="summary-metric">
                                        <strong>Cities:</strong> {event_type_events['city'].nunique()}
                                    </div>
                                    <div class="summary-metric">
                                        <strong>Average Duration:</strong> {event_type_events['duration_minutes'].mean():.2f} minutes
                                    </div>
                                    <div class="summary-metric">
                                        <strong>Maximum Duration:</strong> {event_type_events['duration_minutes'].max():.2f} minutes
                                    </div>
                                    <div class="summary-metric">
                                        <strong>Minimum Duration:</strong> {event_type_events['duration_minutes'].min():.2f} minutes
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("An error occurred while processing your question. Please try again.")
            else:
                st.warning("Please enter a question")

    elif choice == "🤖 Ask with Ragas":
        st.subheader("🤖 Advanced Question Answering with Ragas")
        st.markdown("""
        This section provides enhanced question answering with quality metrics using Ragas.
        The system will evaluate the response quality and provide detailed metrics.
        """)
        
        # Question input
        user_query = st.text_area("Enter your manufacturing question", height=100)
        
        if st.button("Get Ragas-Powered Answer"):
            if not user_query:
                st.warning("Please enter a question")
                return
                
            try:
                # Get limited events data for faster processing
                all_events = fetch_df(conn, """
                    SELECT 
                        id,
                        event_type,
                        machine_name,
                        notes,
                        timestamp,
                        duration_minutes,
                        city
                    FROM manufacturing_events 
                    ORDER BY timestamp DESC
                    LIMIT 50
                """)
                
                if all_events.empty:
                    st.warning("No events found in the database. Please ensure data is available.")
                    return
                
                # Prepare context from events (limit to most relevant)
                context = "\n".join([
                    f"Event {row['id']}: {row['event_type']} on machine {row['machine_name']} in {row['city']}. {row['notes']} (Duration: {row['duration_minutes']} minutes, Time: {row['timestamp']})"
                    for _, row in all_events.head(10).iterrows()
                ])
                
                # Define the prompt first
                initial_prompt = f"""Based on the following manufacturing events data, answer the question: {user_query}

                Manufacturing Events:
                {context}

                Please provide a clear and concise answer that:
                1. Directly addresses the question
                2. Uses specific data points from the events
                3. Includes relevant statistics
                4. Provides practical insights
                5. Cites specific events where applicable
                """
                
                with st.spinner("Generating answer..."):
                    # Generate initial response immediately
                    with st.spinner("Analyzing manufacturing data..."):
                        response = cached_generate_content(initial_prompt)
                        
                        # Display internal response immediately
                        st.markdown(f"""
                        <div style="background-color: #2D2D2D; border: 2px solid #4CAF50; border-radius: 8px; padding: 20px; margin: 20px 0;">
                            <h3 style="color: #FFFFFF; margin-bottom: 15px;">🧠 Initial Analysis</h3>
                            <div style="color: #FFFFFF; margin: 15px 0; background-color: #1E1E1E; padding: 15px; border-radius: 5px;">
                                {response}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Handle external searches separately
                    with st.spinner("Enhancing with external knowledge..."):
                        try:
                            web_data = fetch_web_data(user_query)
                            if web_data['wikipedia'] or web_data['google']:
                                external_knowledge = ""
                                if web_data['wikipedia']:
                                    external_knowledge += "\n\nRelevant Wikipedia Information:\n"
                                    for article in web_data['wikipedia']:
                                        external_knowledge += f"\n• {article['title']}:\n"
                                        external_knowledge += f"  {article['summary']}\n"
                                        external_knowledge += f"  Source: {article['url']}\n"
                                
                                if web_data['google']:
                                    external_knowledge += "\n\nAdditional Web Resources:\n"
                                    for result in web_data['google']:
                                        external_knowledge += f"\n• {result['title']}:\n"
                                        external_knowledge += f"  {result['description']}\n"
                                        external_knowledge += f"  Source: {result['url']}\n"
                                
                                final_prompt = f"""Based on the manufacturing data and external knowledge, provide a comprehensive answer to: {user_query}

                                Manufacturing Data Analysis:
                                {response}

                                External Knowledge:
                                {external_knowledge}

                                Please provide a well-structured answer that:
                                1. Starts with a direct answer to the question
                                2. Integrates relevant information from external sources
                                3. Supports the answer with specific data points
                                4. Includes key statistics
                                5. Provides practical insights
                                6. Cites sources where applicable
                                """
                                
                                final_answer = cached_generate_content(final_prompt)
                                
                                st.markdown(f"""
                                <div style="background-color: #2D2D2D; border: 2px solid #4CAF50; border-radius: 8px; padding: 20px; margin: 20px 0;">
                                    <h3 style="color: #FFFFFF; margin-bottom: 15px;">🧠 Enhanced Analysis</h3>
                                    <div style="color: #FFFFFF; margin: 15px 0; background-color: #1E1E1E; padding: 15px; border-radius: 5px;">
                                        {final_answer}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if web_data['wikipedia'] or web_data['google']:
                                    st.markdown("### 📚 Sources")
                                    if web_data['wikipedia']:
                                        st.markdown("#### Wikipedia Articles")
                                        for article in web_data['wikipedia']:
                                            st.markdown(f"- [{article['title']}]({article['url']})")
                                    
                                    if web_data['google']:
                                        st.markdown("#### Web Sources")
                                        for result in web_data['google']:
                                            st.markdown(f"- [{result['title']}]({result['url']})")
                        except Exception as web_error:
                            st.info("Could not fetch external knowledge. Using internal analysis only.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with a different question or check your input.")

    elif choice == "🤖 AI Agent Assistant":
        st.subheader("🤖 AI Agent Assistant")
        st.markdown("""
        This section provides an AI agent that can help you with manufacturing-related questions.
        The agent can access both internal data and external knowledge sources.
        """)
        
        # Question input
        user_query = st.text_area("Enter your manufacturing question", height=100)
        
        if st.button("Ask AI Agent"):
            if not user_query:
                st.warning("Please enter a question")
                return
                
            try:
                # Get limited events data for faster processing
                all_events = fetch_df(conn, """
                    SELECT 
                        id,
                        event_type,
                        machine_name,
                        notes,
                        timestamp,
                        duration_minutes,
                        city
                    FROM manufacturing_events 
                    ORDER BY timestamp DESC
                    LIMIT 30
                """)
                
                if all_events.empty:
                    st.warning("No events found in the database. Please ensure data is available.")
                    return
                
                # Prepare context from events (limit to most relevant)
                context = "\n".join([
                    f"Event {row['id']}: {row['event_type']} on machine {row['machine_name']} in {row['city']}. {row['notes']} (Duration: {row['duration_minutes']} minutes, Time: {row['timestamp']})"
                    for _, row in all_events.head(10).iterrows()
                ])
                
                # Define the prompt first
                prompt = f"""Based on the following manufacturing events data, answer the question: {user_query}

                Manufacturing Events:
                {context}

                Please provide a clear and concise answer that:
                1. Directly addresses the question
                2. Uses specific data points from the events
                3. Includes relevant statistics
                4. Provides practical insights
                5. Cites specific events where applicable
                """
                
                with st.spinner("Generating answer..."):
                    # Generate initial response immediately
                    with st.spinner("Analyzing manufacturing data..."):
                        response = cached_generate_content(prompt)
                        
                        # Display internal response immediately
                        st.markdown(f"""
                        <div style="background-color: #2D2D2D; border: 2px solid #4CAF50; border-radius: 8px; padding: 20px; margin: 20px 0;">
                            <h3 style="color: #FFFFFF; margin-bottom: 15px;">🧠 Initial Analysis</h3>
                            <div style="color: #FFFFFF; margin: 15px 0; background-color: #1E1E1E; padding: 15px; border-radius: 5px;">
                                {response}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Start external searches in background
                    with st.spinner("Enhancing with external knowledge..."):
                        try:
                            web_data = fetch_web_data(user_query)
                            
                            if web_data['wikipedia'] or web_data['google']:
                                external_knowledge = ""
                                
                                if web_data['wikipedia']:
                                    external_knowledge += "\n\nRelevant Wikipedia Information:\n"
                                    for article in web_data['wikipedia']:
                                        external_knowledge += f"\n• {article['title']}:\n"
                                        external_knowledge += f"  {article['summary']}\n"
                                        external_knowledge += f"  Source: {article['url']}\n"
                                
                                if web_data['google']:
                                    external_knowledge += "\n\nAdditional Web Resources:\n"
                                    for result in web_data['google']:
                                        external_knowledge += f"\n• {result['title']}:\n"
                                        external_knowledge += f"  {result['description']}\n"
                                        external_knowledge += f"  Source: {result['url']}\n"
                                
                                # Generate final answer with external knowledge
                                final_prompt = f"""Based on the manufacturing data and external knowledge, provide a comprehensive answer to: {user_query}

                                Manufacturing Data Analysis:
                                {response}

                                External Knowledge:
                                {external_knowledge}

                                Please provide a well-structured answer that:
                                1. Starts with a direct answer to the question
                                2. Integrates relevant information from external sources
                                3. Supports the answer with specific data points
                                4. Includes key statistics
                                5. Provides practical insights
                                6. Cites sources where applicable
                                """
                                
                                final_answer = cached_generate_content(final_prompt)
                                
                                # Display the enhanced answer
                                st.markdown(f"""
                                <div style="background-color: #2D2D2D; border: 2px solid #4CAF50; border-radius: 8px; padding: 20px; margin: 20px 0;">
                                    <h3 style="color: #FFFFFF; margin-bottom: 15px;">🧠 Enhanced Analysis</h3>
                                    <div style="color: #FFFFFF; margin: 15px 0; background-color: #1E1E1E; padding: 15px; border-radius: 5px;">
                                        {final_answer}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display sources if available
                                if web_data['wikipedia'] or web_data['google']:
                                    st.markdown("### 📚 Sources")
                                    if web_data['wikipedia']:
                                        st.markdown("#### Wikipedia Articles")
                                        for article in web_data['wikipedia']:
                                            st.markdown(f"- [{article['title']}]({article['url']})")
                                    
                                    if web_data['google']:
                                        st.markdown("#### Web Sources")
                                        for result in web_data['google']:
                                            st.markdown(f"- [{result['title']}]({result['url']})")
                        except Exception as web_error:
                            st.info("Could not fetch external knowledge. Using internal analysis only.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with a different question or check your input.")

if __name__ == "__main__":
    main()
    