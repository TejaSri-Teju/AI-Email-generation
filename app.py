from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models for summarization and text generation
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
generator = pipeline('text-generation', model='gpt2')

def scrape(url):
    try:
        if url:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                text = []
                # <p> tags
                p_tags = []
                max_tags = 20

                names = soup.find_all("p")
                for i, tag in enumerate(names):
                    if i >= max_tags:
                        break  # Stop after extracting the first 10 tags
                    if tag.text != '':
                        name = tag.text
                        p_tags.append(name)
                text += [str(item) for item in p_tags]

                # Return the cleaned text
                return " ".join(text)
        return ""
    except Exception as e:
        print(f"An error occurred while scraping {url}: {str(e)}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    # Read the uploaded CSV file
    df = pd.read_csv(file.file)
    results = []
    for _, row in df.iterrows():
        url = row['Website']
        scraped_text = scrape(url)
        row_data = row.to_dict()
        row_data['ScrapedText'] = scraped_text
        email_content = generate_email(row_data)
        results.append(email_content)
    
    return templates.TemplateResponse("result.html", {"request": request, "emails": results})

def generate_email(data):
    # Summarize the scraped text
    summarized_text = summarizer(data['ScrapedText'], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    draft_pitch = """
    Dear {RecipientName},
    I hope this message finds you well. I am reaching out to introduce our services at {CompanyName}. {ScrapedText}
    """
    template = draft_pitch.format(
        RecipientName=data['RecipientName'],
        CompanyName=data['CompanyName'],
        ScrapedText=summarized_text
    )
    response = generator(text_inputs=template, num_return_sequences=1, temperature=0.7)

    return response[0]['generated_text']

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

