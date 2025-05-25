
import openai

# Initialize Wikipedia API

def get_wine_quality_info():
    
    if not page.exists():
        return "No information available."

    return page.summary  # Return the summary of the page

def summarize_text(text, max_tokens=150):
    # OpenAI GPT-based summarization
    openai.api_key = 'your-openai-api-key'

    response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=max_tokens
    )

    return response.choices[0].text.strip()

# Display the information and summarize it if it's too long
def show_wine_quality_info():
    full_info = get_wine_quality_info()
    if len(full_info.split()) > 100:  # If the information is too long, summarize it
        summary = summarize_text(full_info)
        return summary
    else:
        return full_info
