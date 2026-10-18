from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

groq_client = Groq()

"""
>>> for t in df.target_label.unique():
...     print(t, '-->', len(df[df['target_label'] == t]))

    HTTP Status --> 1017
    Critical Error --> 161
    Security Alert --> 371
    Error --> 177
    System Notification --> 356
    Resource Usage --> 177
    User Action --> 144
    Workflow Error --> 4
    Deprecation Warning --> 3
"""

def classify_with_llm(log_msg):
    """
    Generate a variant of the input sentence. For example,
    If input sentence is "User session timed out unexpectedly, user ID: 9250.",
    variant would be "Session timed out for user 9251"
    """
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error,
    (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        # model='deepseek-r1-distill-llama-70b',
        model="openai/gpt-oss-120b",
        temperature=0.5
    )

    content = chat_completion.choices[0].message.content
    match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1)

    return category

if __name__ == "__main__":
    print(classify_with_llm("System reboot initiated by user 12345."))
    print(
        classify_with_llm(
            """
            The model `deepseek-r1-distill-llama-70b` has been decommissioned and 
            is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation 
            on which model to use instead.
            """
        )
    )