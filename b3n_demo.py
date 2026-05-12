from openai import OpenAI

key = "sk-oUfqC3eduj7x7d8D972095C533F54a889c9f7608FdC905Ce"
url = "https://gcisdbuhxjlm.sealosbja.site/v1"
client = OpenAI(api_key=key, base_url=url)

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion)