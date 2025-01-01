from groq import Groq

# API istemcisi oluştur
client = Groq(api_key = "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")

# LLM tamamlayıcı fonksiyonu
def get_completion_3_1(messages: list, model="llama-3.1-70b-versatile"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,  # stream yerine False kullanıldı çünkü dönen yanıt işleniyor.
        stop=None,
    )
    return response.choices[0].message.content  # Yanıtın doğru kısmı alınıyor.






# LangChain yapılandırması
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Çıktı şeması
summary_schema = ResponseSchema(
    name="score",
    description="A score between 1 and 10 that evaluates the summary quality based on the given content."
)

response_schemas = [summary_schema]

# Yapılandırılmış çıktı ayrıştırıcı
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Biçim talimatlarını al
format_instructions = output_parser.get_format_instructions()

# Prompt şablonu
content_template = """\
Evaluate the given summary based on the provided content. Assign a score between 1 and 10:

Content: {content}

Summary: {summary}

{format_instructions}
"""

# Test edilecek içerik ve özet
test_content = """
 Here's the thing there's this girl (17)that I 
(17) like and have liked for a few years I'm good friends with her and I've known she has liked me recently in the past and I want to ask her out but I'm to afraid of ruining anything I'm not sure if she likes me right now but it is possible I've really liked her for three years now and she knows that I have liked her in the past its possible that we booth like each other right know but don't know it   She always brings the best out of me and I've always liked her a little bit even when I was dating other girls a I've never felt This way about a girl in the past she always brings the best out in me and I always have a good time with her but I don't want to ruin anything cause we are great friends but I still want to date her and have something lasting with her.
"""
test_summary = """
The speaker has been friends with a girl (17) for three years and has developed romantic feelings for her, but is hesitant to ask her out due to fear of ruining their friendship.
"""

# Prompt oluştur
prompt = ChatPromptTemplate.from_template(template=content_template)
message = prompt.format_messages(content=test_content, summary=test_summary, format_instructions=format_instructions)
print(message)
messages = [{"role": "user", "content": message[0].content}]
# Modeli çağır ve yanıtı al
response = get_completion_3_1(messages)

# Yanıtı ayrıştır
output_dict = output_parser.parse(response)

# Skoru al
score = output_dict.get('score')
print(f"Summary Score: {score}")
