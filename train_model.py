from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[reportMissingModuleSource]
from sklearn.naive_bayes import MultinomialNB  # type: ignore[reportMissingModuleSource]
from sklearn.pipeline import make_pipeline  # type: ignore[reportMissingModuleSource]
import pickle

data = {
    'text': [
        # 1-10
        'Hello world', 'How are you', 
        'Bonjour le monde', 'Comment ça va', 
        'Hola mundo', '¿Cómo estás?', 
        'Hallo Welt', 'Wie geht es dir', 
        'Ciao mondo', 'Come stai', 
        'Olá mundo', 'Como você está', 
        'Hallo wereld', 'Hoe gaat het', 
        'Привет мир', 'Как дела', 
        'こんにちは世界', 'お元気ですか', 
        '안녕 세상', '어떻게 지내세요', 
        # 11-20
        '你好世界', '你最近怎么样', 
        'مرحبا بالعالم', 'كيف حالك', 
        'नमस्ते दुनिया', 'आप कैसे हैं', 
        'ওহে বিশ্ব', 'তুমি কেমন আছ', 
        'ہیلو دنیا', 'آپ کیسے ہیں', 
        'Merhaba dünya', 'Nasılsın', 
        'Chào thế giới', 'Bạn khỏe không', 
        'สวัสดีชาวโลก', 'คุณเป็นอย่างไรบ้าง', 
        'Γειά σου Κόσμε', 'Πώς είσαι', 
        'Witaj świecie', 'Jak się masz', 
        # 21-30
        'Hej världen', 'Hur mår du', 
        'Hej Verden', 'Hvordan har du det', 
        'Hei maailma', 'Mitä kuuluu', 
        'Hei Verden', 'Hvordan går det', 
        'Ahoj světe', 'Jak se máš', 
        'Helló Világ', 'Hogy vagy', 
        'Salut Lume', 'Ce mai faci', 
        'Ahoj svet', 'Ako sa máš', 
        'Привіт світе', 'Як справи', 
        'Здравей свят', 'Как си', 
        # 31-40
        'Здраво свете', 'Како си', 
        'Pozdrav svijete', 'Kako si', 
        'Zdravo svijete', 'Kako si', 
        'Pozdravljen svet', 'Kako si', 
        'Здраво свету', 'Како си', 
        'Labas pasauli', 'Kaip laikaisi', 
        'Sveika pasaule', 'Kā tev iet', 
        'Tere maailm', 'Kuidas läheb', 
        'Halló heimur', 'Hvernig hefurðu það', 
        'Dia duit ar an domhan', 'Conas atá tú', 
        # 41-50
        'Helo Byd', 'Sut wyt ti', 
        'Salamu Dunia', 'Habari gani', 
        'Hallo wêreld', 'Hoe gaan dit', 
        'שלום עולם', 'מה שלומך', 
        'سلام دنیا', 'حالت چطوره', 
        'வணக்கம் உலகம்', 'நீங்கள் எப்படி இருக்கிறீர்கள்', 
        'హలో వరల్డ్', 'మీరు ఎలా ఉన్నారు', 
        'ഹലോ ലോകം', 'സുഖമാണോ', 
        'ಹಲೋ ವರ್ಲ್ಡ್', 'ನೀವು ಹೇಗಿದ್ದೀರಿ', 
        'हॅलो वर्ल्ड', 'तू कसा आहेस'
    ],
    'language': [
        'English', 'English', 'French', 'French', 'Spanish', 'Spanish', 'German', 'German', 
        'Italian', 'Italian', 'Portuguese', 'Portuguese', 'Dutch', 'Dutch', 'Russian', 'Russian', 
        'Japanese', 'Japanese', 'Korean', 'Korean', 'Chinese', 'Chinese', 'Arabic', 'Arabic', 
        'Hindi', 'Hindi', 'Bengali', 'Bengali', 'Urdu', 'Urdu', 'Turkish', 'Turkish', 
        'Vietnamese', 'Vietnamese', 'Thai', 'Thai', 'Greek', 'Greek', 'Polish', 'Polish', 
        'Swedish', 'Swedish', 'Danish', 'Danish', 'Finnish', 'Finnish', 'Norwegian', 'Norwegian', 
        'Czech', 'Czech', 'Hungarian', 'Hungarian', 'Romanian', 'Romanian', 'Slovak', 'Slovak', 
        'Ukrainian', 'Ukrainian', 'Bulgarian', 'Bulgarian', 'Serbian', 'Serbian', 'Croatian', 'Croatian', 
        'Bosnian', 'Bosnian', 'Slovenian', 'Slovenian', 'Macedonian', 'Macedonian', 'Lithuanian', 'Lithuanian', 
        'Latvian', 'Latvian', 'Estonian', 'Estonian', 'Icelandic', 'Icelandic', 'Irish', 'Irish', 
        'Welsh', 'Welsh', 'Swahili', 'Swahili', 'Afrikaans', 'Afrikaans', 'Hebrew', 'Hebrew', 
        'Persian', 'Persian', 'Tamil', 'Tamil', 'Telugu', 'Telugu', 'Malayalam', 'Malayalam', 
        'Kannada', 'Kannada', 'Marathi', 'Marathi'
    ]
}

print("Training model with 50 languages...")
model_pipeline = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1, 3)), MultinomialNB())
model_pipeline.fit(data['text'], data['language'])

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)
print("Model trained and saved as model.pkl")